"""Harvester: converts FastPM snapshots from producer jobs into nbody.h5.

Producers run cmass.nbody.fastpm with nbody.postprocess=False
nbody.harvest=True and a fixed shared meta.scratchdir. This job scans their
per-lhid workdirs, moves finished snapshots to node-local disk, converts them
(same process_single_snapshot as the pipeline), concatenates each completed
lhid into <outbase>/<lhid>/nbody.h5, and cleans up.

Priority: move (drain quota'd shared scratch) while local disk < cap;
convert otherwise. Producer job scripts touch .fastpm_done / .fastpm_failed
in each lhid workdir.

Example:
    PYTHONPATH=. python scripts/harvest_fastpm.py \
        --scratchbase /work/hdd/bdne/maho3/cmass_scratch/harvest \
        --localbase /tmp/maho3/harvest \
        --outbase /work/hdd/bdne/maho3/cmass-ili/mtnglike/fastpm/L3000-N384 \
        --lhids 3000 3014
"""
import argparse
import logging
import os
import shutil
import time
import h5py
import multiprocessing as mp
from os.path import join, isdir, exists

from omegaconf import OmegaConf
from cmass.nbody.fastpm import process_single_snapshot

# Snapshot list comes from the suite config so harvester and producers can
# never disagree about which snapshots make a complete lhid
_NBODY_CONF = join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'cmass', 'conf', 'nbody', 'mtnglike.yaml')
ASAVE = list(OmegaConf.load(_NBODY_CONF).asave)
B = int(OmegaConf.load(_NBODY_CONF).B)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s-HARVEST] %(message)s',
                    datefmt='%H:%M:%S')


def snapname(a):
    return f'fastpm_B{B}_{a:.4f}'


def du_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(join(root, f))
            except OSError:
                pass
    return total / 1024**3


class LhidState:
    def __init__(self, lhid, scratchbase, localbase, outbase, cfg):
        self.lhid = lhid
        self.work = join(scratchbase, 'mtnglike', 'fastpm',
                         'L3000-N384', str(lhid))
        self.local = join(localbase, str(lhid))
        self.outdir = join(outbase, str(lhid))
        self.cfg = cfg
        self.concat_done = exists(join(self.outdir, 'nbody.h5'))

    def producer_done(self):
        return exists(join(self.work, '.fastpm_done'))

    def producer_failed(self):
        return exists(join(self.work, '.fastpm_failed'))

    def done_h5(self, a):
        # converted files are parked on shared scratch (survive restarts)
        return exists(join(self.work, f'nbody_{a:.4f}.h5'))

    def movable(self):
        """Snapshots complete in scratch: next snapshot exists or producer done."""
        out = []
        for i, a in enumerate(ASAVE):
            src = join(self.work, snapname(a))
            if (not isdir(src) or self.done_h5(a)
                    or exists(join(self.local, f'.moved_{a:.4f}'))):
                continue
            nxt = i + 1 < len(ASAVE) and isdir(join(self.work,
                                                    snapname(ASAVE[i+1])))
            if nxt or self.producer_done():
                out.append(a)
        return out

    def convertible(self):
        return [a for a in ASAVE
                if exists(join(self.local, f'.moved_{a:.4f}'))
                and not self.done_h5(a)]

    def converted(self):
        return [a for a in ASAVE if self.done_h5(a)]

    def ready_to_concat(self):
        return (not self.concat_done and self.producer_done()
                and len(self.converted()) == len(ASAVE))


def do_move(args):
    """Copy a snapshot to local disk. The scratch copy is kept until
    conversion succeeds (do_convert deletes it), so a harvester crash or
    restart never loses data — node-local /tmp is ephemeral."""
    work, local, a = args
    src, dst = join(work, snapname(a)), join(local, snapname(a))
    marker = join(local, f'.moved_{a:.4f}')
    os.makedirs(local, exist_ok=True)
    if not isdir(dst):
        shutil.copytree(src, dst + '.tmp', dirs_exist_ok=True)
        os.rename(dst + '.tmp', dst)
    open(marker, 'w').close()
    logging.info(f'copied {os.path.basename(work)} a={a:.4f}')
    return a


def do_convert(args):
    cfg, work, local, a = args
    process_single_snapshot(cfg, local, a, delete_files=True)
    # Park the small (~1GB) converted file on SHARED scratch before dropping
    # the 47GB scratch original: after this, nothing irreplaceable lives on
    # the ephemeral local disk, and quota is freed.
    lf, wf = join(local, f'nbody_{a:.4f}.h5'), join(work, f'nbody_{a:.4f}.h5')
    if not exists(wf):
        shutil.copyfile(lf, wf + '.tmp')
        os.replace(wf + '.tmp', wf)
    src = join(work, snapname(a))
    if isdir(src):
        shutil.rmtree(src)
    logging.info(f'converted lhid={cfg.nbody.lhid} a={a:.4f}')
    return a


def concat(st):
    os.makedirs(st.outdir, exist_ok=True)
    tmp = join(st.outdir, 'nbody.h5.tmp')
    with h5py.File(tmp, 'w') as outfile:
        for a in sorted(ASAVE):
            with h5py.File(join(st.work, f'nbody_{a:.4f}.h5'), 'r') as f:
                group = outfile.create_group(f'{a:.6f}')
                group.create_dataset('rho', data=f['rho'][:])
                group.create_dataset('fvel', data=f['fvel'][:])
    os.replace(tmp, join(st.outdir, 'nbody.h5'))
    shutil.rmtree(st.local, ignore_errors=True)
    shutil.rmtree(st.work, ignore_errors=True)
    st.concat_done = True
    logging.info(f'HARVESTED lhid={st.lhid} -> {st.outdir}/nbody.h5')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratchbase', required=True)
    ap.add_argument('--localbase', required=True)
    ap.add_argument('--outbase', required=True)
    ap.add_argument('--lhids', type=int, nargs=2, required=True,
                    help='inclusive range, e.g. 3000 3014')
    ap.add_argument('--shard', type=int, nargs=2, default=None,
                    metavar=('K', 'N'),
                    help='process only lhids with lhid %% N == K, for '
                         'running N independent harvesters in parallel')
    ap.add_argument('--nmove', type=int, default=6)
    ap.add_argument('--nconvert', type=int, default=5)
    ap.add_argument('--local-cap-gb', type=float, default=500)
    ap.add_argument('--idle-timeout-min', type=float, default=120)
    ap.add_argument('--poll-sec', type=float, default=15)
    args = ap.parse_args()

    lhid_list = list(range(args.lhids[0], args.lhids[1] + 1))
    if args.shard is not None:
        k, n = args.shard
        lhid_list = [l for l in lhid_list if l % n == k]
        logging.info(f'shard {k}/{n}: {len(lhid_list)} lhids')

    os.makedirs(args.localbase, exist_ok=True)
    states = {}
    for lhid in lhid_list:
        cfg = OmegaConf.create(
            {'nbody': {'B': B, 'L': 3000, 'N': 384, 'lhid': lhid,
                       'asave': ASAVE,
                       'cosmo': [0.3089, 0.0486, 0.6774, 0.9667, 0.8159]}})
        states[lhid] = LhidState(
            lhid, args.scratchbase, args.localbase, args.outbase, cfg)

    mv_pool = mp.Pool(args.nmove)
    cv_pool = mp.Pool(args.nconvert)
    mv_busy, cv_busy = {}, {}   # (lhid, a) -> AsyncResult
    last_action = time.time()

    while True:
        pending = [s for s in states.values()
                   if not s.concat_done and not s.producer_failed()]
        if not pending and not mv_busy and not cv_busy:
            logging.info('all lhids harvested')
            break

        # reap finished tasks
        for d in (mv_busy, cv_busy):
            for key in [k for k, r in d.items() if r.ready()]:
                d.pop(key).get()  # raise on worker error
                last_action = time.time()

        local_gb = du_gb(args.localbase)
        for st in pending:
            # concat when complete
            if st.ready_to_concat() and not any(
                    k[0] == st.lhid for k in list(mv_busy) + list(cv_busy)):
                concat(st)
                last_action = time.time()
                continue
            # moves first while local disk has room
            if local_gb < args.local_cap_gb and len(mv_busy) < args.nmove:
                for a in st.movable():
                    key = (st.lhid, a)
                    if key not in mv_busy and len(mv_busy) < args.nmove:
                        mv_busy[key] = mv_pool.apply_async(
                            do_move, ((st.work, st.local, a),))
            # conversions
            if len(cv_busy) < args.nconvert:
                for a in st.convertible():
                    key = (st.lhid, a)
                    if (key not in cv_busy and key not in mv_busy
                            and len(cv_busy) < args.nconvert):
                        cv_busy[key] = cv_pool.apply_async(
                            do_convert, ((st.cfg, st.work, st.local, a),))

        failed = [s.lhid for s in states.values() if s.producer_failed()]
        if failed:
            logging.warning(f'skipping failed producers: {failed}')

        if time.time() - last_action > args.idle_timeout_min * 60:
            logging.warning('idle timeout; exiting (rerun to resume)')
            break
        time.sleep(args.poll_sec)

    mv_pool.close()
    cv_pool.close()
    done = sorted(l for l, s in states.items() if s.concat_done)
    logging.info(f'harvested lhids: {done}')


if __name__ == '__main__':
    main()
