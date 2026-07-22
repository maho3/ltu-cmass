"""Harvester: converts FastPM snapshots from producer jobs into nbody.h5.

Producers run cmass.nbody.fastpm with nbody.postprocess=False
nbody.harvest=True and a fixed shared meta.scratchdir. Each harvester job
scans its shard of per-lhid workdirs and converts snapshots IN PLACE on the
shared scratch (single serial converter, no copying to node-local disk):
snapshot complete -> process_single_snapshot() reads it from scratch, writes
nbody_{a}.h5 beside it, deletes the 47GB raw. When all snapshots of an lhid
are converted and the producer is done, they are concatenated to
<outbase>/<lhid>/nbody.h5 and the scratch dir is removed.

Scale by running N sharded instances (lhid % N == K); each lhid has exactly
one owner, so no locking. Everything is idempotent and restart-safe: state
lives entirely in the shared filesystem.

Producer job scripts touch .fastpm_done / .fastpm_failed in each lhid
workdir; failed producers are skipped.

Example (one shard of 10):
    PYTHONPATH=. python scripts/harvest_fastpm.py \
        --scratchbase /work/hdd/bdne/maho3/cmass_scratch/harvest \
        --outbase /work/hdd/bdne/maho3/cmass-ili/mtnglike/fastpm/L3000-N384 \
        --lhids 3000 3100 --shard 0 10
"""
import argparse
import logging
import os
import shutil
import subprocess
import time
import h5py
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


class LhidState:
    def __init__(self, lhid, scratchbase, outbase, cfg):
        self.lhid = lhid
        self.work = join(scratchbase, 'mtnglike', 'fastpm',
                         'L3000-N384', str(lhid))
        self.outdir = join(outbase, str(lhid))
        self.cfg = cfg
        self.concat_done = exists(join(self.outdir, 'nbody.h5'))

    def producer_done(self):
        return exists(join(self.work, '.fastpm_done'))

    def producer_failed(self):
        return exists(join(self.work, '.fastpm_failed'))

    def done_h5(self, a):
        return exists(join(self.work, f'nbody_{a:.4f}.h5'))

    def convertible(self):
        """Snapshots complete in scratch and not yet converted. Snapshot i is
        complete once snapshot i+1's dir exists (FastPM writes sequentially)
        or the producer has exited."""
        out = []
        for i, a in enumerate(ASAVE):
            src = join(self.work, snapname(a))
            if not isdir(src) or self.done_h5(a):
                continue
            nxt = i + 1 < len(ASAVE) and isdir(join(self.work,
                                                    snapname(ASAVE[i+1])))
            if nxt or self.producer_done():
                out.append(a)
        return out

    def ready_to_concat(self):
        return (not self.concat_done and self.producer_done()
                and all(self.done_h5(a) for a in ASAVE))


def producers_alive(jobname):
    """True if any producer job is still queued or running.

    Queue waits here reach ~10h, so a harvester that idle-exits while
    producers are merely PENDING leaves them writing 47GB snapshots with
    nothing draining scratch -- this repeatedly filled the disk quota. On any
    squeue failure we return True: 'unknown' must never be read as 'no
    producers left'."""
    try:
        out = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h',
             '-n', jobname, '-o', '%i'],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return True
        return bool(out.stdout.strip())
    except Exception:
        return True


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
    shutil.rmtree(st.work, ignore_errors=True)
    st.concat_done = True
    logging.info(f'HARVESTED lhid={st.lhid} -> {st.outdir}/nbody.h5')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratchbase', required=True)
    ap.add_argument('--outbase', required=True)
    ap.add_argument('--lhids', type=int, nargs=2, required=True,
                    help='inclusive range, e.g. 3000 3100')
    ap.add_argument('--shard', type=int, nargs=2, default=None,
                    metavar=('K', 'N'),
                    help='process only lhids with lhid %% N == K')
    ap.add_argument('--idle-timeout-min', type=float, default=120)
    ap.add_argument('--producer-name', default='3gpch_prod',
                    help='SLURM job name of the producers; the harvester will '
                         'not idle-exit while any are queued or running')
    ap.add_argument('--poll-sec', type=float, default=15)
    args = ap.parse_args()

    lhid_list = list(range(args.lhids[0], args.lhids[1] + 1))
    if args.shard is not None:
        k, n = args.shard
        lhid_list = [l for l in lhid_list if l % n == k]
        logging.info(f'shard {k}/{n}: {len(lhid_list)} lhids')

    states = {}
    for lhid in lhid_list:
        cfg = OmegaConf.create(
            {'nbody': {'B': B, 'L': 3000, 'N': 384, 'lhid': lhid,
                       'asave': ASAVE,
                       'cosmo': [0.3089, 0.0486, 0.6774, 0.9667, 0.8159]}})
        states[lhid] = LhidState(lhid, args.scratchbase, args.outbase, cfg)

    last_action = time.time()
    while True:
        pending = [s for s in states.values()
                   if not s.concat_done and not s.producer_failed()]
        if not pending:
            logging.info('all lhids harvested')
            break

        worked = False
        for st in pending:
            for a in st.convertible():
                # one snapshot at a time, directly on shared scratch
                process_single_snapshot(st.cfg, st.work, a, delete_files=True)
                logging.info(f'converted lhid={st.lhid} a={a:.4f}')
                worked = True
            if st.ready_to_concat():
                concat(st)
                worked = True

        failed = [s.lhid for s in states.values() if s.producer_failed()]
        if failed:
            logging.warning(f'skipping failed producers: {failed}')

        if worked:
            last_action = time.time()
        elif time.time() - last_action > args.idle_timeout_min * 60:
            if producers_alive(args.producer_name):
                logging.info(
                    f'idle {args.idle_timeout_min:.0f}min but producers still '
                    'queued/running; staying alive')
                last_action = time.time()
                time.sleep(args.poll_sec)
            else:
                logging.warning('idle timeout, no live producers; exiting '
                                '(rerun to resume)')
                break
        else:
            time.sleep(args.poll_sec)

    done = sorted(l for l, s in states.items() if s.concat_done)
    logging.info(f'harvested lhids: {done}')


if __name__ == '__main__':
    main()
