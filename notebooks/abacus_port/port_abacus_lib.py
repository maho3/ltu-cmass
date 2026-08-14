"""Shared logic for porting AbacusSummit `base` cleaned-CompaSO halos (z=0.5) into
ltu-cmass format, with an NFW-fit c200c concentration. Used by both port_abacus.ipynb
(interactive / single-sim) and port_abacus_one.py (SLURM array, one sim per task).

Output per lhid: {OUT}/halos.h5, group '0.666667', datasets
  pos  [cMpc/h, 0..2000]   vel [physical km/s]   mass [log10 Msun/h]   concentration [c200c]
"""
import os
import shutil
import pickle
import tarfile
from glob import glob
from os.path import join, exists, dirname
import numpy as np
import pandas as pd
import yaml
import h5py
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

# ------------------------------------------------------------------ config
TAR       = '/anvil/scratch/x-mho1/abacus/z0.500_base.tar'             # c001-c181 (114 sims)
TAR_C000  = '/anvil/scratch/x-mho1/abacus/z0.500_base_c000.tar'        # c000, 25 phases (ph000-ph024)
TARS      = [TAR, TAR_C000]
INDEX_PKL = '/anvil/scratch/x-mho1/abacus/z0.500_base.index.pkl'      # cached multi-tar member index
TABLE     = join(dirname(__file__), 'abacus_custom_table.csv')         # LHID -> SimName
OUT_TMPL  = '/anvil/scratch/x-mho1/cmass-ili/abacus/nbodyall/L2000-N256/{lhid}'
TMPROOT   = '/anvil/scratch/x-mho1/abacus/port_tmp'                    # scratch for extracted slabs

ZDIR = 'z0.500'
Z    = 0.5
A    = 1.0 / (1.0 + Z)          # 0.666667 -> halos.h5 group key
MMIN = 5e12                     # CHARM mass cut [Msun/h]

FRAC    = np.array([.10, .25, .33, .50, .67, .75, .90, .95, .98])
RFIELDS = ['r10_L2com', 'r25_L2com', 'r33_L2com', 'r50_L2com', 'r67_L2com',
           'r75_L2com', 'r90_L2com', 'r95_L2com', 'r98_L2com']

_tab = pd.read_csv(TABLE)
SIMNAME = dict(zip(_tab['LHID'].astype(int), _tab['SimName']))          # lhid -> AbacusSummit_base_cXXX_phYYY


# ------------------------------------------------------------------ per-lhid cosmology + config.yaml
def cosmo_for_lhid(lhid):
    """[Om, Ob, h, ns, sigma8] for this lhid, from abacus_custom_table.csv's own columns
    (same rescaling as notebooks/port_abacus_cosmologies.ipynb's 'since 5/2/25' section:
    Om,Ob rescaled from omega_x/h^2). Verified to reproduce the true CompaSO-header
    cosmology (e.g. lhid 6 -> Om=0.276126, matching AbacusSummit_base_c001_ph000 exactly)."""
    row = _tab[_tab['LHID'] == lhid].iloc[0]
    h = float(row['h'])
    Om = (row['omega_cdm'] + row['omega_b'] + row['omega_ncdm']) / h**2
    Ob = row['omega_b'] / h**2
    return [float(Om), float(Ob), h, float(row['n_s']), float(row['sigma8_m'])]


def write_config(outdir, lhid):
    """Write a provenance config.yaml with meta+nbody sections only (this port never
    runs the downstream bias/HOD/survey pipeline stages, so those sections -- present
    in the sibling abacus/nbody and abacus/custom config.yamls -- are intentionally
    omitted here rather than filled with fabricated values)."""
    zi, zf = 20, 0.5
    cfg = {
        'meta': {
            'wdir': '/anvil/scratch/x-mho1/cmass-ili',
            'logdir': '${meta.wdir}/logs/',
            'cosmofile': join(dirname(__file__), 'abacus_custom_table.csv'),
            'fastpm_exec': '/home/x-mho1/git/fastpm/src/fastpm',
            'pinocchio_exec': '/home/x-mho1/git/Pinocchio/src/pinocchio.x',
        },
        'sim': 'nbody',
        'multisnapshot': False,
        'nbody': {
            'suite': 'abacus',
            'L': 2000,
            'N': 256,
            'lhid': int(lhid),
            'matchIC': True,
            'supersampling': 3,
            'save_particles': False,
            'save_transfer': False,
            'zi': zi,
            'zf': zf,
            'asave': [A],
            'transfer': 'CLASS',
            'order': 2,
            'B': 2,
            'N_steps': 32,
            'COLA': True,
            'mass_function': 'Watson_2013',
            'ai': 1.0 / (1.0 + zi),
            'af': A,
            'quijote': False,
            'cosmo': cosmo_for_lhid(lhid),
            'concentration_method': 'nfw_fit_r10-r98_L2com_c200c',
        },
    }
    with open(join(outdir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------ NFW concentration fit
def _mu(x):
    return np.log1p(x) - x / (1.0 + x)


def fit_c200c(R, N, Mpart, z, Om0, rs_grid=None):
    """Fit M(<r)=A*mu(r/rs) in log-space to (r10..r98, f*N*Mpart), then solve the
    overdensity condition for R200c. Returns c200c (=R200c/rs). Memory: (Nh,9) per grid step."""
    Nh = len(N)
    lnm = np.log(FRAC[None, :] * (N[:, None] * Mpart))
    if rs_grid is None:
        rs_grid = np.geomspace(3e-3, 2.0, 500)          # cMpc/h
    best = np.full(Nh, np.inf); rs_fit = np.empty(Nh); lnA = np.empty(Nh)
    for rs in rs_grid:
        d = lnm - np.log(_mu(R / rs))
        lnA_g = d.mean(axis=1)
        sse = ((d - lnA_g[:, None]) ** 2).sum(axis=1)
        u = sse < best
        best[u] = sse[u]; rs_fit[u] = rs; lnA[u] = lnA_g[u]
    rho_s = np.exp(lnA) / (4 * np.pi * rs_fit ** 3)
    a = 1.0 / (1.0 + z); E2 = Om0 * (1 + z) ** 3 + (1 - Om0); rhoc0 = 2.775e11
    rhs = 200.0 * rhoc0 * E2 * a ** 3 / (3.0 * rho_s)
    yg = np.geomspace(0.3, 80, 6000); fy = _mu(yg) / yg ** 3
    return np.interp(rhs, fy[::-1], yg[::-1])           # c200c = R200c/rs


# ------------------------------------------------------------------ tar random access
def _open_tar_random(tar):
    """Open the tar for random access with readahead DISABLED. On Lustre the default
    readahead prefetches ~10 MB per seek, so a bare tarfile header-scan drags hundreds
    of GB across the wire; POSIX_FADV_RANDOM makes seeks read only the 512-byte headers."""
    f = open(tar, 'rb')
    try:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
    except (AttributeError, OSError):
        pass
    return tarfile.open(fileobj=f, mode='r')


def get_index(tars=TARS, cache=INDEX_PKL):
    """Return (tf_dict, index) across all `tars`: index is {name: (tar_path, TarInfo)}
    for the halo_info + cleaned_halo_info members, tf_dict is {tar_path: open tarfile
    handle}. The index is built once per tar (readahead OFF -> fast header scan) and
    pickled together, then loaded on subsequent calls. Each returned handle is a
    *normal* one (readahead ON) for fast sequential slab extraction."""
    if cache and exists(cache):
        with open(cache, 'rb') as f:
            idx = pickle.load(f)
    else:
        idx = {}
        for tar in tars:
            scan = _open_tar_random(tar)          # readahead OFF: seek reads only headers
            while True:
                m = scan.next()
                if m is None:
                    break
                if m.isfile() and m.name.endswith('.asdf') and (
                        '/halo_info/halo_info_' in m.name
                        or '/cleaned_halo_info/cleaned_halo_info_' in m.name):
                    idx[m.name] = (tar, m)
                scan.members = []
            scan.close()
        if cache:
            with open(cache, 'wb') as f:
                pickle.dump(idx, f)
    tf_dict = {tar: tarfile.open(tar, 'r') for tar in tars}   # readahead ON: fast 2 GB slab reads
    return tf_dict, idx


def extract_sim(lhid, index, tf_dict):
    """Pull this sim's halo_info + cleaned_halo_info slabs into TMPROOT/lhid,
    preserving tar paths so CompaSO auto-detects the cleaning/ dir. Members can come
    from any tar in tf_dict (e.g. the main grid tar or the c000-phases tar)."""
    sim = SIMNAME[lhid]; dst = join(TMPROOT, str(lhid))
    bp = f'{sim}/halos/{ZDIR}/halo_info/halo_info_'
    cp = f'cleaning/{sim}/{ZDIR}/cleaned_halo_info/cleaned_halo_info_'
    for name, (tar, m) in index.items():
        if name.startswith(bp) or name.startswith(cp):
            outp = join(dst, name); os.makedirs(dirname(outp), exist_ok=True)
            with tf_dict[tar].extractfile(m) as s, open(outp, 'wb') as o:
                shutil.copyfileobj(s, o, length=32 * 1024 * 1024)
    return dst, sim


# ------------------------------------------------------------------ read + fit + save
def load_and_fit(dst, sim, verbose=False):
    """Slab-by-slab cleaned CompaSO read (bounded memory) -> pos/vel/mass/c200c past the cut."""
    hdir = join(dst, sim, 'halos', ZDIR, 'halo_info')
    slabs = sorted(glob(join(hdir, 'halo_info_*.asdf')))
    Mpart = z = Om0 = None
    acc = {k: [] for k in ('pos', 'vel', 'mass', 'conc')}
    for fn in slabs:
        cat = CompaSOHaloCatalog(fn, cleaned=True, verbose=False,
                                 fields=['N', 'x_L2com', 'v_L2com'] + RFIELDS)
        h = cat.halos
        if Mpart is None:
            Mpart = cat.header['ParticleMassHMsun']; z = cat.header['Redshift']; Om0 = cat.header['Omega_M']
        N = np.asarray(h['N'], float)
        R = np.stack([np.asarray(h[k], float) for k in RFIELDS], axis=1)
        Mh = N * Mpart
        sel = (Mh >= MMIN) & (N > 0) & np.all(R > 0, axis=1)
        if not sel.any():
            continue
        N, R, Mh = N[sel], R[sel], Mh[sel]
        acc['pos'].append((np.asarray(h['x_L2com'], np.float32)[sel] + 1000.) % 2000.)
        acc['vel'].append(np.asarray(h['v_L2com'], np.float32)[sel])
        acc['mass'].append(np.log10(Mh).astype(np.float32))
        acc['conc'].append(fit_c200c(R, N, Mpart, z, Om0).astype(np.float32))
        if verbose:
            print(f'  {os.path.basename(fn)}: {int(sel.sum())} halos', flush=True)
    return dict(pos=np.concatenate(acc['pos']), vel=np.concatenate(acc['vel']),
                mass=np.concatenate(acc['mass']), conc=np.concatenate(acc['conc']), a=A)


def save_snapshot(outdir, a, hpos, hvel, hmass, **meta):
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(join(outdir, 'halos.h5'), 'a') as f:
        g = f.create_group(f'{a:.6f}')
        g.create_dataset('pos', data=hpos)     # comoving [Mpc/h]
        g.create_dataset('vel', data=hvel)     # physical [km/s]
        g.create_dataset('mass', data=hmass)   # log10 [Msun/h]
        for k, v in meta.items():
            g.create_dataset(k, data=v)


def process_lhid(lhid, index, tf_dict, overwrite=False, keep_tmp=False, verbose=False):
    """Extract -> read+fit -> save -> cleanup for one lhid. Resumable."""
    if lhid not in SIMNAME:
        return ('nomap', lhid, None)
    sim = SIMNAME[lhid]; outdir = OUT_TMPL.format(lhid=lhid)
    if exists(join(outdir, 'halos.h5')) and not overwrite:
        return ('skip', lhid, None)
    if not any(n.startswith(f'{sim}/halos/{ZDIR}/halo_info/') for n in index):
        return ('missing', lhid, None)
    dst, _ = extract_sim(lhid, index, tf_dict)
    try:
        halos = load_and_fit(dst, sim, verbose=verbose)
        fp = join(outdir, 'halos.h5')
        if exists(fp):
            os.remove(fp)
        save_snapshot(outdir, halos['a'], halos['pos'], halos['vel'], halos['mass'],
                      concentration=halos['conc'])
        write_config(outdir, lhid)
    finally:
        if not keep_tmp:
            shutil.rmtree(dst, ignore_errors=True)
    return ('done', lhid, len(halos['mass']))
