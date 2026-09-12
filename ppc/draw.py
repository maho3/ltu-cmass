"""
Step 1 of the posterior predictive check (PPC) campaign.

Draws theta from q(theta | x_obs) at a single fiducial test point, and writes
everything the SLURM stages need to simulate those draws:

    params/ppc_<tag>_cosmo.txt   cosmology rows, indexed by draw id (= lhid)
    <out>/posterior_draws.npz    theta_draws, x_obs, theta_obs, id_obs, ...
    <out>/manifest.tsv           one row per draw
    <out>/overrides/<id>.txt     hydra overrides for the HOD + noise stage

Draws are joint: one theta per simulation, straight from ensemble.sample().
Nothing here is expensive; it runs on the login node.

Out-of-distribution: --testing_suite/--testing_sim draw x_obs from another
suite's test split instead of the training suite's, mirroring infer.testing in
cmass.infer.validate and cmass.infer.resim. The posterior, the forward chain and
the quantile reference pool all stay the training suite's; only the observation
moves. Outputs land under a testing/<suite>_<sim>/ segment so an OOD campaign
cannot collide with the in-distribution one.

--obs_lhid names the observed point outright instead of taking the most central
one. Suites like abacus mix LCDM, massive-neutrino and non-LCDM cosmologies in
one test set, and the forward chain reproduces only the five LCDM parameters, so
centrality is the wrong criterion there -- the point has to be one the chain can
actually resimulate.

Appending (campaign step 5): pass --start N to extend an existing campaign.
The full stream of start+ndraw draws is regenerated from the same seed and the
first N are checked against the existing npz, so earlier draws are never
silently altered.
"""

import argparse
import os
from os.path import join, exists
import numpy as np
import torch
from omegaconf import OmegaConf

from cmass.infer.resim import load_pool, load_labels, load_test_split, \
    param_names, select_test_point, empirical_quantiles
from cmass.infer.validate import load_ensemble

WDIR = '/work/hdd/bdne/maho3/cmass-ili'
EXP_PATH = join(
    WDIR, 'abacuslike/fastpm_charm6_comphod/models/galaxy',
    'zPk0+zPk2+zPk4/kmin-0.0_kmax-0.4')

# theta layout: 5 cosmology, then HOD (alphabetical, from hodprior.csv),
# then noise_radial, noise_transverse.
N_COSMO = 5
N_NOISE = 2
COSMO_NAMES = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
NOISE_NAMES = ['noise_radial', 'noise_transverse']

# slurm_hod.sh runs with bias.hod.seed=1, so the diagnostics of each draw land
# in hod00001.h5. Keep in step with collect.py's --hod_seed.
HOD_SEED = 1

# Flags that decide how a summary vector is built. x_obs is preprocessed by the
# testing suite's own run, so these must agree or it does not mean what the
# posterior was trained to read.
PREPROC_KEYS = ('correct_shot', 'loglinear_start_idx', 'pca_features')


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp_path', default=EXP_PATH)
    p.add_argument('--wdir', default=WDIR)
    p.add_argument('--ndraw', type=int, default=10,
                   help='number of new draws to generate')
    p.add_argument('--start', type=int, default=0,
                   help='draw id to start at; earlier draws are reproduced '
                        'from the same seed and verified, not rewritten')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--nnets', type=int, default=None,
                   help='ensemble size (default: infer.Nnets from exp config)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--expect_lhid', type=int, default=1880,
                   help='abort if select_test_point disagrees; -1 to disable. '
                        'Ignored when --obs_lhid names the point outright')
    p.add_argument('--obs_lhid', type=int, default=None,
                   help='condition on this lhid instead of the most central '
                        'test point. For suites whose cosmologies are not all '
                        'LCDM (abacus), pick one the forward chain can '
                        'actually reproduce')
    p.add_argument('--tag', default=None,
                   help='names the output dir and params/ppc_<tag>_cosmo.txt '
                        '(default: obs<lhid>)')
    p.add_argument('--outroot', default=None,
                   help='default: <wdir>/ppc/<suitetag>/<exptag>/<tag>')
    p.add_argument('--testing_suite', default=None,
                   help='draw x_obs from this suite instead of the training '
                        'one (cf. infer.testing.suite)')
    p.add_argument('--testing_sim', default=None,
                   help='sim of --testing_suite (cf. infer.testing.sim)')
    return p


def default_outroot(wdir, exp_path, tag, testing=None):
    # .../<suite>/<sim>/models/<tracer>/<summaries>/<kcut>
    parts = exp_path.rstrip('/').split(os.sep)
    kcut, summ = parts[-1], parts[-2]
    sim, suite = parts[-5], parts[-6]
    root = join(wdir, 'ppc', f'{suite}_{sim}', f'{summ}_{kcut}')
    if testing is not None:
        root = join(root, 'testing', f'{testing[0]}_{testing[1]}')
    return join(root, tag)


def testing_exp_path(wdir, exp_path, suite, sim):
    """The same experiment -- tracer, summaries, k-cut -- under another suite.

    Derived from exp_path rather than rebuilt from a config, so the OOD
    experiment cannot silently resolve to a different k-cut or summary set than
    the model was trained on.
    """
    tail = exp_path.rstrip('/').split(os.sep)[-4:]  # models/tracer/summ/kcut
    return join(wdir, suite, sim, *tail)


def check_theta_layout(names):
    """Refuse experiments whose theta is not [5 cosmo][HOD...][2 noise].

    override_string slices by that layout and nothing downstream re-checks it.
    A cosmology-only experiment emits `bias.hod.theta={}`, which parse_hod
    silently ignores -- the HOD stays prior-sampled -- and collect.py only
    notices at stage D, once every draw has already been simulated.
    """
    if names[:N_COSMO] != COSMO_NAMES:
        raise SystemExit(
            f'theta must begin with {COSMO_NAMES}, got {names[:N_COSMO]}. '
            'infer.subselect_cosmo is not supported.')
    if names[-N_NOISE:] != NOISE_NAMES:
        raise SystemExit(
            f'theta must end with {NOISE_NAMES}, got {names[-N_NOISE:]}. '
            'This model was trained with infer.include_noise=False, so the '
            'last two HOD parameters would be injected as noise.')
    if len(names) <= N_COSMO + N_NOISE:
        raise SystemExit(
            'This is a cosmology-only experiment (infer.include_hod=False). '
            'The campaign injects HOD parameters per draw, so there is '
            'nothing to inject and the HOD would be drawn from its prior '
            'instead of the posterior. Use a model trained with '
            'include_hod=True.')


def check_preprocessing(cfg, test_path):
    """The testing suite must build x the way the training suite did."""
    tcfg = OmegaConf.load(join(test_path, 'config.yaml')).infer
    bad = [(k, cfg.infer.get(k), tcfg.get(k)) for k in PREPROC_KEYS
           if cfg.infer.get(k) != tcfg.get(k)]
    if bad:
        raise SystemExit(
            f'{test_path} was preprocessed differently than the training '
            'suite, so its x_obs is not what the posterior reads:\n' +
            '\n'.join(f'  infer.{k}: training={a!r}, testing={b!r}'
                      for k, a, b in bad))


def select_by_lhid(theta_src, ids_src, theta_pool, lhid, mask=None):
    """The most central row carrying this lhid.

    One lhid usually has several HOD/noise realizations. Among those, this
    keeps the same quantile-centrality criterion select_test_point uses, so
    naming an lhid narrows which rows are eligible without changing how the
    choice is made among them.
    """
    present = np.asarray(ids_src).astype(int) == lhid
    ok = present if mask is None else (present & mask)
    sel = np.flatnonzero(ok)
    if len(sel) == 0:
        if present.any():
            raise SystemExit(
                f'lhid {lhid} is in this experiment but not in its test '
                'split. A PPC must not condition on a point the posterior '
                'was trained on.')
        raise SystemExit(
            f'lhid {lhid} is not in this experiment. It holds '
            f'{len(np.unique(np.asarray(ids_src).astype(int)))} distinct '
            'lhids.')
    q = empirical_quantiles(theta_pool, theta_src[sel])
    return int(sel[np.argmin(np.linalg.norm(q - 0.5, axis=-1))])


def draw_theta(ensemble, x_obs, n, seed, device):
    """n joint draws from q(theta|x_obs), rejecting any outside prior support.

    Deterministic given seed: the accept/reject stream is fixed.
    """
    torch.manual_seed(seed)
    xt = torch.Tensor(x_obs).to(device)
    kept, n_rejected = [], 0
    with torch.no_grad():
        while sum(len(k) for k in kept) < n:
            want = n - sum(len(k) for k in kept)
            t = ensemble.sample((max(want, 64),), xt, show_progress_bars=False)
            lp = ensemble.prior.log_prob(t)
            good = torch.isfinite(lp)
            n_rejected += int((~good).sum())
            kept.append(t[good].cpu().numpy()[:want])
    return np.concatenate(kept)[:n], n_rejected


def write_cosmofile(path, cosmo, start):
    """Space-delimited, one row per draw id, matching latin_hypercube_params."""
    if start > 0:
        if not exists(path):
            raise FileNotFoundError(
                f'--start={start} but no existing cosmofile at {path}')
        old = np.loadtxt(path, ndmin=2)
        if len(old) < start:
            raise ValueError(
                f'{path} has {len(old)} rows, need at least {start}')
        if not np.allclose(old[:start], cosmo[:start], atol=1e-8):
            raise ValueError(
                f'Regenerated draws disagree with existing rows in {path}. '
                'Refusing to overwrite. Check --seed.')
    np.savetxt(path, cosmo, fmt='%.8f', delimiter=' ')


def override_string(names, theta):
    """Hydra overrides injecting one draw's HOD and noise. No spaces, so the
    job script can splice it in unquoted."""
    hod = dict(zip(names[N_COSMO:-N_NOISE], theta[N_COSMO:-N_NOISE]))
    hodstr = '{' + ','.join(f'{k}:{float(v)!r}' for k, v in hod.items()) + '}'
    return (f'bias.hod.theta={hodstr} '
            f'noise.params.radial={float(theta[-2])!r} '
            f'noise.params.transverse={float(theta[-1])!r}')


def main():
    args = build_argparser().parse_args()
    if (args.testing_suite is None) != (args.testing_sim is None):
        raise SystemExit('--testing_suite and --testing_sim go together.')
    testing = (None if args.testing_suite is None
               else (args.testing_suite, args.testing_sim))
    test_path = (None if testing is None else
                 testing_exp_path(args.wdir, args.exp_path, *testing))

    cfg = OmegaConf.load(join(args.exp_path, 'config.yaml'))
    nnets = args.nnets if args.nnets is not None else cfg.infer.Nnets
    if cfg.infer.pca_features:
        raise RuntimeError(
            'Experiment uses PCA; ppc_collect must load pca.pkl. Aborting.')

    # --- the observed point -------------------------------------------------
    x, theta, ids, tags = load_pool(args.exp_path, ('train', 'val', 'test'))
    labels, startidx = load_labels(args.exp_path)
    names = param_names(args.exp_path)
    assert len(names) == theta.shape[1], (names, theta.shape)
    check_theta_layout(names)

    if testing is None:
        if args.obs_lhid is None:
            iobs = select_test_point(theta, tags, theta)
        else:
            iobs = select_by_lhid(theta, ids, theta, args.obs_lhid,
                                  mask=(tags == 'test'))
        x_obs, theta_obs, id_obs, split_obs = (
            x[iobs], theta[iobs], ids[iobs], tags[iobs])
    else:
        check_preprocessing(cfg, test_path)
        x_t, theta_t, ids_t = load_test_split(test_path)
        if x_t.shape[1] != x.shape[1] or theta_t.shape[1] != theta.shape[1]:
            raise SystemExit(
                f'Testing suite at {test_path} has incompatible shapes '
                f'(x: {x_t.shape[1]} vs {x.shape[1]}, theta: '
                f'{theta_t.shape[1]} vs {theta.shape[1]}). It must be '
                'preprocessed with the same summaries and k-cut.')
        # Quantiles stay referenced to the training pool, as in resim.py: the
        # question is which OOD point is most central to what the model saw.
        if args.obs_lhid is None:
            iobs = select_test_point(theta_t, None, theta)
        else:
            iobs = select_by_lhid(theta_t, ids_t, theta, args.obs_lhid)
        x_obs, theta_obs, id_obs, split_obs = (
            x_t[iobs], theta_t[iobs], ids_t[iobs], 'test')

    if (args.obs_lhid is None and args.expect_lhid >= 0
            and int(id_obs) != args.expect_lhid):
        raise SystemExit(
            f'select_test_point returned lhid {id_obs}, expected '
            f'{args.expect_lhid}. Stopping (see TODO.md Phase 0.1). '
            'Pass --expect_lhid -1 to accept whichever point it picks.')
    # The tag derives from the observed lhid unless overridden, so naming a
    # different point cannot silently overwrite another campaign's cosmofile.
    tag = args.tag or f'obs{int(id_obs):05d}'
    out = args.outroot or default_outroot(
        args.wdir, args.exp_path, tag, testing)
    os.makedirs(join(out, 'overrides'), exist_ok=True)

    q_obs = empirical_quantiles(theta, theta_obs[None])[0]
    print(f'Pool: {x.shape[0]} vectors, {theta.shape[1]} params, '
          f'x is {x.shape[1]}-dim')
    if testing is not None:
        print(f'x_obs drawn out-of-distribution from {test_path}')
    print(f'x_obs: index {iobs}, lhid {id_obs}, split {split_obs}'
          + (' (requested)' if args.obs_lhid is not None else ''))
    print(f'tag:   {tag}')
    for n_, v_, q_ in zip(names, theta_obs, q_obs):
        print(f'  {n_:46s} {v_:12.6g}  (q={q_:.2f})')

    # --- draw ---------------------------------------------------------------
    ensemble = load_ensemble(args.exp_path, nnets, plot=False,
                             clean=False).to(args.device)
    npz_path = join(out, 'posterior_draws.npz')
    n_total = args.start + args.ndraw

    # Append-only: earlier draws are carried over verbatim from the npz, never
    # regenerated. New draws come from their own RNG stream (seed + start), so
    # correctness doesn't depend on torch reproducing a prefix of a longer
    # sample. They are still i.i.d. from the same q(theta|x_obs).
    if args.start > 0:
        old = np.load(npz_path, allow_pickle=True)
        if len(old['theta_draws']) < args.start:
            raise SystemExit(
                f'{npz_path} holds {len(old["theta_draws"])} draws, '
                f'need at least --start={args.start}')
        if not np.allclose(old['x_obs'], x_obs):
            raise SystemExit(
                'x_obs differs from the existing posterior_draws.npz. '
                'Refusing to append.')
        prev = old['theta_draws'][:args.start]
        prev_seeds = (old['seed_blocks'].tolist()
                      if 'seed_blocks' in old else [int(old['seed'])])
    else:
        prev = np.empty((0, theta.shape[1]))
        prev_seeds = []

    block_seed = args.seed + args.start
    new_draws, n_rejected = draw_theta(
        ensemble, x_obs, args.ndraw, block_seed, args.device)
    theta_draws = np.concatenate([prev, new_draws])
    seed_blocks = prev_seeds + [block_seed]
    print(f'\nDrew {args.ndraw} new theta ({n_rejected} rejected outside '
          f'prior support), seed={block_seed}; {len(theta_draws)} total')

    # --- cosmofile (round-trip so theta_ppc matches what the sims read) -----
    cosmofile = join('params', f'ppc_{tag}_cosmo.txt')
    write_cosmofile(cosmofile, theta_draws[:, :N_COSMO], args.start)
    theta_draws[:, :N_COSMO] = np.loadtxt(cosmofile, ndmin=2)
    print(f'Wrote {cosmofile} ({n_total} rows)')

    np.savez(
        npz_path,
        theta_draws=theta_draws, x_obs=x_obs, theta_obs=theta_obs,
        id_obs=id_obs, index_obs=iobs, split_obs=split_obs,
        seed=args.seed, seed_blocks=np.array(seed_blocks),
        n_rejected=n_rejected, exp_path=args.exp_path,
        test_path=test_path or '',
        testing_suite=args.testing_suite or '',
        testing_sim=args.testing_sim or '',
        param_names=np.array(names), labels=np.array(labels),
        startidx=np.array(startidx), nnets=nnets,
    )

    # --- per-draw overrides + manifest --------------------------------------
    for i in range(args.start, n_total):
        with open(join(out, 'overrides', f'{i}.txt'), 'w') as f:
            f.write(override_string(names, theta_draws[i]) + '\n')

    with open(join(out, 'manifest.tsv'), 'w') as f:
        f.write('\t'.join(['draw_id', 'status', 'wall_s'] + names +
                          ['sim_dir', 'diag_file']) + '\n')
        for i in range(n_total):
            simdir = join(out, 'fastpm', 'L2000-N256', str(i))
            f.write('\t'.join(
                [str(i), 'pending', ''] +
                [f'{v:.10g}' for v in theta_draws[i]] +
                [simdir, join(simdir, 'diag', 'galaxies',
                              f'hod{HOD_SEED:05d}.h5')]
            ) + '\n')

    print(f'Wrote {npz_path}, manifest.tsv, and {n_total - args.start} '
          f'override files under {out}')
    print(f'\nSuite for the job scripts:\n  '
          f'{os.path.relpath(out, args.wdir)}')


if __name__ == '__main__':
    main()
