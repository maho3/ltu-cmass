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

from cmass.infer.resim import load_pool, load_labels, param_names, \
    select_test_point, empirical_quantiles
from cmass.infer.validate import load_ensemble

WDIR = '/work/hdd/bdne/maho3/cmass-ili'
EXP_PATH = join(
    WDIR, 'abacuslike/fastpm_charm6_comphod/models/galaxy',
    'zPk0+zPk2+zPk4/kmin-0.0_kmax-0.4')

# theta layout: 5 cosmology, then HOD (alphabetical, from hodprior.csv),
# then noise_radial, noise_transverse.
N_COSMO = 5
N_NOISE = 2


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
                   help='abort if select_test_point disagrees; -1 to disable')
    p.add_argument('--tag', default='obs01880')
    p.add_argument('--outroot', default=None,
                   help='default: <wdir>/ppc/<suitetag>/<exptag>/<tag>')
    return p


def default_outroot(wdir, exp_path, tag):
    # .../<suite>/<sim>/models/<tracer>/<summaries>/<kcut>
    parts = exp_path.rstrip('/').split(os.sep)
    kcut, summ = parts[-1], parts[-2]
    sim, suite = parts[-5], parts[-6]
    return join(wdir, 'ppc', f'{suite}_{sim}', f'{summ}_{kcut}', tag)


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
    out = args.outroot or default_outroot(args.wdir, args.exp_path, args.tag)
    os.makedirs(join(out, 'overrides'), exist_ok=True)

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

    iobs = select_test_point(theta, tags, theta)
    x_obs, theta_obs, id_obs = x[iobs], theta[iobs], ids[iobs]
    if args.expect_lhid >= 0 and int(id_obs) != args.expect_lhid:
        raise SystemExit(
            f'select_test_point returned lhid {id_obs}, expected '
            f'{args.expect_lhid}. Stopping (see TODO.md Phase 0.1).')
    q_obs = empirical_quantiles(theta, theta_obs[None])[0]
    print(f'Pool: {x.shape[0]} vectors, {theta.shape[1]} params, '
          f'x is {x.shape[1]}-dim')
    print(f'x_obs: index {iobs}, lhid {id_obs}, split {tags[iobs]}')
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
    cosmofile = join('params', f'ppc_{args.tag}_cosmo.txt')
    write_cosmofile(cosmofile, theta_draws[:, :N_COSMO], args.start)
    theta_draws[:, :N_COSMO] = np.loadtxt(cosmofile, ndmin=2)
    print(f'Wrote {cosmofile} ({n_total} rows)')

    np.savez(
        npz_path,
        theta_draws=theta_draws, x_obs=x_obs, theta_obs=theta_obs,
        id_obs=id_obs, index_obs=iobs, split_obs=tags[iobs],
        seed=args.seed, seed_blocks=np.array(seed_blocks),
        n_rejected=n_rejected, exp_path=args.exp_path,
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
                [simdir, join(simdir, 'diag', 'galaxies', 'hod00000.h5')]
            ) + '\n')

    print(f'Wrote {npz_path}, manifest.tsv, and {n_total - args.start} '
          f'override files under {out}')
    print(f'\nSuite for the job scripts:\n  '
          f'{os.path.relpath(out, args.wdir)}')


if __name__ == '__main__':
    main()
