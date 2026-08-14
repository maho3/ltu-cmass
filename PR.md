# Posterior predictive checks for trained NPE posteriors

Adds an end-to-end posterior predictive check (PPC): draw parameters from
`q(theta | x_obs)` at a single test point, resimulate each draw through the
identical forward chain, and compare the resulting data vectors against the
observed one.

This replaces the importance-sampling approach in `cmass/infer/resim.py`, which
reweights existing simulations rather than generating new ones and collapses in
high dimension: on `abacuslike/fastpm_charm6_comphod` (17 parameters) it gave
ESS = 3 out of 19777. Drawing and simulating directly has no such failure mode.
`resim.py` is left in place and untouched by this PR; the PPC scripts reuse its
loaders and plotting helpers.

Validated on a 100-simulation self-consistent run. Write-up:
`ltu-gobig-notes/experiments/2026-08-13_ppc_abacuslike-fastpm_charm6_comphod/`.

## What's new

All of it lives in a new root-level `ppc/` folder, following the `power_tests/`
precedent (feature folder, unprefixed filenames, a README).

| File | |
|---|---|
| `ppc/README.md` | Stages, how to adapt to another experiment, outputs |
| `ppc/draw.py` | Draw theta from the posterior; write the cosmology file, per-draw hydra overrides, manifest, `posterior_draws.npz` |
| `ppc/slurm_nbody.sh` | Stage A: FastPM, array over draws |
| `ppc/run_charm.sh` | Stage B: CHARM. Not a SLURM script, run by hand on a GPU machine |
| `ppc/slurm_hod.sh` | Stage C: `apply_hod` + `diagnostics.summ`, array over draws |
| `ppc/slurm_collect.sh` | Stage D: collect + figures |
| `ppc/collect.py` | Build `x_ppc.npy` / `theta_ppc.npy`, verify every draw, plot bands / corner / logprob |

## One library change

`cmass/bias/rho_to_halo.py`: `apply_charm_new` takes optional `charm_yaml` and
`charm_ckpt`, wired to `bias.halo.charm_yaml` / `bias.halo.charm_ckpt`. Both
default to the module-level `CHARM_YAML` / `CHARM_CKPT` constants, so behaviour
is unchanged when the keys are absent. This exists because reproducing an older
suite requires its CHARM checkpoint, and the path was previously hardcoded: the
PPC target suite is charm6 (`charm_joint_best_val_ft15.pth`) while HEAD defaults
to charm7 (`charm_joint_v19.pth`). Running with the wrong one silently swaps the
halo bias model.

## Design notes for review

- **Parameter injection.** Cosmology goes through a generated cosmology file
  indexed by `nbody.lhid`, HOD through a full `bias.hod.theta` dict override
  (a dotted override does not attach, since `bias.hod.theta` defaults to
  `None`), noise through `noise=fixed`. `bias.hod.seed=1` is kept so
  `parse_hod` still derives `lhid*1e4+1` for galaxy placement as in training;
  the sampled parameters it produces are then fully overwritten by the theta
  dict.
- **Independent IC phases.** `matchIC=0`, so `gen_white_noise` seeds on the
  draw index. Draws are not phase-matched to the observed simulation, which
  they must not be.
- **Verification is not optional.** `ppc/collect.py` checks every draw's
  recorded cosmology, HOD and noise parameters in its diagnostics file against
  the drawn values, drops mismatches from `x_ppc` and `theta_ppc` together, and
  records the reason in `manifest.tsv`. It aborts rather than write a
  misaligned array if the block layout does not match the training data, and
  refuses to run if a `pca.pkl` is present (it must be applied, never refit).
- **Appending is append-only.** `--start N` carries the first N draws over
  verbatim from the existing npz and draws only the new ones, so extending a
  campaign cannot perturb draws that have already been simulated.
- **Held-out summaries.** The bands figure covers every available z-space
  summary, not just the ones the posterior was conditioned on; panel titles
  mark which is which. Observed vectors for held-out summaries are recomputed
  from the observed simulation's own diagnostics, located by matching recorded
  HOD and noise parameters rather than by filename.

## Known issue, not fixed here

Training summaries produced before `ba2334f` (pylians to pypower for
periodic-box P(k)) sit on a uniform dk=0.01 grid, while current runs sit on
pypower's effective-k grid. Both yield 39 bins at kmax=0.4, so shapes match and
nothing errors, but bin centres differ by up to 6.7% at the lowest k (under 2%
in P, confined to the three lowest bins; negligible above k~0.1). Bispectrum
binning is unaffected. Left as-is by decision: the affected summaries are stale
and future runs are pypower throughout. `ppc/collect.py` documents this and
plots against the PPC k values. Worth knowing before comparing any pre- and
post-`ba2334f` data vector, not only in the PPC.

## Not covered

No Mahalanobis distance or p-value is computed. `x_ppc.npy` and `theta_ppc.npy`
are row-aligned and `x_obs` is in `posterior_draws.npz` so that a distance
statistic is trivial to add downstream; the conservative band coverage and the
asymmetric predictive distributions for `zQk0` / `zQk2` argue against applying a
Gaussian statistic blindly.

## Before merging

- The whole `ppc/` folder is already staged: `draw.py`, `run_charm.sh` and
  `slurm_nbody.sh` as renames out of `scripts/` and `jobs/` (git detects them at
  92-100% similarity), the rest as new files. Nothing further to `git add`.
- The branch also carries unrelated work (charm7 integration, MTNG jobs, other
  analysis scripts, job reorganisation into `jobs/old/`). Scope the merge
  deliberately; the diff against `main` is far larger than this PR describes.
- Paths in the job scripts are Delta-specific (account `bdne-delta-cpu`, wdir
  `/work/hdd/bdne/maho3/cmass-ili`), consistent with the other job scripts in
  `jobs/`.
