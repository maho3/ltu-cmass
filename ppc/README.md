# Posterior predictive checks

Draw parameters from `q(theta | x_obs)` at one test point, resimulate each draw
through the identical forward chain, and compare the resulting data vectors
against the observed one.

This replaces importance-sampling `cmass/infer/resim.py`, which reweights
existing simulations instead of generating new ones and collapses in high
dimension (ESS = 3 of 19777 on a 17-parameter model). Nothing here modifies
`resim.py`; the scripts reuse its loaders and plotting helpers.

Validated on a 100-simulation self-consistent run against
`abacuslike/fastpm_charm6_comphod`, zPk0+zPk2+zPk4 at kmax=0.4. Write-up:
`ltu-gobig-notes/experiments/2026-08-13_ppc_abacuslike-fastpm_charm6_comphod/`.

## Stages

Run from the repo root; `PYTHONPATH=.` is needed for the python entry points,
as in `power_tests/` (the sbatch scripts set it themselves). Stage B is
deliberately not a SLURM job: CHARM wants a GPU, often a different machine from
the one running FastPM.

| | snapshot tracer (`galaxy`) | lightcone tracer |
|---|---|---|
| | `PYTHONPATH=. python ppc/draw.py --ndraw 10` | same |
| A | `sbatch ppc/slurm_nbody.sh` | `sbatch ppc/slurm_nbody_lc.sh` |
| B | `bash ppc/run_charm.sh 0 99` | `bash ppc/run_charm_lc.sh 0 99` |
| C | `sbatch ppc/slurm_hod.sh` | `sbatch ppc/slurm_lightcone.sh` |
| D | `sbatch ppc/slurm_collect.sh` | same, plus `--ppc_dir` |

Every stage is idempotent and resume-safe: A and C skip draws that already have
their output, B skips draws with no `nbody.h5` yet or an existing `halos.h5`, so
B and C can be re-run as the stage before them drains.

`draw.py` prints the campaign path for the `ppcdir` variable in the job scripts,
and `--start N` extends a campaign (earlier draws are carried over verbatim from
the npz, never regenerated).

`ppc/layout.py` owns both path conventions — the trained experiment's
`<suite>/<sim>/models/<tracer>/<summaries>/<kcut>` and the campaign output tree.
Both entry points read the tracer out of `--exp_path` and switch on it, so
lightcone campaigns need no extra flags.

## Choosing the observed point

| flag | |
|---|---|
| *(default)* | the test-split row closest to the training pool's median, in per-parameter quantile distance |
| `--obs_lhid N` | condition on lhid `N` instead |
| `--testing_suite S --testing_sim M` | take `x_obs` from another suite's test split (as `infer.testing` does in `validate.py` / `resim.py`) |
| `--tag` | names the output dir and `params/ppc_<tag>_cosmo.txt`; defaults to `obs<lhid>` |

**`--obs_lhid` is required on suites whose test set mixes cosmology types.**
Abacus holds LCDM `Mnu=0`, LCDM `Mnu>0` and non-LCDM models together, while the
forward chain carries only the five LCDM parameters — resimulating a
massive-neutrino point silently drops what made it that point, and the check
then measures the missing physics rather than the model. Which lhids are LCDM
`Mnu=0` is recorded in `<wdir>/scratch/abacus_custom_table.csv` (`LCDM` and
`Massive Neutrinos` columns); `ltu-gobig-notes/scripts/ood_abacus_inference.py`
reads the same table. An lhid usually has several HOD and noise realizations;
among those, the default centrality criterion still picks which to use.

**Out-of-distribution** runs move only the observation — posterior, forward
chain and quantile reference pool all stay the training suite's. This asks
whether the forward model, driven by the parameters the posterior believes
explain an observation from elsewhere, reproduces that observation;
disagreement is a finding, not a bug, so run the self-consistent check first.
Outputs gain a `testing/<suite>_<sim>/` segment, so `ppcdir` and `--ppc_dir`
change. `--expect_lhid` defaults to the self-consistent campaign's point and
will abort — pass `-1` or name the point with `--obs_lhid`.

## Before reusing these scripts

The job scripts hardcode the suite, box and forward-chain config of the run
being reproduced, which **must** match what produced the training summaries,
not today's defaults. A mismatch does not error — it silently tests a different
forward model, and a failed check then tells you about the config difference
rather than about the model. Check:

- `nbody=`, `bias=`, `multisnapshot`, `nbody.zf` in the stage A and C scripts
- `CHARM_CKPT` in `run_charm.sh` (a `bias.halo.charm_ckpt` override). HEAD's
  default is the current CHARM, not necessarily the one that built your suite
- `diag.summaries`, pinned rather than inheriting today's `diag/default.yaml`
- the HOD model. The mtnglike suite moved from `zheng07` to `zheng07zinterp`,
  changing the HOD parameter set from 10 to 16; models trained before the switch
  cannot be checked against simulations run after it

`draw.py` refuses up front anything whose theta is not `[5 cosmo][HOD][2 noise]`
— cosmology-only models (`include_hod=False`), `include_noise=False`, and
`subselect_cosmo`. For an out-of-distribution run it also requires the testing
suite's `correct_shot`, `loglinear_start_idx` and `pca_features` to match, since
`x_obs` comes from that suite's own preprocessing run.

## Lightcone campaigns

Three things differ from the snapshot chain:

- **`hodlightcone` replaces `apply_hod`.** It reads `halos.h5` and applies the
  HOD while stitching the lightcone, so there is no separate `galaxies/` stage.
  It still routes through `parse_hod`/`parse_noise`, so the per-draw overrides
  inject exactly as in the snapshot campaign.
- **`multisnapshot=True` is required** — the lightcone is stitched out of
  `nbody.asave`. Stage A costs proportionally more wall time and scratch, and
  stage B runs CHARM once per snapshot.
- **`aug_seed`** does not exist in the snapshot chain. It is pinned to 1
  alongside `bias.hod.seed=1`, matching the training suite's `aug_seed ==
  hod_seed` pairing, so diagnostics land in `hod00001_aug00001.h5`.

Change `cap` at the top of `slurm_lightcone.sh` to run `ngc`, `sgc` or `simbig`
instead of `mtng` — the diag flag, `survey.geometry` and `bias.hod.custom_prior`
all follow it, and the experiment's tracer must match (`<cap>_lightcone`). The
three scripts reproduce `jobs/slurm_fastpm_3gpch.sh`, `jobs/slurm_charm.sh` and
the mtng block of `jobs/slurm_mtng_bias.sh` on the `rundelta` branch.

## Outputs

Under `<wdir>/ppc/<suite>_<sim>/<summaries>_<kcut>/[testing/<tsuite>_<tsim>/]<tag>/`:

| | |
|---|---|
| `x_ppc.npy` | (Ndraw, Nfeat) inference blocks only, training x ordering |
| `theta_ppc.npy` | (Ndraw, Nparam), row-aligned with `x_ppc` |
| `x_ppc_all.npz` | every plotted block, its observed vector, its k axis, and `<block>_k123` triangle sides for bispectra |
| `posterior_draws.npz` | `x_obs`, `theta_obs`, `theta_draws`, `seed_blocks`, `param_names` |
| `logq_ppc.npy` | log q(theta \| x_obs) per simulated draw |
| `manifest.tsv` | per-draw status and all parameters |
| `plots/ppc_{bands,corner,logprob}.png` | |
| `plots/ppc_pcapvalue.png`, `ppc_pcapvalues.tsv` | OOD p-values in PCA space, see below |
| `plots/ppc_kbinpvalue.png`, `ppc_kbinpvalues.tsv` | OOD p-values on k-bin subsets, see below |

`collect.py` verifies every draw's recorded cosmology, HOD and noise parameters
against the drawn values, drops mismatches from `x_ppc` and `theta_ppc`
together, and records the reason in the manifest. It aborts rather than write a
misaligned array if the block layout disagrees with the training data.

Plots cover all available summaries, not just the conditioned ones. Held-out
summaries carry no weight in the inference, so disagreement there is
informative; panel titles mark which is which.

## Interpreting the p-values

Both figures test one null hypothesis: **`x_obs` is a draw from the posterior
predictive** that the resimulations sample. Small p means `x_obs` is out of
distribution (OOD): no set of parameters the posterior believes can reproduce
it. Every value is `-log10 p` on the plots, so up/right means more OOD.

**How p is computed.** This is the matched leave-one-out test of
`predictive_checks.ppc_mahalanobis` (it reproduces that package's `pca` mode to
1e-8). In fold `j`, fit a mean and covariance to the other N−1 draws, then
score both the held-out draw `j` and `x_obs` against that same fit with a
Mahalanobis distance `T`. Then

    p = (1 + #{j : T_j >= T_obs,j}) / (N + 1)

Under H0, `x_obs` and draw `j` are interchangeable in every fold, so p is
calibrated with no Gaussian assumption. Three consequences:

- **p cannot go below 1/(N+1)** (≈ 0.01 for 100 draws). A test at the floor
  says "more extreme than every draw", not how much more.
- **`p_F`** (Hotelling F) extrapolates below the floor by assuming the draws
  are Gaussian. Use it to rank how bad a failure is, never as the headline.
  The draws are often heavier-tailed than Gaussian, so it can overstate
  significance.
- **`p_loo_std`** is the Monte Carlo noise from having only N draws. It is
  ≈ 0.05 near p = 0.5, so differences smaller than that mean nothing.

**Inference vs held-out blocks.** Blocks the posterior was conditioned on
(`[inf]`) reuse `x_obs` for both fitting and testing, so their p is
conservative (biased high) and a pass is weak evidence. Held-out blocks
(`[held]`, e.g. the bispectrum when training on P(k)) are a clean test. A model
that passes on its inference blocks but fails on held-out ones reproduces the
statistics it was fitted to without capturing the physics behind them.

### `ppc_pcapvalue.png`: PCA space

With about 100 draws you cannot estimate a covariance over 100–500 features,
so `T` is computed in the top `k` principal components of the draws
(`--n_pca`, default 10).

| panel | shows | read it as |
|---|---|---|
| (a) | draws and `x_obs` in the first two PCs of the inference vector | the star inside the cloud: typical along the directions the draws vary most |
| (b), (c) | one point per fold, `T_j` against `T_obs,j`, for the inference and held-out vectors | p is the fraction of points on or above the diagonal. All points below it means OOD |
| (d) | p per block: top-k PCA (circle), Hotelling (square), all-feature Ledoit–Wolf (triangle) | past the dashed line fails at 0.05; the dotted line is the floor |
| (e) | p against k for every block | a row that flips colour with k is a fragile verdict. Report the k it holds at |
| (f) | p for the deviation outside the top-k PCs | catches structure the draws never produce. It uses an unweighted norm of raw features, so high-variance features dominate |

**The PCA and Ledoit–Wolf tests answer different questions.** The top PCs are
mostly directions the parameters can move, so the PCA test asks whether any
posterior draw lands near `x_obs`. Ledoit–Wolf keeps every direction,
including low-variance combinations that no parameter setting produces. A block
that passes PCA but fails Ledoit–Wolf is off in a direction the model cannot
reach. The abacus OOD check's combined inference vector is the example:
p = 0.67 with PCA, 0.0099 with Ledoit–Wolf.

### `ppc_kbinpvalue.png`: k-bin subsets

The same matched test with no PCA: the full Mahalanobis distance on the raw
features inside a k-range. A subset is only tested if it has d ≤ N/2 features,
so each fold's covariance stays well estimated. Subsets over the limit, or with
no bins, are hatched and not tested. Heatmap cells print their d.

| panel | shows | read it as |
|---|---|---|
| (a) | p in sliding k-windows (`--win_width` 0.06, `--win_step` 0.02) for each P(k) multipole and their combination, BAO range shaded | *which scales* are OOD |
| (b) | p for all k ≤ kmax, P(k) averaged into `--kcoarse` 0.04 bins | *from which kmax* on the model stops fitting. Compare to the training kmax line |
| (c) | per-bin `(x_obs − mean)/σ` of the draws | what drives (a) and (b), and the sign of the offset. Marginal only: a smooth 1σ offset across many bins can still fail the full test |
| (d), (e) | (a) and (b) for every block, bispectra included. Triangles are placed by their largest side and are not averaged, so the cumulative scan stops once it passes N/2 triangles | where the held-out statistics break |
| (f) | the matched folds behind the most OOD P(k) window | how far out that window is |

**Do not over-read a single window.** Panel (a) makes about 17 overlapping
tests per curve, so an isolated dip to p ≈ 0.04 is expected by chance even for
an in-distribution point. Trust a run of adjacent low windows, or the
cumulative curve (b).

Bispectrum rows need `<block>_k123` in `x_ppc_all.npz`. Campaigns collected
before it was stored show only P(k) and `zEqBk0`; re-run `collect.py
--no_theta_plots` to add it (it rewrites the deliverables unchanged).

### Rerunning

`--pvalue_only` rebuilds both figures and tables from `x_ppc_all.npz` in under
a minute, without the per-draw sims or the ensemble. `--no_pvalue` skips them
in a full collect.

## Reparameterized models (`*_reparam`)

Experiments trained with `infer.reparam_degeneracy=True` sample
`(degen_r, degen_phi)` in place of `eta_vb_centrals` / `noise_radial`. `draw.py`
inverts them to the physical values for the sims (bounds from the non-reparam
sibling suite's `hodprior.csv` and the experiment's `noiseprior.yaml`), rejects
draws that map outside the physical prior box, and stores both
`theta_draws` (reparam) and `theta_phys` in the npz. `collect.py` verifies sims
against `theta_phys`. The stage scripts take `TAG` from the environment
(`TAG=obs01880 sbatch ppc/slurm_nbody.sh`).
