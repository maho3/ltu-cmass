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
| `x_ppc_all.npz` | every plotted block, its observed vector, its k axis |
| `posterior_draws.npz` | `x_obs`, `theta_obs`, `theta_draws`, `seed_blocks`, `param_names` |
| `logq_ppc.npy` | log q(theta \| x_obs) per simulated draw |
| `manifest.tsv` | per-draw status and all parameters |
| `plots/ppc_{bands,corner,logprob}.png` | |

`collect.py` verifies every draw's recorded cosmology, HOD and noise parameters
against the drawn values, drops mismatches from `x_ppc` and `theta_ppc`
together, and records the reason in the manifest. It aborts rather than write a
misaligned array if the block layout disagrees with the training data.

Plots cover all available summaries, not just the conditioned ones. Held-out
summaries carry no weight in the inference, so disagreement there is
informative; panel titles mark which is which.

No Mahalanobis distance or p-value is computed. `x_ppc` / `theta_ppc` are
row-aligned and `x_obs` is in the npz, so a distance statistic is trivial to add.
