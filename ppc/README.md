# Posterior predictive checks

Draw parameters from `q(theta | x_obs)` at one test point, resimulate each draw
through the identical forward chain, and compare the resulting data vectors
against the observed one.

This is the replacement for importance-sampling `cmass/infer/resim.py`, which
reweights existing simulations instead of generating new ones and collapses in
high dimension (ESS = 3 of 19777 on a 17-parameter model). Nothing here modifies
`resim.py`; the scripts reuse its loaders and plotting helpers.

Validated on a 100-simulation self-consistent run against
`abacuslike/fastpm_charm6_comphod`, zPk0+zPk2+zPk4 at kmax=0.4. Write-up:
`ltu-gobig-notes/experiments/2026-08-13_ppc_abacuslike-fastpm_charm6_comphod/`.

## Stages

Run from the repo root. `PYTHONPATH=.` is needed for the two python entry
points, as in `power_tests/`; the sbatch scripts set it themselves. Stage B is
deliberately not a SLURM job: CHARM wants a GPU, which is often a different
machine from the one running FastPM.

```bash
PYTHONPATH=. python ppc/draw.py --ndraw 10  # draw theta, write cosmofile + overrides
sbatch ppc/slurm_nbody.sh                  # A: FastPM          (~14 min/draw, 128 cores)
bash   ppc/run_charm.sh 0 99               # B: CHARM           (by hand, GPU machine)
sbatch ppc/slurm_hod.sh                    # C: HOD + summaries (~3 min/draw, 16 cores)
sbatch ppc/slurm_collect.sh                # D: arrays + figures (~1 min)
```

Every stage is idempotent and resume-safe: A and C skip draws that already have
their output, B skips draws with no `nbody.h5` yet or an existing `halos.h5`, so
C and B can be re-run as the one before them drains.

To extend a campaign, `PYTHONPATH=. python ppc/draw.py --start 10 --ndraw 90`
appends. Earlier draws are carried over verbatim from the existing npz and never
regenerated, so extending cannot perturb draws already simulated.

## Adapting to a new experiment

`ppc/draw.py --exp_path` picks the trained model; the output root and the `tag`
in the job scripts derive from it. The job scripts hardcode the suite, box and
forward-chain config of the run being reproduced, which **must** match the
config that produced the training summaries, not the current defaults. Check
before reusing:

- `nbody=`, `bias=`, `multisnapshot`, `nbody.zf` in `slurm_nbody.sh` and
  `slurm_hod.sh`
- `CHARM_CKPT` in `run_charm.sh`. The checkpoint path is a `bias.halo.charm_ckpt`
  override; HEAD's default is the current CHARM, which is not necessarily the
  one that built the suite you are reproducing
- `diag.summaries` in `slurm_hod.sh`, pinned to match the training run rather
  than inheriting today's `diag/default.yaml`

A mismatch here does not error. It produces a PPC that tests a different forward
model than the posterior was trained on, and a failed check then tells you about
the config difference rather than the model.

The model must have been trained with `include_hod=True` and
`include_noise=True`, and without `subselect_cosmo`: the campaign injects theta
as `[5 cosmo][HOD][2 noise]` and slices by that layout. `draw.py` refuses
anything else up front. A cosmology-only model would otherwise emit an empty
`bias.hod.theta`, which `parse_hod` ignores — leaving the HOD sampled from its
prior rather than the posterior, and `collect.py` would only catch it at stage D
once every draw had already been simulated.

## Out-of-distribution test points

By default `x_obs` is the training suite's own most-central test point, so the
check is self-consistent: the observation comes from the same forward model
being tested. To instead condition on an observation the model never saw, pass
the testing suite the same way `infer.testing` does in `cmass/infer/validate.py`
and `cmass/infer/resim.py`:

```bash
PYTHONPATH=. python ppc/draw.py --ndraw 10 \
    --testing_suite abacus --testing_sim nbody_comp_gridnoise \
    --expect_lhid -1 --tag obs_abacus
```

Only the observation moves. The posterior, the forward chain that simulates the
draws, and the pool the centrality quantiles are measured against all stay the
training suite's, matching `resim.py`. The testing experiment is resolved by
swapping suite/sim into `--exp_path`, so its tracer, summaries and k-cut cannot
silently differ; it must already be preprocessed at that same k-cut, and `draw.py`
aborts if the x or theta widths disagree.

`x_obs` comes from the testing suite's own preprocessing run, so `draw.py` also
checks that `correct_shot`, `loglinear_start_idx` and `pca_features` match the
training experiment. A difference there would feed the posterior a vector built
to a different recipe than it was trained to read, and nothing downstream would
notice.

This asks a different question than the self-consistent run: whether the forward
model, driven by the parameters the posterior believes explain an observation
from elsewhere, reproduces that observation. Disagreement is a real finding, not
a bug — which is why it is worth running the self-consistent check first, so a
failure here is attributable to the model rather than to the plumbing.

Two practical notes:

- `--expect_lhid` defaults to the self-consistent campaign's point and will
  abort. Run once with `-1`, then pin the lhid it reports — or name the point
  with `--obs_lhid`, which makes the guard moot.
- Outputs gain a `testing/<suite>_<sim>/` segment (below), so `ppcdir` in the
  three job scripts and `--ppc_dir` for `collect.py` both change. `draw.py`
  prints the exact path to paste in; `slurm_collect.sh` relies on `collect.py`'s
  built-in default and needs `--ppc_dir` added for an OOD campaign.

### Choosing the observed point

By default the observed point is whichever test-split row sits closest to the
median of the training pool, in per-parameter quantile distance. That is the
right question only when every candidate is a cosmology the forward chain can
reproduce.

It is the wrong question on suites whose test set mixes cosmology types. Abacus
holds LCDM with `Mnu = 0`, LCDM with `Mnu > 0`, and non-LCDM models in one set,
while the forward chain here carries only the five LCDM parameters — so
resimulating a massive-neutrino or non-LCDM point silently drops what made it
that point, and the check then measures the missing physics rather than the
model. `--obs_lhid` names the point instead:

```bash
PYTHONPATH=. python ppc/draw.py --ndraw 10 \
    --testing_suite abacus1gpch --testing_sim custom_hodz_gridnoise \
    --obs_lhid 7
```

An lhid usually has several HOD and noise realizations. Among the rows carrying
the one named, the same centrality criterion picks which to use, so `--obs_lhid`
narrows the candidates without changing how the choice is made among them. In
the self-consistent case the point must still come from the test split; naming
one the posterior trained on is refused.

`--tag` defaults to `obs<lhid>`, so it tracks the point automatically and a
second campaign cannot overwrite the first's `params/ppc_<tag>_cosmo.txt`.

Which Abacus lhids are LCDM with `Mnu = 0` is recorded in
`<wdir>/scratch/abacus_custom_table.csv`, in its `LCDM` and `Massive Neutrinos`
columns; `ltu-gobig-notes/scripts/ood_abacus_inference.py` reads the same table
to classify test points.

## Outputs

Under `<wdir>/ppc/<suite>_<sim>/<summaries>_<kcut>/<tag>/`, or
`<wdir>/ppc/<suite>_<sim>/<summaries>_<kcut>/testing/<tsuite>_<tsim>/<tag>/` for
an out-of-distribution run:

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

No Mahalanobis distance or p-value is computed. `x_ppc` / `theta_ppc` are
row-aligned and `x_obs` is in the npz so a distance statistic is trivial to add.
