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

## Outputs

Under `<wdir>/ppc/<suite>_<sim>/<summaries>_<kcut>/<tag>/`:

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
