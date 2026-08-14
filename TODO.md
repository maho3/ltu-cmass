# TODO — Posterior Predictive Check (PPC) campaign

**Audience:** a fresh Claude Code instance on Delta. Read `CLAUDE.md` first, then
this file. This is the active task spec.

**One-line goal:** take the trained posterior at a single fiducial test point,
draw parameters from it, run *new* simulations at those parameters, and check
whether the observed data vector looks like a typical draw from the resulting
posterior predictive distribution.

---

## 1. What this is, and what it is not

The posterior predictive distribution is

```
p(x_new | x_obs) = ∫ p(x | θ) q(θ | x_obs) dθ
```

Every simulated data vector must come from a θ drawn from the **joint**
posterior. Do not fix cosmology and vary only HOD — that samples a conditional
and the spread will be wrong.

**In scope**

- Draw θ from `q(θ | x_obs)` for one fiducial `x_obs`.
- Simulate each θ through the *identical* forward chain that produced the
  training data.
- Preprocess the new summaries into x-space through the *identical* path.
- Save the resulting data vectors as `.npy`.
- Plot bands (median / 68% / 95%) against `x_obs`.

**Explicitly out of scope**

- **Do not compute Mahalanobis distances or p-values.** The user will do that
  themselves from the saved `.npy`. Just make sure the arrays are saved in a
  form that makes it trivial.
- No retraining, no sequential/truncated SBI, no proposal corrections. This is
  a pure diagnostic; the posterior is taken at face value.
- No importance sampling. That was tried (`cmass/infer/resim.py`) and failed:
  ESS = 3 out of 19777 in 17-D. This campaign exists because of that failure.

---

## 2. Decisions already made — do not relitigate

| Question | Decision |
|---|---|
| Output location | New top-level `ppc/` root under `wdir` (§6) |
| Model / experiment | `zPk0+zPk2+zPk4`, `kmin-0.0_kmax-0.4` |
| `x_obs` | lhid **1880**, from the training suite's own test split |
| SLURM autonomy | Submit the 10-sim batch, then **STOP** and report |
| Sim count | 10 first, inspect, then 100 after user approval |

Source experiment directory (the "exp_path" throughout):

```
/work/hdd/bdne/maho3/cmass-ili/abacuslike/fastpm_charm6_comphod/models/galaxy/zPk0+zPk2+zPk4/kmin-0.0_kmax-0.4
```

---

## 3. Verified facts and anchors

These were checked on 2026-08-06. Re-verify anything you depend on; do not
assume the rest.

- **Pool**: 19778 data vectors, 17 parameters (5 cosmo + 10 HOD + 2 noise),
  x is 117-dim (3 × 39 bins).
- **lhid 1880 cosmology** = `[0.3119, 0.04797, 0.6129, 0.9997, 0.7557]`
  (`Ωm, Ωb, h, ns, σ8`). This matches `theta_obs[:5]` from the pool — a useful
  end-to-end cross-check that your θ bookkeeping is right.
- **Training sims were produced on Anvil**, not Delta. The per-lhid
  `config.yaml` has `meta.wdir: /anvil/scratch/x-mho1/cmass-ili`. Paths in it
  are Anvil paths; the *physics* config is what matters.
- **FastPM executable exists on Delta**: `/u/maho3/git/fastpm/src/fastpm`, and
  `cmass/conf/global.yaml` already points `meta.fastpm_exec` there.
- **The per-lhid `config.yaml` is authoritative for the forward chain**, not
  the `config.yaml` saved in the model directory (that one is the config at
  *preprocess* time and its `nbody`/`bias` blocks are stale leftovers —
  e.g. it says `lhid: 3`). Read:
  `/work/hdd/bdne/maho3/cmass-ili/abacuslike/fastpm_charm6_comphod/L2000-N256/1880/config.yaml`
- **Nbody config to replicate** (from that file): `suite: abacuslike`,
  `L: 2000`, `N: 256`, `matchIC: false`, `supersampling: 3`, `zi: 10`,
  **`zf: 0.3`**, `transfer: CAMB`, `order: 2`, `B: 2`, `N_steps: 32`,
  `COLA: true`, `mass_function: Watson_2013`, and the 10-entry `asave` list
  from 0.58622 to 0.76721. Note `zf: 0.3` with summaries taken at a ≈ 0.6667 —
  the sims run past the analysis snapshot to support lightcones. Replicate it.
- **Bias**: `bias.halo.model: CHARM`, `config_charm: config_v0.yaml`,
  `base_suite: calib_1gpch_z0.5`, `L: 1000`, `N: 128`, `vel: CIC`.
  CHARM operates at 1 Gpc/h × 128³ (7.8125 Mpc/h cells) — the same cell size as
  the 2 Gpc/h × 256³ box.
- **Snapshot group key is `0.666660`**, *not* `0.666667` as in most other
  suites. Hardcoding the usual key will raise a `KeyError`.
- **On-disk state of training sims**: only `halos.h5` and `diag/` survive —
  `nbody.h5` and `galaxies/` were deleted (`rm_galaxies=True`). You cannot
  re-derive the training x from intermediate products; you must regenerate.
- **~5 HOD realizations per lhid** (`diag/galaxies/hod00001.h5` …
  `hod00005.h5`), consistent with 19778 ≈ 5 × 4000.

---

## 4. Phase 0 — preflight (do not skip, do not run heavy jobs)

Everything here is cheap and belongs on the login node. Stop and ask the user
if any check fails.

1. **Confirm the test point.** Run `select_test_point` from
   `cmass/infer/resim.py` on the `zPk0+zPk2+zPk4/kmin-0.0_kmax-0.4` pool. It
   should return lhid **1880** (it selects on θ only, and the split is shared
   across experiments, so it should agree with the `zPk0` run). If it does not,
   **stop and report** — do not silently use a different point.
2. **Read the full `bias.hod` block** of lhid 1880's `config.yaml` and compare
   against the 10 parameter names in `hodprior.csv` at exp_path:
   `alpha, conc_gal_bias_satellites, eta_vb_centrals, eta_vb_satellites,
   logM0, logM1, logMmin, mean_occupation_centrals_assembias_param1,
   mean_occupation_satellites_assembias_param1, sigma_logM`.
   These names imply `assem_bias=True`, `vel_assem_bias=True`, `use_conc=True`.
   If the per-lhid config disagrees, resolve it before generating anything —
   the HOD model must match or the PPC tests the wrong forward model.
3. **Determine how noise is applied.** The last two θ entries are
   `noise_radial`, `noise_transverse`. Read `cmass/diagnostics/summ.py` and
   `parse_noise` ([cmass/bias/tools/hod.py:187](cmass/bias/tools/hod.py#L187))
   and establish whether the training data drew them via `diag.noise_seed` +
   `diag.noise_file` (`noise_priors/noisegrid.csv`) or via the `noise=*` config
   group. For the PPC you need to inject **exact values**, which means
   `noise=fixed noise.params.radial=<v> noise.params.transverse=<v>`.
   Verify that path produces the values you asked for.
4. **Check the CHARM model is available on Delta** (`charm` package importable,
   `config_v0.yaml` and the `calib_1gpch_z0.5` calibration present).
5. **Confirm disk headroom.** 100 × (2 Gpc/h, 256³) runs is not small. Check
   quota on `/work/hdd/bdne/`, and plan to delete `nbody.h5` and `galaxies/`
   after diagnostics (`rm_galaxies=True`) as the training runs did.

---

## 5. Parameter injection — the three mechanisms

All three were verified to exist. Each needs an assertion that it actually
took effect (§8).

**Cosmology** — `load_params(lhid, cosmofile)` reads row `lhid` of a text file
([cmass/nbody/tools.py:56](cmass/nbody/tools.py#L56)). So:

- Write `params/ppc_<tag>_cosmo.txt`, one row per draw, columns
  `Ωm Ωb h ns σ8`, **matching the exact whitespace/format of
  `params/latin_hypercube_params.txt`**.
- Run with `meta.cosmofile=./params/ppc_<tag>_cosmo.txt nbody.lhid=<draw_id>
  nbody.matchIC=0`.
- Draw index doubles as `lhid`, which also gives each draw a different IC phase
  — which is what we want (§7).

**HOD** — `cfg.bias.hod.theta.<param>` is applied last and overrides everything
([cmass/bias/tools/hod.py:161](cmass/bias/tools/hod.py#L161)). Also set
`bias.hod.from_samples=False` and `bias.hod.seed=0` so nothing resamples on top.

> ⚠️ `bias.hod.theta` defaults to `null`. A dotted override
> (`bias.hod.theta.logMmin=13.07`) may not attach to a null node — the override
> loop does `hasattr(cfg.bias.hod.theta, key)`, which is False for `None`.
> Test this on one sim and confirm the resolved `config.yaml` and the populated
> galaxy catalog reflect your values. Use `++`, a full-dict override, or a
> generated YAML if needed.

**Noise** — `noise=fixed noise.params.radial=<v> noise.params.transverse=<v>`,
pending the Phase 0 check that this is the same channel the training data used.

---

## 6. Directory structure

```
/work/hdd/bdne/maho3/cmass-ili/ppc/
  abacuslike_fastpm_charm6_comphod/
    zPk0+zPk2+zPk4_kmin-0.0_kmax-0.4/
      obs01880/
        posterior_draws.npz    # theta_draws, x_obs, theta_obs, id_obs,
                               # seed, exp_path, param_names
        manifest.tsv           # one row per draw: draw_id, status, wall time,
                               # all 17 params, output paths
        theta_ppc.npy          # (Ndraw, 17) θ actually simulated
        x_ppc.npy              # (Ndraw, 117) preprocessed, training x ordering
        plots/ppc_bands.png
        L2000-N256/
          0000/ {nbody.h5, halos.h5, galaxies/, diag/}
          0001/
          ...
```

`x_ppc.npy` and `theta_ppc.npy` are the deliverable the user computes
Mahalanobis distances from. Rows must align: `x_ppc[i]` is the data vector
simulated at `theta_ppc[i]`. Any draw that fails must be recorded in the
manifest and **dropped from both arrays together** — never padded, never
silently skipped in one but not the other.

Keep `x_obs` in `posterior_draws.npz` so the comparison is self-contained.

---

## 7. Correctness invariants — the ways this silently goes wrong

These are the whole reason to be careful. Each one produces a plausible-looking
but meaningless result.

1. **Identical forward chain.** Same suite conventions, L/N, `zf`, `asave`,
   `supersampling`, `N_steps`, CHARM version, RSD axis (z), mesh resolution,
   `correct_shot`. If the PPC sims differ in configuration, a failed check tells
   you about the config difference, not the model. This is the most likely way
   to waste the compute.
2. **Independent IC phases.** Do *not* match ICs to lhid 1880. `x_obs` is one
   realization carrying its own cosmic variance, and the posterior predictive
   must include that scatter. Distinct `lhid` per draw with `matchIC=0` gives
   this; confirm the phases actually differ.
3. **Identical preprocessing.** Reuse the functions in
   `cmass/infer/preprocess.py` (`preprocess_Pk`, and the summary concatenation
   order) rather than reimplementing. Same `kmin=0.0`, `kmax=0.4`,
   `correct_shot`, `loglinear_start_idx`. Note `zPk2`/`zPk4` are normalized by
   `zPk0` while `zPk0` is not — match `run_preprocessing` exactly. Concatenate
   in the order given by `x_startidx.txt` at exp_path (`zPk0,zPk2,zPk4`,
   starts `0,39,78,117`).
4. **If a `pca.pkl` exists at exp_path, load and apply it — never refit.**
   Refitting the scaler/PCA on the PPC ensemble is a silent, total corruption
   of the comparison. (Check: this experiment appears not to use PCA. Verify.)
5. **Joint draws only.** One θ per simulation, straight from
   `ensemble.sample()`. Do not reuse one cosmology with several HOD draws in
   this first campaign — the cost-saving variant requires drawing HOD from the
   *conditional* posterior given that cosmology, which is a different and more
   delicate construction. Keep it simple: 1 draw = 1 box.
6. **Cross-check the plumbing** on draw 0000 before scaling: the resolved
   `config.yaml` in the output dir must contain your intended cosmology, HOD
   parameters, and noise values.

---

## 8. Execution plan

**Step 1 — draw and record.** Load the ensemble
(`cmass.infer.validate.load_ensemble`, `Nnets` from config), sample
`N=10` θ from `q(θ|x_obs)` with a fixed, recorded seed. Write
`posterior_draws.npz`, `params/ppc_<tag>_cosmo.txt`, and the manifest skeleton.
Sanity-check every draw lies inside the prior support before simulating.

**Step 2 — job script.** Write `jobs/slurm_ppc.sh` following the conventions in
`jobs/slurm_infresim.sh` and the other job scripts (`bdne-delta-cpu`, `cpu`
partition, output to `/work/hdd/bdne/maho3/jobout/`). SLURM array over draws.
Per draw, the chain is:

```
cmass.nbody.fastpm  →  cmass.bias.rho_to_halo  →  cmass.bias.apply_hod  →  cmass.diagnostics.summ
```

with the cosmofile / HOD-theta / noise overrides from §5, and `rm_galaxies=True`
to keep disk under control.

**Step 3 — run 10, then STOP.** Submit the 10-draw array. Report to the user:
per-sim wall time and the extrapolated cost of 100, how many draws succeeded,
and the bands plot. **Do not submit the 100-draw run without explicit
approval** — see the standing rule in memory about requeueing.

**Step 4 — collect and plot.** Build `x_ppc.npy` / `theta_ppc.npy`. For the
plot, reuse `plot_dataspace` from `cmass/infer/resim.py` (it already does
per-summary-block panels with 68/95 bands, a residual row, and the observed
vector overplotted) — the PPC ensemble is unweighted, so pass uniform weights.
Bands only. **No p-value, no Mahalanobis.**

**Step 5 — after approval, 100 draws.** Same tag, extend the cosmofile and the
array. Do not regenerate the first 10; append.

---

## 9. Local rules

- **Login node**: light commands only (inspection, subsampled checks). Anything
  heavy goes in a SLURM job.
- **Never delete or requeue** existing data or jobs without asking first. This
  includes anything under the training suites.
- New ad-hoc scripts go in `scripts/`, plots in `figures/` (PPC plots go in the
  PPC output dir per §6), job scripts in `jobs/`.
- Report failures with their output. If a step is skipped, say so.

---

## 10. Ask the user before starting

1. **`zf: 0.3`** in the per-lhid config, while summaries are taken at
   a ≈ 0.6667 (z = 0.5). Confirm the PPC sims should run to z = 0.3 with the
   full `asave` list, rather than stopping at the analysis snapshot — this is a
   meaningful cost difference.
2. **Anvil-vs-Delta reproduction.** The training sims were generated on Anvil.
   Confirm that regenerating on Delta with the same physics config is
   acceptable, or whether a bitwise-comparable environment is required.
3. **HOD realizations per box.** Training has ~5 HOD seeds per lhid. Confirm
   1 draw = 1 box for the PPC (§7.5), or whether the user wants the multi-HOD
   structure mirrored.
4. **Wall-time budget** for a single 2 Gpc/h, 256³, 32-step FastPM run on
   Delta, if the user already knows it — otherwise measure it on draw 0000 and
   report before scaling.
