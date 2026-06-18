# ltu-cmass — Context for Claude instances

This file teaches a fresh Claude how to navigate the `ltu-cmass` repo and its
associated dataset on `maho3`'s machine. Read once at the start of a session.

## Who / what

- **Codebase**: `ltu-cmass`, a simulation-based-inference (SBI) pipeline that
  turns N-body simulations into mock galaxy catalogs and summary statistics,
  then trains neural posterior estimators (NPE/NLE) to infer cosmology + HOD
  parameters from galaxy clustering.
- **User**: Matthew Ho. Runs expensive steps on SLURM compute nodes; writes
  Python interactively. Expects terse, physics-literate responses.

## Compute clusters

Three environments are in use. Paths, accounts, and conda envs differ — check
the job script header before adapting one for a different cluster.

| Cluster | Login / git path | conda env | SLURM account | Partitions | wdir |
|---------|-----------------|-----------|---------------|------------|------|
| **Delta** (local) | `/u/maho3/git/ltu-cmass` | `cmass` (`source ~/.bashrc`) | `bdne-delta-cpu` | `cpu`, `ghx4` (GPU) | `/work/hdd/bdne/maho3/cmass-ili/` |
| **Anvil** (Purdue) | `/home/x-mho1/git/ltu-cmass-run` | `cmassrun` (`module restore cmass`) | `phy240043` | `shared`, `wholenode` | `/anvil/scratch/x-mho1/cmass-ili/` |
| **Bridges2** (PSC) | — | — | `phy240015p` | — | `/ocean/projects/phy240015p/mho1/cmass-ili/` |

Job output lands in `$wdir/../jobout/` (e.g. `/work/hdd/bdne/maho3/jobout/`).

## Convention summary (one-line rules)

- **Default box**: `L = 1000 Mpc/h`, `N = 128` → voxel `Δ ≈ 7.8125 Mpc/h`.
- **Default snapshot**: `a = 0.666667`, i.e. `z = 0.5`.
- **RSD axis**: `z` (axis=2), everywhere.
- **Observational noise**: "radial" = LOS (z), "transverse" = in-plane. Applied
  via `cmass.diagnostics.tools.noise_positions(ra=0, dec=90, ...)`.
- **Mass convention**: `halos.h5/mass` is stored as `log10(M_200c)` in Msun/h.
- **Positions**: Mpc/h, in `[0, L)` (periodic). `Pk_library.MA` requires
  `float32` positions — cast before calling or it will raise cryptically.
- **Pk columns**: `calcPk` returns `(k, Pk)` with `Pk.shape = (Nk, 3)` for
  `[P_0, P_2, P_4]`.
- **lhid=0 is NOT fiducial**. Fiducial sits at the middle of the latin
  hypercube, not the first row.

## Repo layout

```
ltu-cmass/
  cmass/               # the package
    nbody/             # FastPM / BORG / Pinocchio / PMWD drivers
    bias/              # halo-bias models (LIMD, CHARM, ZhengBiased HOD)
      apply_hod.py     # populate_hod() — halotools wrapper; call after parse_hod()
      rho_to_halo.py   # apply_charm(), apply_limd()  — rho → halos
      tools/hod.py     # parse_hod(), parse_noise(), build_HOD_model()
      tools/halo_sampling.py  # sample_3d() — grid-index → Mpc/h convention
    diagnostics/
      summ.py          # main summary driver (halo, galaxy, lightcone tracers)
      tools.py         # noise_positions(), get_mesh_resolution()
      calculations.py  # MA, MAz, calcPk, calcBk_*
      pypower.py       # MPI P(k) for lightcones (canonical survey backend)
    survey/            # lightcone stitching + geometries (ngc, sgc, mtng, simbig)
    infer/             # SBI: preprocess → train_nle/train → validate_nle
    conf/              # hydra configs (see "Hydra" below)
    utils/utils.py     # get_source_path, load_params, cosmo_to_astropy, ...
  scripts/             # ad-hoc analysis scripts (new work goes here)
  figures/             # output drop for plots
  jobs/                # SLURM submission scripts
  params/latin_hypercube_params.txt  # 4001 lines × [Ωm, Ωb, h, ns, σ8]
  notebooks/           # exploratory
  matts_tests/         # Matt's sandbox — do NOT edit
```

## Data root

The on-disk wdir is `/work/hdd/bdne/maho3/cmass-ili/` **on Delta** (configured in
`cmass/conf/global.yaml` under `meta.wdir` — change per cluster). Key knobs in
`global.yaml`:
- `meta.summ_dir` — load summaries from a *different* path than `wdir` (useful
  when you have read-only access to another cluster's summaries).
- `meta.cosmofile` — must match the suite (see "Cosmofiles" below).
- `meta.fastpm_exec` / `meta.pinocchio_exec` — paths to compiled executables
  (currently pointing to Anvil paths; update for Delta).

Directory tree:

```
/work/hdd/bdne/maho3/cmass-ili/
  quijote/                   # Quijote N-body halos (reference truth)
    nbody/                   #   canonical
    nbody_1pnoise/           #   variants with extra noise injections
    nbody_4noise/ ...
    meshed/ meshed_hodz/ ... #   halos snapped to grid (CHARM-like)
  quijotelike/               # FastPM ICs matched to Quijote lhid-by-lhid
    fastpm/                  #   standard
    fastpm_4k/ fastpm_CAMELS/ ...
  abacus/custom_4noise/      # Abacus variants (real)
  abacuslike/fastpm_recnoise*/  # Abacus-like mocks
  hod_priors/                # {ngc,sgc,mtng,simbig}.npy  — HOD param samples
  noise_priors/              # noise1p.csv, noisegrid.csv — (σ_r, σ_t) rows
  logs/                      # hydra run logs
  scratch/                   # misc CSVs (abacus cosmologies, etc.)
```

Only some suites are on the local disk at any time — others are moved in via
**Globus** as needed (storage-limited). `data_accounting/manifest.tsv` tracks
which suites exist on DELTA, BRIDGES, and ANVIL. Before assuming a path exists,
check; if missing, ask the user rather than inventing a different source.

### Box sizes

| Box | L (Mpc/h) | N | Suites |
|-----|-----------|---|--------|
| 1 Gpc/h (default) | 1000 | 128 | quijote, quijotelike, quijotelike-fid, quijotelike_nophase, fastpm_CAMELS, fastpm_4k |
| 2 Gpc/h | 2000 | 256 | abacus, abacuslike |
| 3 Gpc/h | 3000 | 384 | mtng, mtnglike |

Notable 1 Gpc/h suite variants:
- `fastpm_CAMELS` — ICs at CAMELS Big Sobol cosmologies (`big_sobol_params.txt`)
- `fastpm_4k` / `fastpm_4k_hodz` — ~4000-sim stitched dataset (`stupid_fastpm_4k_params.txt`)
- `quijotelike-fid/fastpm` — fixed fiducial cosmology, 2000 lhids
- `quijotelike_nophase/fastpm` — fixed ICs (vary cosmo only)
- `fastpm_charm2/3/4/5` — successive CHARM model versions on `quijotelike` density

### Cosmofiles

Set `meta.cosmofile` to match the suite:

| File | Suite |
|------|-------|
| `params/latin_hypercube_params.txt` | quijote / quijotelike (default, 4001 rows) |
| `params/big_sobol_params.txt` | fastpm_CAMELS |
| `params/stupid_fastpm_4k_params.txt` | fastpm_4k / fastpm_4k_hodz |
| `params/abacus_cosmologies.txt` | abacus / abacuslike |
| `params/abacus_custom_cosmologies.txt` | newer abacus custom runs |
| `params/mtng_cosmologies.txt` | mtng / mtnglike (single cosmology) |

### Per-lhid layout (`<suite>/<sim>/L{L}-N{N}/{lhid}/`)

```
config.yaml              # full resolved hydra config from that run
halos.h5                 # group '0.666667' -> {pos, vel, mass}; CHARM halos same schema
nbody.h5                 # group '0.666667' -> {rho (N³), fvel (N³×3)}     (only some suites)
galaxies/hod{SEED:05d}.h5         # populated galaxies, if run
diag/galaxies/hod{SEED:05d}.h5    # pre-computed summaries (see below)
```

`halos.h5` and `nbody.h5` share matched ICs by lhid — the Quijote halos and the
FastPM density field for the same lhid live in the same "universe". CHARM is
typically run on the `quijotelike/fastpm` density to produce a
`quijotelike/fastpm/.../halos.h5`.

### `diag/galaxies/hodNNNNN.h5` schema

Single group `0.666667` with datasets:

- `Pk_k3D (Nk,)`, `Pk (Nk, 3)` — real-space multipoles `[P0, P2, P4]`
- `zPk_k3D`, `zPk (Nk, 3)` — redshift-space multipoles
- `Bk_k123 (3, Nt)`, `Bk (2, Nt)`, `Qk (2, Nt)`, `bPk_k3D`, `bPk` — bispectrum
  triangle configs and their P(k)s; `zBk`, `zQk`, `zbPk` for z-space
- attrs: `Ngalaxies`, `boxsize`, `nbar`, `log10nbar`, `high_res`, `timestamp`
- group-level attrs: `HOD_model`, `HOD_names`, `HOD_params`, `HOD_seed`,
  `cosmo_names`, `cosmo_params`, `noise_dist`, `noise_radial`,
  `noise_transverse`, and `config` (the full hydra yaml as a string)

If these already exist for the lhid you need, **load them** instead of
recomputing — recomputation is expensive.

## Hydra configs

Entry-points like `cmass.bias.apply_hod`, `cmass.diagnostics.summ`,
`cmass.survey.lightcone` are `@hydra.main` with `config_path="../conf"` and
`config_name="config"`. Override from CLI, e.g.:

```bash
python -m cmass.bias.apply_hod sim=nbody nbody=quijote bias=zheng_biased \
    nbody.lhid=663 bias.hod.seed=1
```

Top-level defaults (`cmass/conf/config.yaml`):

```
global, nbody:1gpch, fit:quijote_HR, bias:cic_hod, survey:default,
diag:default, noise:fixed, infer:default, net:hyperprior
sim: pmwd        # overridable
```

Useful sub-configs:

- `nbody/quijote.yaml` — suite=quijote, L=1000, N=128, zf=0.5
- `bias/zheng_biased.yaml` — ZhengBiased HOD with `assem_bias=True`,
  `vel_assem_bias=True`, `default_params=reid2014_cmass`, `from_samples=true`,
  `use_conc=true`, `noise_uniform=True`. `hod.seed`: `0` = defaults, `>0` =
  sample; internally parse_hod maps it to `seed*1e4 + lhid`.
- `bias/zhenginterp_biased.yaml` — **more common in recent jobs**. Model
  `zheng07zinterp` interpolates HOD params across redshift using `zpivot:
  [0.4, 0.5, 0.7]`. Typically paired with `bias.hod.custom_prior=ngc`.
- `noise/reciprocal.yaml` — `dist=Reciprocal, a=0.1, b=4.5105` (= Δ/√3)
- `diag/default.yaml` — `survey_backend: pypower`, `bispectrum_backend: polybin`
- `infer/default.yaml` — 10 experiments over `{Pk0, Pk2, Pk4, Bk0, Qk0}` and
  their `z*` variants, `kmin ∈ {0, 0.02}`, `kmax ∈ {0.1..0.5}`

## Running the standard chain

Hydra-driven, executed on compute nodes:

1. `cmass.nbody.{fastpm|borglpt|pmwd|pinocchio}` → writes `nbody.h5`.
   - `multisnapshot=True` (default) uses all snapshots for lightcone
     extrapolation; `False` uses final snapshot only.
   - `nbody.matchIC=0` disables matching Quijote ICs (needed for non-Quijote
     cosmologies like Abacus or CAMELS).
2. `cmass.bias.rho_to_halo` (CHARM or LIMD) → writes `halos.h5`.
3. `cmass.bias.apply_hod` → writes `galaxies/hod{SEED:05d}.h5`.
4. `cmass.survey.{lightcone|hodlightcone|simbig_selection}` → lightcone catalog
   (optional). `survey.aug_seed` controls augmentation; output filename:
   `hod{SEED:05d}_aug{AUG:05d}.h5`.
5. `cmass.diagnostics.summ` → writes `diag/.../hod{SEED:05d}.h5`.
   - `diag.from_scratch=True` — recompute even if file exists.
   - `diag.noise_seed=N` — apply noise realization N at the diagnostics stage
     (used for noise-grid experiments like `nbody_hodz_gridnoise`); distinct
     from `bias.hod.seed`.
6. `cmass.infer.preprocess` → aggregates summaries, splits by lhid.
7. `cmass.infer.optuna` → Optuna hyperparameter search (run before training).
8. `cmass.infer.train[_nle]` with `infer.retrain=True` → trains using best
   Optuna params. `validate[_nle]` evaluates.

**Disk management in jobs**: set `rm_galaxies=True` to delete `galaxies/` and
lightcone dirs after diagnostics are written (saves ~GB per lhid).

**Key inference flags** (used in nearly all current jobs):
- `infer.include_noise=True` — jointly infer noise nuisance params with cosmo.
- `infer.include_hod=False` — do not jointly infer HOD params.
- `net=niall2 infer.embedding_net=fun` — non-default network; standard for
  recent runs.

**Canonical P(k) backend for surveys: `pypower` (MPI).** Box P(k) still uses
Pk_library via `calcPk`. Bispectrum default: `polybin`.

### SLURM array pattern

Jobs cover large lhid ranges via a two-level loop: SLURM array task ID sets a
base offset, then an inner `for offset in $(seq 0 STEP MAX)` iterates further
lhids per task. Example: `--array=0-99`, inner step 100 over 0–4799 → 100 tasks
× 48 lhids each = 4800 total. When adapting a job, check both the array range
and the inner loop range.

## Helpers you'll reuse often

```python
from cmass.utils.utils import get_source_path, load_params, cosmo_to_astropy
from cmass.bias.apply_hod import populate_hod
from cmass.bias.tools.hod import parse_hod, parse_noise
from cmass.diagnostics.calculations import MA, MAz, calcPk
from cmass.diagnostics.tools import noise_positions
```

- `get_source_path(wdir, suite, sim, L, N, lhid)` → the canonical per-lhid dir.
- `load_params(lhid, 'params/latin_hypercube_params.txt')` → `[Ωm, Ωb, h, ns, σ8]`;
  pass `"fid"` for the Quijote fiducial list.
- `cosmo_to_astropy([Ωm, Ωb, h, ...])` → `FlatLambdaCDM`.
- `parse_hod(cfg)` — resolves `theta` from defaults/file/overrides; call before
  `populate_hod`. Needs a minimal cfg with `meta`, `sim`, `nbody`, `bias`.
- `populate_hod(hpos, hvel, hmass, cosmo, L, z, model, theta, seed=..., mdef='200c', ...)`
  → halotools `galaxy_table` with columns `x, y, z, vx, vy, vz, gal_type,
  halo_id`. `gal_type` ∈ {`centrals`, `satellites`}.
- `noise_positions(pos, ra=0, dec=90, noise_radial=σr, noise_transverse=σt)`
  applies separable Gaussian kernels; with `(ra,dec)=(0,90)` radial = +z axis.
- `MA(pos, L, N, MAS='TSC')` — positions must be float32. Returns overdensity
  `δ` ready for `calcPk`.

## Voxelization idioms

To simulate "CHARM-output halos redistributed inside their voxel" (common
diagnostic):

```python
delta = L / N
idx = np.clip(np.floor(hpos / delta).astype(int), 0, N-1)
centers = (idx + 0.5) * delta
offsets = rng.uniform(-delta/2, delta/2, size=hpos.shape)
hpos_vox = (centers + offsets) % L
```

Do NOT rely on the `noise_uniform` flag in `zheng_biased.yaml` to do this —
that applies uniform noise **post-HOD to galaxies** (for CHARM-produced halos
stored *at* voxel centers). When you want to voxelize in a custom pipeline,
set `bias.hod.noise_uniform=False` and do it yourself at the halo level.

## CHARM specifics

- External package: `from charm.infer_halos_from_PM import get_model_interface`
  (see `cmass/bias/rho_to_halo.py::apply_charm`).
- Pre-trained at `Npix=128`, `L=1000`, `pad=4`.
- Input: `rho`, `fvel` (both 128³×3) from FastPM — usually at
  `quijotelike/fastpm/L1000-N128/{lhid}/nbody.h5`.
- Output tuple includes `hconc`, but `halos.h5` on disk typically only stores
  `pos, vel, mass`. When concentration isn't saved, `populate_hod` falls back
  to the mass–concentration relation (this is the common path on this system).
- Minimum halo mass ≈ `log10(5e12) ≈ 12.7`.

## Inference conventions

`cmass/infer/preprocess.py`:

- `load_summaries(suitepath, tracer, Nmax, ...)` walks all lhid subdirs under
  `suitepath` and pools their `diag/.../hodNNNNN.h5` summaries in parallel.
- `split_train_val_test(x, theta, ids, val_frac, test_frac, seed)` splits by
  **unique lhid** (after shuffling), so all samples from a given lhid land in
  exactly one partition. Defaults in `infer/default.yaml`:
  `val_frac=0.1, test_frac=0.1, seed=0`.
- HOD priors are reconstructed from the first lhid's `config.yaml`; noise
  priors are reconstructed from the suite via `_construct_noise_prior`.
- Training supports both NPE (`train.py`, `validate.py`) and NLE
  (`train_nle.py`, `validate_nle.py`). Backend default: `lampe`. Engine: `NPE`.

## User preferences (from durable history)

- **Don't execute long-running scripts** unless told. Write, review, hand off.
- **scripts/** for new ad-hoc tools; **figures/** for outputs. Overwrite is OK.
- Don't save intermediate catalogs; a single `.npz` for measurements is fine.
- No timestamping; overwrites are acceptable.
- Keep comments sparse; no explanatory prose in commits unless asked.
- Prefer reusing repo helpers (HOD, noise, P(k), RSD) over reimplementing.
- When a CLAUDE.md exists at repo root, it is **the active task spec** — read
  it first and follow its "preflight" instructions literally.
- Flag ambiguity rather than improvising; ask clarification questions up front
  on new visualization/analysis tasks (slice geometry, subsampling, etc.).
- **Tests don't run locally.** Don't try `pytest`.

## Frequent gotchas

- `Pk_library.MA` error "Buffer dtype mismatch, expected float32_t but got
  double" → cast `pos.astype(np.float32)`.
- `populate_hod` needs `parse_hod(cfg)` first to resolve `theta`.
- `cfg.bias.hod.seed=0` means "use defaults" (no param sampling) — not "RNG
  seed 0". For RNG control in `populate_mock` pass `seed=` to `populate_hod`.
- When building a minimal OmegaConf for `parse_hod`, include `meta.wdir`,
  `sim`, `nbody.{suite,L,N,lhid,cosmo,zf}`, and the full bias section.
- Halo files have `mass = log10(M)`. If you feed `10**mass` to halotools, it
  expects Msun/h.
- CHARM halos file at `.../quijotelike/fastpm/.../halos.h5` has the same
  group/dataset schema as Quijote's — you can treat them interchangeably in
  loader code.
- The `latin_hypercube_params.txt` has 4001 rows but not every lhid has sim
  outputs on disk — scan directories and intersect before iterating.

## When you start a new script

1. Check `/work/hdd/bdne/maho3/cmass-ili/` for the intersection of
   `suite × lhid` you need. Don't assume.
2. Prefer loading precomputed `diag/.../hodNNNNN.h5` summaries over rerunning
   HOD + P(k).
3. Reuse `populate_hod`, `noise_positions`, `MA/calcPk`. Cast float32 where
   needed.
4. For any plot: output to `figures/` as PNG ≥150 dpi. Title should name the
   lhid(s), space (real/z), and any non-default knobs.
5. Put the script in `scripts/` with an argparse CLI and sensible defaults.
   The user will run it on a compute node.
