# SBI Inference Pipeline

Notes on the neural-posterior-estimation (NPE) pipeline that maps galaxy
summary statistics → cosmology (+ noise nuisance) posteriors. Four stages, each
a Hydra entry-point run as a SLURM job:

```
preprocess  →  optuna  →  retrain  →  validate
(infpre)      (infoptuna) (infretrain) (infvalid)
```

Common config for the current run (`jobs/slurm_inf*.sh`):
`nbody=quijotelike sim=fastpm_charm6_rebin infer=simple net=niall2`,
`infer.embedding_net=fun`, `infer.tracer=galaxy`, `nbody.zf=0.5`, `device=cpu`,
`infer.include_noise=True infer.include_hod=False`, `infer.Nmax=4000`.

Outputs live under `{wdir}/{suite}/{sim}/models/{tracer}/{summary}/kmin-{kmin}_kmax-{kmax}/`.
Backend: **lampe**; engine: **NPE**; density estimator: **NSF** (neural spline flow).

The inference relies on a customized fork of `ltu-ili`
(`/u/maho3/git/ltu-ili`, branch `ltu` off `main`; `git diff main..ltu`). The
relevant customizations are described inline below.

---

## 0. Experiment grid

An "experiment" = a set of summaries + a (kmin, kmax) cut. Defined in
`conf/infer/simple.yaml::experiments`. Current grid (z-space only), 16 total:

| summaries | kmax values (kmin=0) |
|---|---|
| `zPk0` | 0.2, 0.3, 0.4, 0.5, 0.6 |
| `zPk0, zPk2, zPk4` | 0.2, 0.3, 0.4, 0.5, 0.6 |
| `zPk0, zPk2, zPk4, zBk0` | 0.2, 0.3, 0.4 |
| `zPk0, zPk2, zPk4, zEqBk0` | 0.2, 0.3, 0.4 |

**Targets `theta`** (in order): `[Ωm, Ωb, h, ns, σ8]` from `config.yaml`, then
noise params `[σ_radial, σ_transverse]` (since `include_noise=True`). If
`include_hod=True`, the HOD parameters are inserted between cosmo and noise. For
the `zheng07` model with assembly + velocity assembly bias these are:

| parameter | range | note |
|---|---|---|
| `logMmin` | 12.0–14.0 | min halo mass for a central |
| `sigma_logM` | 0.1–0.6 | central occupation softening |
| `logM0` | 13.0–15.0 | satellite cutoff mass |
| `logM1` | 13.0–15.0 | satellite normalization mass |
| `alpha` | 0.0–1.5 | satellite power-law slope |
| `mean_occupation_centrals_assembias_param1` | −1…1 (truncnorm, σ=0.2) | central assembly bias |
| `mean_occupation_satellites_assembias_param1` | −1…1 (truncnorm, σ=0.2) | satellite assembly bias |
| `eta_vb_centrals` | 0.0–0.7 | central velocity bias |
| `eta_vb_satellites` | 0.2–2.0 | satellite velocity bias |
| `conc_gal_bias_satellites` | 0.2–2.0 | satellite concentration bias |

---

## 1. Preprocess (`cmass.infer.preprocess`)

Aggregates raw per-sim `diag/galaxies/hodNNNNN.h5` summaries into train/val/test
`.npy` tensors, one directory per (summary, kmin, kmax). Each
(lhid, HOD seed, noise realization) file is one sample; cosmology is shared
across a lhid's samples.

### Per-summary transforms (`preprocess_Pk`, `preprocess_Bk`)

All summaries are first cut to the band $k \in [k_{\min}, k_{\max}]$, then
processed per multipole and concatenated along the feature axis. NaNs → 0.

**Power spectrum monopole** $P_0(k)$ (and $P_0^{s}$ in redshift space).
Optionally shot-noise-subtracted ($\bar n^{-1}$, `correct_shot=true` default),
then signed-log compressed to tame the ~5-decade dynamic range and keep the
large-$k$ tail on scale:
$$
\tilde P_0(k) = \mathrm{sign}\!\big(P_0\big)\,\log_{10}\!\big(1 + |P_0|\big).
$$

**Higher power multipoles** $P_2, P_4$ (real or redshift space). Divided by the
(uncompressed) monopole to form a dimensionless ratio rather than logged:
$$
\tilde P_\ell(k) = \frac{P_\ell(k)}{P_0(k)}, \quad \ell \in \{2, 4\}.
$$

**Bispectrum monopole** $B_0$. Triangles are first selected by a
configuration mask (label tags): default = all $k$-triples in band satisfying
the triangle inequality; `Eq` = equilateral, `Sq` = squeezed
($k_2 \simeq k_3,\ k_1 < k_2$), `Is` = isoceles, `Ss` = subsampled (every 5th
triangle). The current grid uses full triangles (`zBk0`) and equilateral
(`zEqBk0`). Then signed-log compressed as for $P_0$.

**Higher bispectrum multipoles / reduced bispectrum** ($B_2$, $Q_\ell$): divided
by the $B_0$ monopole (same masking), analogous to the $P_\ell$ ratio.

$\bar n$ and $n(z)$ can optionally be appended as extra (log10) features; not
used for box summaries here.

### Split (`split_train_val_test`)

Grouped by **unique lhid** so all samples from one lhid stay in one partition (no
leakage). Assignment is a stable per-lhid hash of `(seed, lhid)` → uniform draw
→ threshold, so adding/removing lhids never reshuffles the existing ones.
`val_frac=0.1, test_frac=0.1, seed=0`.

### Priors

Reconstructed from the first sim's config:
- **Cosmo** (`prior=quijote`, hard-coded Quijote box, all uniform):
  Ωm (0.1, 0.5), Ωb (0.03, 0.07), h (0.5, 0.9), ns (0.8, 1.2), σ8 (0.6, 1.0).
- **Noise**: `noise_dist` attr from a diag file → matching `conf/noise/<dist>.yaml`
  bounds (e.g. `reciprocal` → Reciprocal(0.1, 4.5105)).
- **HOD** (if enabled): the model's per-parameter bounds/distributions above.

---

## 2. Optuna hyperparameter search (`cmass.infer.optuna`)

We do **~400 Optuna trials** per experiment to find the best density-estimator
architectures + training hyperparameters. Sampler: **TPE**
(`multivariate=True`, `n_startup_trials=50`), maximizing the objective below.
This stage only records trial scores + configs into `optuna_study.db`; it does
**not** save trained models. The winning architectures are re-instantiated and
saved to disk in the retrain stage.

### Objective

For each trial, sample an architecture + hyperparameters (table below), train
it, and score it by **mean posterior log-probability on held-out data**,
$\frac{1}{N}\sum_i \log q(\theta_i \mid x_i)$. To make the score robust and avoid
touching the final preprocess test set, scoring uses **2-fold grouped
cross-validation** (`n_splits=2`): `GroupShuffleSplit` on lhid pools all
data, retrains on each fold's train+val, evaluates on the fold's test split, and
returns the mean over folds. Higher is better.

### Searched hyperparameters (`net=niall2`)

| parameter | range / values | scale |
|---|---|---|
| `model` | `nsf` | fixed |
| `hidden_features` | 8–32 | log-int |
| `num_transforms` | 1–5 | int |
| `batch_size` | $2^{5}$–$2^{9}$ | log2-int |
| `learning_rate` | 1e-4 – 1e-1 | log |
| `weight_decay` | 0 | fixed |
| `max_epochs` | 50–1000 | log-int |
| `noise_percent` | 0.01 | fixed |
| `dropout` | 0 | fixed |
| `early_stopping` | False | fixed |
| `lr_scheduler` | `CosineAnnealingLR` | fixed |
| embedding `hidden_depth` (FunnelNetwork) | 0–3 | int |
| embedding `out_features` | 4–16 | log-int |
| embedding `bypass` | True | fixed |

### Training loop (customized `ltu-ili` `runner_lampe.py`)

Each trial trains one NSF flow with a FunnelNetwork embedding via:

- `stop_after_epochs=30` — patience for the convergence check.
- `clip_max_norm=5` — gradient-norm clipping.
- `validation_fraction=0.1`.
- `lr_scheduler=CosineAnnealingLR` with `T_max=max_epochs`.
- `noise_percent=0.01` — **input-noise augmentation**: a `NoisyDataset` wrapper
  adds $\mathcal N(0, 0.01\cdot\sigma_{\text{feature}})$ Gaussian noise to the
  features on every batch draw, acting as the regularizer in place of dropout /
  weight decay.
- `validation_smoothing_method=ema`, `ema_decay=0.9` — **validation-loss
  smoothing** through `swa_utils.AveragedModel`; convergence / best-model
  decisions use the EMA-smoothed val loss instead of the noisy raw curve.
- `early_stopping=False` — train the full `max_epochs`.

(The scheduler options `ReduceLROnPlateau`/`LambdaLR` and `swa` smoothing are
also available in the fork but unused here.)

---

## 3. Retrain (`cmass.infer.retrain`, = `train.py` with `retrain=True`)

Optuna found and scored the best architectures but discarded the weights; this
stage **materializes them on disk**. It reads the finished study, takes the
**top `Nnets=10` trials by CV log-prob** (`select_top_trials`), and retrains each
selected architecture **from scratch on the original preprocess train/val split**
(the split that CV never touched), saving into `nets/net-{trial_number}/`. Runs
skip if `posterior.pkl` already exists. SLURM array parallelizes over the 10
nets.

---

## 4. Validate (`cmass.infer.validate`)

Builds a **weighted ensemble** (`LampeEnsemble` from the fork) of the top-`Nnets`
retrained posteriors, weighted by `softmax(CV log-prob)`. `clean_models=True`
deletes net dirs outside the top set.

Evaluated on the held-out preprocess **test set** (`x_test.npy`,
`theta_test.npy`), or an external suite if `infer.testing.suite/sim` is set (e.g.
`quijote/nbody_hodz_gridnoise`, `abacus1gpch/custom_hodz_gridnoise` for
OOD/robustness tests, written to `testing/<suite>_<sim>/`).

Diagnostics produced:
- Single-posterior corner plot at a test point (`PlotSinglePosterior`).
- Per-member overplotted corner (`PlotSinglePosteriorEnsemble`).
- `PosteriorCoverage` with: `coverage`, `histogram`, `predictions`, `tarp`,
  `logprob` (2000 samples, direct sampling); posterior samples saved.
- Optuna study diagnostics: optimization history, slice, parameter importance,
  timeline.
</content>
