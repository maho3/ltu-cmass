# Pylians vs pypower for periodic-box P(k) in `cmass.diagnostics.summ`

**TL;DR:** The collaborators are right about accuracy. At the production mesh
(N=128 per Gpc/h, TSC), pylians' compensated-but-uninterlaced estimator is
aliased **+19% in P0 and +27% in P2 (z-space) at k=0.4** — i.e. right at the
kmax used in inference. pypower with `interlacing=2` on the *same* N=128 mesh
stays **within 0.5% of truth over the full k-range**, making the `high_res`
(N=256) pathway unnecessary for P(k). The cost: pypower is ~3–4× slower per
measurement at matched N (0.8 s vs 0.2 s at N=128), which is negligible in the
pipeline. Recommendation: **switch the box P(k) to pypower with interlacing=2,
keeping N=128** — but note this changes the training data vector at
k ≳ 0.25, so summaries must be recomputed consistently (no mixing backends
between training and inference).

## Setup

- Data: 11 Quijote halo boxes (`/work/hdd/bdne/maho3/cmass-ili/quijote/nbody/
  L1000-N128`, lhids 33 81 150 350 538 614 663 1019 1269 1627 1694; seed-0
  draw + the interactive test lhid). L=1000 Mpc/h, z=0.5, ~4–7×10⁵ halos.
- Real space and z-space (RSD along axis=2 via
  `get_redshift_space_pos`), P0/P2/P4.
- Pylians pathway is exactly the `summ.py` path: `MA` (TSC) → `PKL.Pk`
  (window deconvolution, no interlacing). pypower: `CatalogMesh` +
  `MeshFFTPower`, TSC, `los='z'`, k_F-width bin edges matched to pylians;
  everything then mode-weight rebinned onto the production dk=0.01 grid.
- **Ground truth:** pypower N=512 + interlacing=2 (k_Nyq=1.6, fully converged
  at k≤0.4). Cross-check: pylians N=512 agrees with it to **<0.2%** at all
  k≤0.4 (`figures/pk_backend_truthcheck.png`) — the two codes' conventions
  (normalization, window deconvolution, binning) are consistent; the only real
  difference at production settings is aliasing control.
- Shot noise: pylians does not subtract it; pypower subtracts
  `poles.shotnoise` from P0 by default. All comparisons here use pypower's
  **raw** monopole (`remove_shotnoise=False`) to match the pylians/summ.py
  convention.
- Runs on one Delta cpu node (job 20035715), 16 threads, best-of-3 timings.

## Accuracy / anti-aliasing

Ratio to truth, z-space, mean over 11 boxes
(`figures/pk_backend_ratio_{real,zspace}.png`). Binning was extended from the
original kmax=0.4 to kmax=0.6 to see how far past the N=128 production
Nyquist (k_Nyq = πN/L ≈ 0.402 for N=128, L=1000) the N=256/N=512 configs
hold up:

| Config | P0 @ 0.3 | P0 @ 0.4 | P0 @ 0.5 | P0 @ 0.6 | P2 @ 0.3 | P2 @ 0.4 | P2 @ 0.5 | P2 @ 0.6 |
|---|---|---|---|---|---|---|---|---|
| pylians N=128 (production) | +0.8% | **+19.0%** | — | — | +1.0% | +27.5% | — | — |
| pypower N=128, no interlacing | −0.4% | −1.2% | — | — | −1.3% | −5.3% | — | — |
| pypower N=128, interlacing=2 | +0.00% | +0.5% | — | — | +0.00% | +0.1% | — | — |
| pylians N=256 (high_res) | +0.01% | +0.02% | +0.2% | **+0.7%** | −0.05% | −0.5% | +0.1% | **−2.9%** |
| pypower N=256, interlacing=2 | 0.00% | 0.00% | 0.00% | +0.01% | 0.00% | 0.00% | −0.01% | −0.2% |

(`—` = no data: k=0.5 and 0.6 lie beyond the N=128 mesh's Nyquist, so those
bins have zero modes and are NaN in the rebinned output, not silently
truncated.)

- The pylians N=128 aliasing turns on at k ≳ 0.25 (≳1% by k≈0.3) and blows up
  approaching k_Nyq=0.4. This is inherited by every summary computed through
  `calcPk` at production resolution.
- Interlacing, not the code, is the differentiator: pypower *without*
  interlacing is also biased (a few % low; its aliasing residual has opposite
  sign because compensated TSC over-corrects the raw aliased power).
- With interlacing=2, N=128 is already converged over its accessible range:
  the last (edge-truncated) bin at k≈0.40 is the only one off, at +0.5% (P0).
- At the extended range, N=256 without interlacing (pylians) starts drifting
  again as k approaches its own Nyquist (0.804): P0 is still sub-percent at
  k=0.6, but P2 is off by −2.9% (z-space) — the same aliasing mechanism as the
  N=128 case, just pushed out by a factor of 2 in k. pypower N=256+interlacing
  stays <0.2% out to k=0.6 for both P0 and P2, i.e. interlacing continues to
  do the work, not the extra resolution.
- Real-space results are identical in character (P0: +10% pylians bias at
  k=0.4; P2/P4 ratios are noise-dominated since those multipoles ≈ 0, so their
  percent deviations swing more — e.g. pypower N=256 i2 shows +9.8% on real
  P2 at k=0.6, which is noise on a near-zero signal, not a real bias).

## What changes in the SBI data vector

`figures/pk_backend_prod_diff.png`: swapping pylians → pypower(i2) at N=128
leaves the data vector unchanged (<0.1% of P0) for k < 0.25, then removes the
aliasing excess: −1.5% at k≈0.3, −16% at k≈0.4 (P0, both spaces; P2 −4% of P0
at k=0.4 in z-space). Experiments with kmax ≤ 0.2 are unaffected; kmax ≥ 0.3
experiments will see systematically different spectra. Note `infer/default.yaml`
sweeps `kmax` up to 0.5 — any such experiment run at N=128 is *already* past
that mesh's Nyquist (0.402), regardless of backend, so kmax ∈ {0.4, 0.5}
configs need N=256 (with interlacing) to be measuring real power rather than
aliased/absent modes. **Training and inference must use the same backend** —
a model trained on pylians summaries cannot be applied to pypower summaries
at kmax ≳ 0.25. Since the aliasing is
present identically in both simulation and (simulated) observation, the
current pylians-based inferences are internally consistent, not wrong — the
gain from switching is accuracy of the physical spectra and robustness when
comparing to external codes/real data.

## Speed

Per measurement (paint + FFT + multipoles), 16 threads, mean over lhids
(`figures/pk_backend_timing.png`):

| Config | time |
|---|---|
| pylians N=128 | 0.22 s |
| pypower N=128 i0 | 0.49 s |
| pypower N=128 i2 | 0.79 s |
| pylians N=256 | 0.90 s |
| pypower N=256 i2 | 3.5 s |
| pylians N=512 | 6.5 s |
| pypower N=512 i2 | 26 s |

pylians is consistently ~3–4× faster at matched settings (interlacing itself
costs ~1.6× within pypower). But in absolute terms 0.8 s/measurement is
negligible next to HOD population and the polybin bispectrum in a `summ` call.
Notably **pypower N=128 i2 (0.8 s) is both faster and more accurate than the
current high_res pylians N=256 path (0.9 s)** — the high_res flag could be
retired for P(k) if pypower is adopted. pypower also MPI-parallelizes across
ranks (as already used for lightcones), which pylians cannot.

## Practical notes for a swap

- pypower is now installed in the Delta `cmass` env (mpi4py rebuilt from
  source against Cray MPICH — the manylinux wheel fails to find libmpi;
  pmesh/pfft built from git). The same dance will be needed on Anvil/Bridges2.
- Keep `remove_shotnoise=False` for P0 to preserve the current data-vector
  convention (noise level is a learned nuisance in the SBI setup anyway).
- pypower bins on k-shell edges (bin value = mean over modes in shell) vs
  pylians' mode-averaged k_F bins — after the existing `rebin_pk`
  mode-weighted coarsening these agree to <0.2% (truth cross-check), so the
  fixed-grid convention in `calculations.py` carries over unchanged.
- White-noise validation: both codes recover P = 1/n̄ on a Poisson catalog
  (`power_tests/whitenoise_check.py`).

## Recommendation

Adopt **pypower, TSC, interlacing=2, N=128 per Gpc/h** for periodic-box
multipoles in `summ.py`, drop `high_res` for P(k), and regenerate summaries
for any suite used with kmax > 0.2 before retraining. Keep pylians for
painting-only uses (`MA` fields feeding polybin) since PolyBin handles its own
pixel window. For any experiment with kmax > 0.4 (i.e. past the N=128
Nyquist), this recommendation does not apply as-is — bump to **N=256 with
interlacing=2**, which stays <0.2% accurate out to k=0.6 (extended-range test
above), instead of trying to push N=128 further.

## Reproducing

```bash
PYTHONPATH=. python power_tests/compare_backends.py --lhid <lhid>   # per box
sbatch power_tests/slurm_compare.sh                                 # 10 boxes
PYTHONPATH=. python power_tests/analyze.py                          # figures
```

Outputs: `data/scratch/power_tests/compare_lhid*.npz`,
`figures/pk_backend_{ratio_real,ratio_zspace,truthcheck,prod_diff,timing}.png`.
