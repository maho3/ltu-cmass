# Replace pylians with pypower for periodic-volume P(k)

## Motivation

`power_tests/REPORT.md` (see also its kmax=0.6 extension) showed that
pylians' single-pass compensated TSC assignment aliases badly at the mesh
resolutions we use in production: +19% (P0) / +27% (P2) at k=0.4, turning on
around k gtrsim 0.25. `pypower` with `interlacing=2` on the *same* mesh stays
within 0.5% of a converged N=512 truth over the full range we measure.
This PR swaps the backend for the periodic-volume P(k) paths in
`cmass/diagnostics/summ.py` — nbody density (`rho`), halos, and galaxies —
from pylians to pypower+interlacing=2, leaving everything else (lightcones,
bispectrum) untouched.

## What changed

- **`cmass/diagnostics/calculations.py`**: added
  `calcPk_pypower(pos, L, N, axis, resampler, interlacing)` (catalog → P(k)
  multipoles via `pypower.CatalogMesh` + `MeshFFTPower`, matching the
  `compare_backends.py` methodology from `power_tests/`) and
  `calcPk_pypower_field(field, L, axis, MAS)` (same, but for an
  already-painted field via `pypower.ArrayMesh`, for the nbody density case).
  Both return `(k, Pk[:, :3], Nmodes)` in the same layout as the existing
  `calcPk`, so they drop straight into the existing `rebin_pk` (unchanged —
  the fixed K_MIN/DK_PK output grid is backend-agnostic).
- **`cmass/diagnostics/summ.py`**: added `run_pypower_box` (catalogs:
  halos/galaxies, interlacing=2) and `run_pypower_field` (pre-painted field:
  nbody `rho`, no interlacing — see caveat below). Replaced the `run_pylians`
  call sites in `summarize_rho` and the `Pk` block of `summarize_tracer`
  (real- and redshift-space) with these. `run_pylians` itself is untouched
  and still used for `summarize_lightcone_pylians` (survey/sky geometry,
  out of scope here — it already has a separate MPI `pypower` path via
  `summarize_lightcone_pypower`/`cmass/diagnostics/pypower.py`).
  Bispectrum (`run_bispectrum`, `calcBk_polybin`/`calcBk_bfast`) is also
  untouched — PolyBin handles its own pixel window, so pylians painting is
  still correct there (per `REPORT.md`'s recommendation).
- **Resolution unchanged**: `get_mesh_resolution` (in `tools.py`) is not
  modified. `diag.high_res` already defaults to `true`, giving
  N=256 cells / 1000 Mpc/h → voxel side ≈ 1000/256 ≈ 3.91 Mpc/h for halos and
  galaxies — this is the resolution the PR was asked to preserve, and it's
  exactly the mesh at which `power_tests/REPORT.md` shows pypower+i2 accurate
  to <0.2% out to k=0.6. Nothing needed to change here; only the backend
  computing P(k) *at* that resolution changed.
- **Redshift-space positions**: `summarize_tracer` now computes RSD positions
  via `get_redshift_space_pos` (already existed in `calculations.py`, used by
  `power_tests/compare_backends.py`) instead of painting through `MAz`, since
  `run_pypower_box` paints the catalog itself. Note it mutates in place, so
  callers must pass `.copy()` — done at the call site.

## A real caveat: `rho` (nbody density) gets no interlacing

The three periodic paths are not equivalent in how much they benefit:

- **halos/galaxies**: we paint the catalog ourselves in `summ.py`, so
  `pypower`'s `CatalogMesh(..., interlacing=2)` genuinely repaints twice with
  a half-cell shift and removes the aliasing — this is the full effect
  documented in `REPORT.md`.
- **`rho`**: this field arrives already painted by the N-body code
  (FastPM/BORG/etc, see `cmass/nbody/fastpm.py`) at whatever mesh the
  simulation ran at. We cannot retroactively interlace a field that was only
  painted once. So `run_pypower_field` only swaps the FFT/deconvolution
  backend for `rho` — same accuracy as before, verified to agree with the old
  pylians path to <0.5% (see Validation below). This is a backend-consistency
  change for `rho`, not an accuracy improvement; the improvement is specific
  to halos/galaxies.

## A subtlety in `calcPk_pypower_field`

`pypower.ArrayMesh` + `MeshFFTPower` derives its normalization from the
field's own mean (`mesh.csum() / volume`) when given a bare `RealField` (as
opposed to a `CatalogMesh`, which tracks its own particle counts). The
`rho` field stored in `nbody.h5` is the density *contrast* (mean 0,
`rho /= mean(rho); rho -= 1` in `cmass/nbody/fastpm.py`). Passing it directly
gives a near-zero normalization and blows up P(k) by ~14 orders of magnitude.
The fix (already applied) is to pass `1 + field` into `ArrayMesh` — this is
the density itself, mean ~1, matching what `MeshFFTPower`'s docstring assumes
for a `RealField` ("If RealField, assumed to be 1+delta"). This has no effect
on `P(k)` for k>0 (adding a constant only touches the k=0 mode, which is
excluded from every bin we measure), it only fixes normalization.

## Validation performed

Ran both old and new paths on real data (lhid 663,
`quijote/nbody/L1000-N128` for halos, `quijotelike/fastpm/L1000-N128` for
`nbody.h5`) and compared multipoles directly (see conversation for the
one-off script; not checked in):

- Halos, N=256 (current `high_res` default), real space: P0 ratio
  (new/old) is 1.000 ± 0.002 for k ≲ 0.6, degrading toward the N=256
  Nyquist (k≈0.80) as expected — pylians itself starts aliasing there too
  (consistent with `REPORT.md`'s N=256 finding at kmax=0.6).
- `rho` field, N=128: P0 ratio (new/old) is 1.000 ± 0.003 across the full
  range, confirming the field-based backend swap is accuracy-neutral (no
  interlacing available here, as expected).

**Not yet done**: a full pipeline run (`cmass.diagnostics.summ` end-to-end
on a SLURM node) to confirm timing/memory at scale, and no `pytest` run
(per repo convention, tests aren't run locally). Recommend a smoke run on
one lhid on Delta before wide rollout, and regenerating `diag/` summaries
for any suite that will be used with kmax ≳ 0.25 (old and new backends
disagree above that, same caveat as `REPORT.md`).

## Installation

`pypower` was already an optional extra (`pip install -e '.[pypower]'`,
`setup.cfg`) for the lightcone MPI path; it's now a hard dependency of
`cmass.diagnostics.summ` for any periodic-volume run (`diag.density`,
`diag.halo`, `diag.galaxy` with `Pk` in `diag.summaries`). Delta-specific
build instructions (mpi4py/pfft/pmesh from source against Cray MPICH) were
added to `docs/options/PYPOWER.md`. Anvil/Bridges2 will need the same
treatment before running periodic-box diagnostics there — see that doc.
