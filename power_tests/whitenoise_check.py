"""Sanity check: white-noise catalog should give P(k) = 1/nbar in both backends."""
import numpy as np
from cmass.diagnostics.calculations import MA, calcPk
from pypower import CatalogMesh, MeshFFTPower

L, N, Npart = 1000.0, 256, 100_000
nbar = Npart / L**3
rng = np.random.default_rng(42)
pos = rng.uniform(0, L, (Npart, 3)).astype(np.float32)

# pylians (no shot-noise subtraction)
delta = MA(pos, L, N, MAS='TSC')
k, Pk, Nmodes = calcPk(delta, L, MAS='TSC', threads=8)
mask = k < 0.3
print(f"expected 1/nbar = {1/nbar:.1f}")
print(f"pylians  <P0>(k<0.3) = {np.average(Pk[mask,0], weights=Nmodes[mask]):.1f}")

# pypower
mesh = CatalogMesh(data_positions=pos, boxsize=L, boxcenter=L/2, nmesh=N,
                   resampler='tsc', interlacing=2, position_type='pos')
kedges = np.arange(0, 0.41, 2*np.pi/L)
res = MeshFFTPower(mesh, edges=kedges, ells=(0, 2, 4), los='z')
poles = res.poles
print(f"pypower shotnoise attr = {poles.shotnoise:.1f}")
kp = poles.k
m = kp < 0.3
p0_sub = poles(ell=0, complex=False)          # default: shot noise removed
p0_raw = poles(ell=0, complex=False, remove_shotnoise=False)
print(f"pypower  <P0 raw>(k<0.3) = {np.nanmean(p0_raw[m]):.1f}")
print(f"pypower  <P0 sub>(k<0.3) = {np.nanmean(p0_sub[m]):.1f}")
