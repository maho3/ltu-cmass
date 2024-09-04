import os  
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_ENABLE_X64"] = "True"
from pmwd import (Configuration, Cosmology, boltzmann, linear_modes,
                  lpt, nbody, scatter)
import jax.numpy as jnp
import logging
import numpy as np
from os.path import join as pjoin
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import readgadget
import MAS_library as MASL
import pickle as pk
import readfof


# open quijote:
root = '/mnt/home/fvillaescusa/ceph/Quijote/Snapshots/fiducial_HR'
snapnum = 3
snapshot = '%s/%d/snapdir_%03d/snap_%03d'%(root,ji,snapnum,snapnum)
ptypes       = [1]
grid         = 128
BoxSize = 1000.0 #Mpc/h ; size of box
pos = readgadget.read_block(snapshot, "POS ", ptypes)/1e3 #positions in Mpc/h
df_cic = np.zeros((grid,grid,grid), dtype=np.float32)
MASL.MA(np.float32(pos), df_cic, BoxSize, 'CIC', verbose=False)
df_quijote_z0p5 = df_cic/np.mean(df_cic, dtype=np.float64)-1.0



# settings for pmwd. You can change these to make it fit into GPU, particularly lowering supersampling/B
L = BoxSize           # Mpc/h
N_inp = grid            # meshgrid resolution
supersampling = 3  # particles resolution relative to meshgrid
B = 3.1           # force grid resolution relative to particle grid
N = N_inp*supersampling
ptcl_spacing = L/N
ptcl_grid_shape = (N,)*3

path_to_ic = 'test0_fidcosmo_%d'%(N_inp*supersampling)
num_modes_last_d = N // 2 + 1
with open(path_to_ic, 'rb') as f:
    _ = np.fromfile(f, np.uint32, 1)[0]
    modes = np.fromfile(f, np.complex128, -1)
    modes = modes.reshape((N, N, num_modes_last_d))


zi = 127          # initial redshift
zf = 0.5           # final redshift
ai = 1.0 / (1.0 + zi)  # initial scale factor
af = 1.0 / (1.0 + zf)  # final scale factor
Nsteps=49
pmconf = Configuration(ptcl_spacing, ptcl_grid_shape,
                        a_start=ai, a_stop=af,
                        a_nbody_num=Nsteps,
                        mesh_shape=B,
                        a_snapshots=(af, ai),
                        lpt_order=2)

pmcosmo = Cosmology.from_sigma8(
    pmconf, sigma8=0.834, n_s=0.9624, Omega_m=0.3175,
    Omega_b=0.049, h=0.6711)

pmcosmo = boltzmann(pmcosmo, pmconf)

ic_256 = linear_modes(modes, pmcosmo, pmconf)
ptcl_256_lpt, obsvbl_256_lpt = lpt(ic_256, pmcosmo, pmconf)


ptcl_256_z0p5_fid, obsvbl_256_z0p5_fid = nbody(ptcl_256_lpt, obsvbl_256_lpt, pmcosmo, pmconf)
scale = supersampling * B
rho_256_z0p5_fid = scatter(ptcl_256_z0p5_fid, pmconf,
                mesh=jnp.zeros(3*(N_inp,)),
                cell_size=pmconf.cell_size*scale)
rho_256_z0p5_fid /= scale**3  # renormalize
rho_256_z0p5_fid -= 1  # make it zero mean


# measure power spectrum:
import Pk_library as PKL
import MAS_library as MASL
Pk_quijote_z0p5 = PKL.Pk(df_quijote_z0p5, BoxSize, axis=0, MAS=None, threads=1)
Pk_pmwd_z0p5 = PKL.Pk(np.array(rho_256_z0p5_fid), BoxSize, axis=0, MAS=None, threads=1)
