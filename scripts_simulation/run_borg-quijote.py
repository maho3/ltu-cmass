import sys
import os
from os.path import join
import time
import argparse
import numpy as np

os.environ["PYBORG_QUIET"] = "yes"
import borg

# This retrieve the console management object
console = borg.console()
# Reduce verbosity
console.setVerboseLevel(1)


# get config 
parser = argparse.ArgumentParser()
parser.add_argument('--cind', type=int, required=True)
parser.add_argument('--fromIC', action='store_true')
args = parser.parse_args()

print(args)


cind = args.cind
L = 1000
N = 512
zi = 127
zf = 0.
supersampling=1

ai = 1/(1+zi)
af = 1/(1+zf)

transfer = 'EH' # 'CLASS' # 


# define fucnctions
def get_params(index):
    if index=="fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open('latin_hypercube_params.txt','r') as f:
        content = f.readlines()[index+1]
    content = [np.float64(x) for x in content.split()]
    return content

def build_cosmology(pars):
    cpar = borg.cosmo.CosmologicalParameters()
    cpar.default()
    cpar.omega_m, cpar.omega_b, cpar.h, cpar.n_s, cpar.sigma8 = pars
    return cpar

def transfer_EH(chain, box, cpar):
    chain.addModel(borg.forward.models.Primordial(box, ai))
    chain.addModel(borg.forward.models.EisensteinHu(box))

def transfer_CLASS(chain, box, cpar):
    # not currently used
    sigma8_true = np.copy(cpar.sigma8)
    cpar.sigma8 = 0
    cpar.A_s = 2.3e-9 #will be modified to correspond to correct sigma
    cosmo = borg.cosmo.ClassCosmo(cpar, k_per_decade=10, k_max=50, extra={'YHe':'0.24'})
    cosmo.computeSigma8() #will compute sigma for the provided A_s
    cos = cosmo.getCosmology()
    # Update A_s
    cpar.A_s = (sigma8_true/cos['sigma_8'])**2*cpar.A_s
    chain.addModel(borg.forward.model_lib.M_PRIMORDIAL_AS(box)) # Add primordial fluctuations
    transfer_class=borg.forward.model_lib.M_TRANSFER_CLASS(box,opts={"a_transfer":ai,"use_class_sign":False}) # Add CLASS transfer function
    transfer_class.setModelParams({"extra_class_arguments":{"YHe":"0.24","z_max_pk":"200"}})
    chain.addModel(transfer_class)

def load_modes(fn):
    """Loading in Fourier space."""
    num_mesh_1d = N
    num_modes_last_d = num_mesh_1d // 2 + 1
    with open(fn, 'rb') as f :
        num_read = np.fromfile(f, np.uint32, 1)[0]
        modes = np.fromfile(f, np.complex128, num_read).reshape((num_mesh_1d, num_mesh_1d, num_modes_last_d))
    return modes
    
def run_density(cpar):
    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = (L,L,L)
    box.N = (N,N,N)

    chain = borg.forward.ChainForwardModel(box)
    chain.addModel(borg.forward.models.HermiticEnforcer(box))

    if transfer=='CLASS':
        transfer_CLASS(chain, box, cpar)
    elif transfer=='EH':
        transfer_EH(chain, box, cpar)

    # add lpt
    lpt = borg.forward.models.Borg2Lpt(
        box=box, box_out=box, 
        ai=ai, af=af, 
        supersampling=supersampling
    )
    chain.addModel(lpt)
    chain.setCosmoParams(cpar)

    # generate ICs
    if args.fromIC:
        path_to_wn = f'/home/mattho/data/cmass-ili/borg-quijote/ICs/wn_{cind}.dat'
        print(f'Loading ICs from {path_to_wn}...')
        ic = load_modes(path_to_wn)
    else:
        print('Generating new ICs...')
        ic = np.fft.rfftn(np.random.randn(*box.N))/box.Ntot**(0.5)
    
    # forward model
    
    print('Running forward...')
    chain.forwardModel_v2(ic)
    
    print('Storing...')
    Npart = lpt.getNumberOfParticles()
    rho = np.empty(chain.getOutputBoxModel().N)
    pos = np.empty(shape=(Npart,3))
    vel = np.empty(shape=(Npart,3))
    chain.getDensityFinal(rho)
    lpt.getParticlePositions(pos)
    lpt.getParticleVelocities(vel)
    
    return rho, pos, vel


# Set up cosmo
content = get_params(cind)
print(content)

datadir = f'/home/mattho/git/cmass-ili/data/borg-quijote/latin_hypercube_HR-{N}/{cind}'
os.makedirs(datadir, exist_ok=True)
print('I will save to:', datadir)

t0 = time.time()
cpar = build_cosmology(content)
rho, pos, vel = run_density(cpar)

print(f"Done running. I took {time.time()-t0:.1f} seconds.")
# save
np.save(join(datadir,'rho.npy'), rho)
np.save(join(datadir, 'ppos.npy'), pos)
np.save(join(datadir,'pvel.npy'), vel)
print(f"Saved to {datadir}")
