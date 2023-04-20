import sys
import os
from os.path import join
import time

os.environ["PYBORG_QUIET"] = "yes"

import numpy as np
import borg

# This retrieve the console management object
console = borg.console()
# Reduce verbosity
console.setVerboseLevel(1)


cind = "fid" # 10 # int(sys.argv[1])
L = 3000
N = 750
zi = 127
zf = 0.5
supersampling=1
Nsims = 100

ai = 1/(1+zi)
af = 1/(1+zf)

transfer = 'EH' # 'CLASS' # 


# define fucnctions
def get_params(index):
    if index=="fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open('latin_hypercube_params.txt','r') as f:
        content = f.readlines()[index]
    content = [float(x) for x in content.split()]
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
    lpt = borg.forward.models.BorgLpt(
        box=box, box_out=box, 
        ai=ai, af=af, 
        supersampling=supersampling
    )
    chain.addModel(lpt)
    chain.setCosmoParams(cpar)

    # forward model
    ic = np.fft.rfftn(np.random.randn(*box.N))/box.Ntot**(0.5)
    chain.forwardModel_v2(ic)
    rho = np.empty(chain.getOutputBoxModel().N)
    chain.getDensityFinal(rho)
    
    return rho


# Set up cosmo
content = get_params(cind)
print(content)

datadir = '/home/mattho/git/cmass-ili/data/borg/cmass'
fname = join(datadir, f'density_L{L}N{N}_quijLH{cind}.npy')
print('I will save to:', fname)

t0 = time.time()
cpar = build_cosmology(content)
rho = run_density(cpar)

print(f"Done running. I took {time.time()-t0:.1f} seconds.")
# save
np.save(fname, rho)
print(f"Saved to {fname}")
