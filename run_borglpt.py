import sys
import numpy as np
import borg

# params
args = sys.argv

print(args)
L = int(args[1])
N = int(args[2])

# initialize box and chain
box = borg.forward.BoxModel()
box.L = (L,L,L)
box.N = (N,N,N)

chain = borg.forward.ChainForwardModel(box)
chain.addModel(borg.forward.models.HermiticEnforcer(box))
chain.addModel(borg.forward.models.Primordial(box, 0.1))

# Set up cosmo
cpar = borg.cosmo.CosmologicalParameters()

sigma8_true = np.copy(cpar.sigma8)
cpar.sigma8 = 0
cpar.A_s = 2.3e-9 #will be modified to correspond to correct sigma

# Set-up CLASS:
k_max, k_per_decade = 10, 100

extra = {}
extra['YHe'] = '0.24'

cosmo = borg.cosmo.ClassCosmo(cpar, k_per_decade, k_max, extra=extra)
cosmo.computeSigma8() #will compute sigma for the provided A_s
cos = cosmo.getCosmology()

# Update A_s
cpar.A_s = (sigma8_true/cos['sigma_8'])**2*cpar.A_s


chain.addModel(borg.forward.model_lib.M_PRIMORDIAL_AS(box)) # Add primordial fluctuations

transfer_class=borg.forward.model_lib.M_TRANSFER_CLASS(box,opts={"a_transfer":1.0,"use_class_sign":False}) # Add CLASS transfer function
transfer_class.setModelParams({"extra_class_arguments":{"YHe":"0.24"}})
chain.addModel(transfer_class)

# add lpt
lpt = borg.forward.models.BorgLpt(box=box, box_out=box, ai=0.1, af=1.0, supersampling=1)
chain.addModel(lpt)

# forward model
ic = np.fft.rfftn(np.random.randn(*box.N))/box.Ntot**(0.5)
chain.forwardModel_v2(ic)
# rho = np.empty(chain.getOutputBoxModel().N)
# chain.getDensityFinal(rho)

# print(rho.shape)
print('done')
