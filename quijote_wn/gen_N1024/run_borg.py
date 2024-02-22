import os
import numpy as np
import borg
import time
from os.path import join as pjoin
import argparse


def initial_pos(L, N, order="F"):
    values = np.linspace(0, L, N+1)[:-1]  # ensure LLC
    xx, yy, zz = np.meshgrid(values, values, values)

    if order == "F":
        pos_mesh = np.vstack((yy.flatten(), xx.flatten(), zz.flatten())).T
    if order == "C":
        pos_mesh = np.vstack((zz.flatten(), xx.flatten(), yy.flatten())).T

    return pos_mesh


def load_modes(fn):
    """Loading in Fourier space."""
    num_mesh_1d = 1024
    num_modes_last_d = num_mesh_1d // 2 + 1
    with open(fn, 'rb') as f:
        num_read = np.fromfile(f, np.uint32, 1)[0]
        modes = np.fromfile(f, np.complex128, num_read).reshape(
            (num_mesh_1d, num_mesh_1d, num_modes_last_d))
    return modes


def run_forward(s_hat, fwd):
    start_time = time.time()
    rho_desc = np.zeros(fwd.getOutputBoxModel().N)
    fwd.forwardModel_v2(s_hat)
    fwd.getDensityFinal(rho_desc)
    print("--- Full forward pass took %s seconds ---" %
          (time.time() - start_time))
    return rho_desc


def build_gravity_model_test(bb, f_cpar, nr, Eisenstein):
    """
    nr (int) = what params to extract
    """

    # Initialize some default cosmology
    cpar = borg.cosmo.CosmologicalParameters()

    with open(f_cpar) as f:
        lines = [float(param) for param in f.readlines()[
            nr+1].split()]  # nr+1 bcz row0 is header
        cpar.omega_m = float(lines[0])
        cpar.omega_b = float(lines[1])
        cpar.omega_q = 1-cpar.omega_m
        cpar.h = float(lines[2])
        cpar.n_s = float(lines[3])
        sigma_8_quij = float(lines[4])

    print('Cosmology is = ', cpar)

    # Store omega_m as style_param
    Om = cpar.omega_m

    if Eisenstein:
        cpar.sigma8 = sigma_8_quij
    else:
        cpar.sigma8 = 0
        cpar.A_s = 2.3e-9  # will be modified to correspond to correct sigma

        # Let CLASS do the work
        k_max = 10
        k_per_decade = 100
        extra = {}
        extra['YHe'] = '0.24'

        cosmo = borg.cosmo.ClassCosmo(cpar, k_per_decade, k_max, extra=extra)
        cosmo.computeSigma8()
        cos = cosmo.getCosmology()
        print('Updating A_s to correspond to correct sigma_8')
        cpar.A_s = (sigma_8_quij/cos['sigma_8'])**2*cpar.A_s

        cosmo = borg.cosmo.ClassCosmo(cpar, k_per_decade, k_max, extra=extra)
        cosmo.computeSigma8()
        cos = cosmo.getCosmology()
        cpar.sigma8 = 0

        print('Cosmology = ', cpar)

    # Fiducial scale factor to express initial conditions
    a0 = 1
    print('a0 = ', a0)

    chain = borg.forward.ChainForwardModel(bb)

    # Add fluctuations and transfer
    chain.addModel(borg.forward.models.HermiticEnforcer(bb))

    if Eisenstein:
        chain.addModel(borg.forward.models.Primordial(
            bb, a0))  # Add primordial fluctuations
        # Add E&Hu transfer function
        chain.addModel(borg.forward.models.EisensteinHu(bb))

    else:
        # Add primordial fluctuations
        print('adding primordial fluctuations')
        chain.addModel(borg.forward.model_lib.M_PRIMORDIAL_AS(bb))
        print('transfer CLASS')
        transfer_class = borg.forward.model_lib.M_TRANSFER_CLASS(
            bb, opts={"a_transfer": 1.0, "use_class_sign": False})
        transfer_class.setModelParams(
            {"extra_class_arguments": {"YHe": "0.24"}})
        chain.addModel(transfer_class)

    # Run LPT model from a=0.0 to af. The ai=a0 is the scale factor at which the IC are expressed
    print('build lpt')
    lpt = borg.forward.models.BorgLpt(bb, bb, ai=a0, af=1.0)
    chain.addModel(lpt)

    # Set cosmology
    print('set cosmology')
    chain.setCosmoParams(cpar)

    return chain, lpt


def check(disp, L, moved_over_bound, max_disp_1d, i, axis):
    idxsup = disp[:, i] > moved_over_bound
    idx = np.abs(disp[:, i]) <= max_disp_1d
    idxsub = disp[:, i] < -moved_over_bound

    sup = len(disp[:, i][idxsup])
    did_not_cross_boundary = len(disp[:, i][idx])
    sub = len(disp[:, i][idxsub])

    if not sub+did_not_cross_boundary+sup == len(disp[:, i]):
        print(
            f'Disp in {axis[i]} direction under -{moved_over_bound} Mpc/h is = '+str(sub))
        print(
            f'|Disp| in {axis[i]} direction under {max_disp_1d} Mpc/h is = '+str(did_not_cross_boundary))
        print(
            f'Disp in {axis[i]} direction over {moved_over_bound} Mpc/h is = '+str(sup))
        print('These add up to: '+str(sub+did_not_cross_boundary+sup))
        print(f"Should add up to: len(disp[:,i]) {len(disp[:,i])}")
        print('\n')

        assert sub+did_not_cross_boundary + \
            sup == len(
                disp[:, i]), "Incorrect summation"  # cannot lose/gain particles

    return idxsup, idxsub


def correct_displacement_over_periodic_boundaries(disp, L, max_disp_1d=125):
    # Need to correct for positions moving over the periodic boundary

    moved_over_bound = L - max_disp_1d
    axis = ['x', 'y', 'z']

    for i in [0, 1, 2]:

        idx_sup, idx_sub = check(
            disp, L, moved_over_bound, max_disp_1d, i, axis)

        # Correct positions
        disp[:, i][idx_sup] -= L
        disp[:, i][idx_sub] += L

        _, _ = check(disp, L, moved_over_bound, max_disp_1d, i, axis)

        assert np.amin(disp[:, i]) >= -max_disp_1d and np.amax(disp[:, i]
                                                               ) <= max_disp_1d, "Particles outside allowed region"

    return disp


# Params
parser = argparse.ArgumentParser()
parser.add_argument('--lhid', type=int)  # which cosmology to use
args = parser.parse_args()
nr = args.lhid

use_float64 = True
Eisenstein = False
f_cpar = '/home/mattho/git/ltu-cmass/data/quijote/latin_hypercube_params_bonus.txt'
L, Ng = 1000, 1024
q = initial_pos(L, Ng, order="F")
box = borg.forward.BoxModel(L, Ng)
path_to_wn = f'/home/mattho/git/ltu-cmass/data/quijote/wn_quijote/wn_{nr}.dat'

# Run
print('Running...')
s_hat = load_modes(path_to_wn)
print('Building forward model...')
fwd, lpt = build_gravity_model_test(
    box, f_cpar=f_cpar, nr=nr, Eisenstein=Eisenstein)
print('run_forward...')
delta = run_forward(s_hat, fwd)

# Step 0 - Extract particle positions and velocities
print('Step 0')
pos = np.zeros((lpt.getNumberOfParticles(), 3))  # output shape: (N^3, 3)
vel = np.zeros((lpt.getNumberOfParticles(), 3))  # output shape: (N^3, 3)
lpt.getParticlePositions(pos)
lpt.getParticleVelocities(vel)

# Step 1 - find displacements
print('Step 1')
disp = pos - q  # output shape: (N^3, 3)

# Step 2 - correct for particles that moved over the periodic boundary
print('Step 2')
disp_temp = correct_displacement_over_periodic_boundaries(
    disp, L=L, max_disp_1d=L//2)

# Step 3 - reshaping initial pos, vel and displacement
print('Step 3')
dis_in_C = np.reshape(disp_temp.T, (3, Ng, Ng, Ng),
                      order='C')  # output shape: (3, N, N, N)
dis_in_C = dis_in_C[:, ::2, ::2, ::2]

vel = np.reshape(vel.T, (3, Ng, Ng, Ng), order='C')
vel = vel[:, ::2, ::2, ::2]

# Step 4 – Save
outdir = f'/home/mattho/git/ltu-cmass/data/quijote/borg1lpt/{nr}'
os.makedirs(outdir, exist_ok=True)
print(f'Saving to {outdir}')
np.save(pjoin(outdir, 'dis.npy'), dis_in_C)
np.save(pjoin(outdir, 'vel.npy'), vel)
np.save(pjoin(outdir, 'rho.npy'), delta)
