import numpy as np
import aquila_borg as borg


def build_cosmology(omega_m, omega_b, h, n_s, sigma8):
    cpar = borg.cosmo.CosmologicalParameters()
    cpar.default()
    cpar.omega_m, cpar.omega_b, cpar.h, cpar.n_s, cpar.sigma8 = \
        (omega_m, omega_b, h, n_s, sigma8)
    cpar.omega_q = 1.0 - cpar.omega_m
    return cpar


def transfer_EH(chain, box, ai):
    chain.addModel(borg.forward.models.Primordial(box, ai))
    chain.addModel(borg.forward.models.EisensteinHu(box))


def transfer_CLASS(chain, box, cpar, ai):
    # not currently used
    sigma8_true = np.copy(cpar.sigma8)
    cpar.sigma8 = 0
    cpar.A_s = 2.3e-9  # will be modified to correspond to correct sigma
    cosmo = borg.cosmo.ClassCosmo(
        cpar, k_per_decade=10, k_max=50, extra={'YHe': '0.24'})
    cosmo.computeSigma8()  # will compute sigma for the provided A_s
    cos = cosmo.getCosmology()
    # Update A_s
    cpar.A_s = (sigma8_true/cos['sigma_8'])**2*cpar.A_s
    # Add primordial fluctuations
    chain.addModel(borg.forward.model_lib.M_PRIMORDIAL_AS(box))
    # Add CLASS transfer function
    transfer_class = borg.forward.model_lib.M_TRANSFER_CLASS(
        box, opts={"a_transfer": ai, "use_class_sign": False})
    transfer_class.setModelParams(
        {"extra_class_arguments": {"YHe": "0.24", "z_max_pk": "200"}})
    chain.addModel(transfer_class)
