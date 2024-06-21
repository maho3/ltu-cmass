import numpy as np
import aquila_borg as borg


def build_cosmology(omega_m, omega_b, h, n_s, sigma8):
    cpar = borg.cosmo.CosmologicalParameters()
    cpar.default()
    cpar.omega_m, cpar.omega_b, cpar.h, cpar.n_s, cpar.sigma8 = \
        (omega_m, omega_b, h, n_s, sigma8)
    cpar.omega_q = 1.0 - cpar.omega_m
    return cpar


def compute_As(cpar):
    # requires BORG-CLASS
    if not hasattr(borg.cosmo, 'ClassCosmo'):
        raise ImportError(
            "BORG-CLASS is required to compute As, but is not installed.")
    sigma8_true = np.copy(cpar.sigma8)
    cpar.sigma8 = 0
    cpar.A_s = 2.3e-9
    k_max, k_per_decade = 10, 100
    extra_class = {}
    extra_class['YHe'] = '0.24'
    cosmo = borg.cosmo.ClassCosmo(cpar, k_per_decade, k_max, extra=extra_class)
    cosmo.computeSigma8()
    cos = cosmo.getCosmology()
    cpar.A_s = (sigma8_true/cos['sigma_8'])**2*cpar.A_s
    cpar.sigma8 = sigma8_true
    return cosmo


def transfer_EH(chain, box, a_final=1.0):
    chain @= borg.forward.model_lib.M_PRIMORDIAL(
        box, opts=dict(a_final=a_final))
    chain @= borg.forward.model_lib.M_TRANSFER_EHU(
        box, opts=dict(reverse_sign=True))
    return chain


def transfer_CLASS(chain, box, cpar, a_final=1.0):
    if not hasattr(borg.forward.model_lib, 'M_TRANSFER_CLASS'):
        raise ImportError(
            "BORG-CLASS is required to use the CLASS transfer function.")

    # Compute As
    cpar = compute_As(cpar)

    # Add primordial fluctuations
    chain @= borg.forward.model_lib.M_PRIMORDIAL_AS(box)

    # Add CLASS transfer function
    extra_class = {"YHe": "0.24", "z_max_pk": "200"}
    transfer_class = borg.forward.model_lib.M_TRANSFER_CLASS(
        box, opts=dict(a_transfer=a_final))
    transfer_class.setModelParams({"extra_class_arguments": extra_class})
    chain @= transfer_class

    return chain
