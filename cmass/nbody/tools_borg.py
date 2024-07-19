import numpy as np
import aquila_borg as borg
from mpi4py import MPI
from .tools import bin_cube
from ..utils import timing_decorator


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
    chain @= borg.forward.model_lib.M_PRIMORDIAL_AS(
        box, opts=dict(a_final=a_final))

    # Add CLASS transfer function
    extra_class = {"YHe": "0.24", "z_max_pk": "100"}
    transfer_class = borg.forward.model_lib.M_TRANSFER_CLASS(
        box, opts=dict(a_transfer=a_final))
    transfer_class.setModelParams({"extra_class_arguments": extra_class})
    chain @= transfer_class

    return chain


def apply_transfer_fn(wn, L, N, cosmo, af=1./(1+99), transfer='CLASS'):

    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = 3*(L,)
    box.N = 3*(N,)

    chain = borg.forward.ChainForwardModel(box)
    if transfer == 'CLASS':
        chain = transfer_CLASS(chain, box, cosmo, a_final=af)
    elif transfer == 'EH':
        chain = transfer_EH(chain, box, a_final=af)
    else:
        raise NotImplementedError(
            f'Transfer function "{transfer}" not implemented.')

    chain.setAdjointRequired(False)
    chain.setCosmoParams(cosmo)

    # forward model
    chain.forwardModel_v2(wn)

    _, out_localN0, out_N1, out_N2 = chain.getOutputMPISlice()
    rho = np.empty((out_localN0, out_N1, out_N2))
    chain.getDensityFinal(rho)

    del chain, box
    return rho


@timing_decorator
def run_transfer(wn, cpar, cfg):
    # Load MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Apply transfer function
    rho_transfer = apply_transfer_fn(
        wn, cfg.nbody.L, cfg.nbody.N*cfg.nbody.supersampling, cpar,
        af=1./(1+99),  # z=99, for CHARM inputs
        transfer=cfg.nbody.transfer)

    # Bin transfer field to match density field resolution
    rho_transfer = bin_cube(rho_transfer, cfg.nbody.supersampling)
    rho_transfer = rho_transfer.astype(np.float32)

    # Gather and save
    if rank == 0:
        gathered = np.empty((cfg.nbody.N,)*3, dtype=np.float32)
    else:
        gathered = None
    comm.Gather(rho_transfer, gathered, root=0)

    return gathered

# MPI functions


def getMPISlice(cfg):
    nbody = cfg.nbody
    N = nbody.N*nbody.supersampling

    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = 3*(nbody.L,)
    box.N = 3*(N,)

    chain = borg.forward.ChainForwardModel(box)
    startN0, localN0, localN1, localN2 = chain.getMPISlice()
    del chain, box

    return startN0, localN0, localN1, localN2


def send_MPI(rho, cfg):
    """Previously used to send ICs across MPI ranks."""
    # Load MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Send offsets and chunk sizes
    startN0, localN0, localN1, localN2 = getMPISlice(cfg)
    offsets = comm.gather(startN0, root=0)
    sizes = comm.gather(localN0, root=0)
    offsets, sizes = np.array(offsets), np.array(sizes)

    # Setup receive buffer
    recvbuf = np.empty((localN0, localN1, localN2), dtype=np.float64).flatten()

    # Prepare data
    if rank == 0:
        rho = np.ascontiguousarray(rho).flatten()
        sizes *= localN1*localN2
        offsets *= localN1*localN2

    # Send
    comm.Scatterv(
        [rho, sizes, offsets, MPI.DOUBLE],
        recvbuf,
        root=0)
    return recvbuf.reshape(localN0, localN1, localN2)


def gather_MPI(pos, vel):
    # Load MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Number of columns
    cols = 3  # x, y, z

    # Gather particle positions and velocities
    assert len(pos) == len(vel), "Number of particles do not match"
    chunk_sizes = comm.gather(len(pos), root=0)

    # Setup recieve buffers
    if rank == 0:
        displacements = [0] + list(np.cumsum(chunk_sizes[:-1]))
        pos_gathered = np.empty((sum(chunk_sizes)*cols), dtype=np.float32)
        vel_gathered = np.empty((sum(chunk_sizes)*cols), dtype=np.float32)

        chunk_sizes = np.array(chunk_sizes)*cols
        displacements = np.array(displacements)*cols
    else:
        displacements, pos_gathered, vel_gathered = None, None, None

    # Gather
    try:
        comm.Gatherv(
            pos.flatten(),
            [pos_gathered, chunk_sizes, displacements, MPI.FLOAT],
            root=0)
        comm.Gatherv(
            vel.flatten(),
            [vel_gathered, chunk_sizes, displacements, MPI.FLOAT],
            root=0)
    except MPI.Exception as e:
        print("MPI Exception:", e)
        # For mpi4py, you might need to check the specific error code
        if hasattr(e, 'error_code'):
            print("MPI Error Code:", e.error_code)
        raise e

    # Return gathered data on rank 0
    if rank != 0:
        return pos, vel
    return pos_gathered.reshape(-1, cols), vel_gathered.reshape(-1, cols)
