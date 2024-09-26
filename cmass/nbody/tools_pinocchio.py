import logging
import numpy as np
import h5py
import os
import sys
import yaml
from os.path import join
from ..bias.rho_to_halo import save_snapshot
from .tools import (
    save_nbody, rho_and_vfield)

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s-%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[FlushHandler()]
)

def save_cfg_data(outdir, cfg):

    filename = join(outdir, 'snapshot_data.yaml')

    if hasattr(cfg.nbody, 'supersampling'):
        supersampling = cfg.nbody.supersampling
    else:
        supersampling = None
    data = {
            'outdir':outdir,
            'asave':list(cfg.nbody.asave),
            'L':cfg.nbody.L,
            'N':cfg.nbody.N,
            'lhid':cfg.nbody.lhid,
            'omega_m':cfg.nbody.cosmo[0], 
            'h':cfg.nbody.cosmo[2],
            'supersampling':supersampling,
            'save_particles':cfg.nbody.save_particles
            }

    with open(filename, 'w') as f:
        yaml.dump(data, f)

    return


def process_snapshot(outdir, z, L, N, lhid, omega_m, h, supersampling=None):

    # Load the data
    filename = join(
        outdir,
        f'pinocchio.{z:.4f}.pinocchio-L{L}-'
        f'N{N}-{lhid}.snapshot.out')
    data = {}

    with open(filename, 'rb') as f:

        # Function to read a block
        def read_block(expected_name, dtype, count):
            _ = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            block_name = np.fromfile(f, dtype='S4', count=1)[
                0].decode().strip()
            _ = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            if block_name != expected_name:
                raise ValueError(
                    f"Expected block name '{expected_name}', "
                    f"but got '{block_name}'")
            data_block = np.fromfile(f, dtype=dtype, count=count)
            _ = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            return data_block


        # Function to read a large block
        def read_large_block(expected_name, dummy_dtype, data_dtype, num_particles, chunk_size = 900000):

            # Save to file to prevent storing too much in memory
            out_filename = join(
                outdir, f'pinocchio.{z:.4f}.pinocchio-L{L}-'
                f'N{N}-{lhid}.particle_{expected_name}.h5')

            def write_initial_chunk(data_chunk):
                if expected_name == 'ID':
                    shape=(num_particles,)
                    maxshape=(num_particles,)
                elif expected_name == 'VEL':
                    shape=(num_particles//3,)
                    maxshape=(num_particles//3,)
                else:
                    shape=(num_particles//3,)+data_chunk.shape[1:]
                    maxshape=(num_particles//3,)+data_chunk.shape[1:]

                with h5py.File(out_filename, 'w') as fout:
                    if expected_name == 'VEL':
                        for i, suffix in enumerate(['X', 'Y', 'Z']):
                            dset = fout.create_dataset(
                                expected_name + suffix,
                                shape=shape,
                                maxshape=maxshape,  # Preallocate space for the number of particles
                                chunks=True,
                                dtype=data_chunk.dtype
                            )
                            dset[:data_chunk.shape[0]] = data_chunk[:, i]
                    else:
                        dset = fout.create_dataset(
                            expected_name,
                            shape=shape,
                            maxshape=maxshape,  # Preallocate space for the number of particles
                            chunks=True,
                            dtype=data_chunk.dtype
                        )
                        dset[:data_chunk.shape[0]] = data_chunk


            def append_chunk(new_chunk):
                with h5py.File(out_filename, 'a') as fout:
                    if expected_name == 'VEL':
                        for i, suffix in enumerate(['X', 'Y', 'Z']):
                            dset = fout[expected_name+suffix]
                            dset[total_read//3:total_read//3+new_chunk.shape[0]] = new_chunk[:,i]
                    elif expected_name == 'ID':
                        dset = fout[expected_name]
                        dset[total_read:total_read+new_chunk.shape[0]] = new_chunk
                    else:
                        dset = fout[expected_name]
                        dset[total_read//3:total_read//3+new_chunk.shape[0]] = new_chunk

            _ = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            block_name = np.fromfile(f, dtype='S4', count=1)[
                0].decode().strip()
            _ = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            if block_name != expected_name:
                raise ValueError(
                    f"Expected block name '{expected_name}', "
                    f"but got '{block_name}'")

            _ = np.fromfile(f, dtype=dummy_dtype, count=1)[0]

            assert chunk_size % 3 == 0, f"{chunk_size} is not divisible by 3"
            total_read = 0

            while total_read < num_particles:
                count = min(chunk_size, num_particles - total_read)
                data = np.fromfile(f, dtype=data_dtype, count=count)

                if expected_name in ['POS', 'VEL']:
                    data = data.reshape(-1,3)
                    data[:, [0, 1, 2]] = data[:, [2, 1, 0]]

                # Periodic boundary conditions
                if expected_name == 'POS':
                    data = np.mod(data, L)

                if total_read == 0:
                    write_initial_chunk(data)
                else:
                    append_chunk(data)

                total_read += count

            _ = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used

            return out_filename

        # Read the HEADER block
        header_dtype = np.dtype([
            ('dummy', np.int64),
            ('NPart', np.uint32, 6),
            ('Mass', np.float64, 6),
            ('Time', np.float64),
            ('RedShift', np.float64),
            ('flag_sfr', np.int32),
            ('flag_feedback', np.int32),
            ('NPartTotal', np.uint32, 6),
            ('flag_cooling', np.int32),
            ('num_files', np.int32),
            ('BoxSize', np.float64),
            ('Omega0', np.float64),
            ('OmegaLambda', np.float64),
            ('HubbleParam', np.float64),
            ('flag_stellarage', np.int32),
            ('flag_metals', np.int32),
            ('npartTotalHighWord', np.uint32, 6),
            ('flag_entropy_instead_u', np.int32),
            ('flag_metalcooling', np.int32),
            ('flag_stellarevolution', np.int32),
            ('fill', np.int8, 52)
        ])
        header_block = read_block('HEAD', header_dtype, 1)[0]
        data['header'] = {name: header_block[name]
                          for name in header_dtype.names}

        # Number of particles
        num_particles = header_block['NPart'][1]

        # Read INFO block
        info_dtype = np.dtype(
            [('name', 'S4'), ('type', 'S8'), ('ndim', np.int32),
             ('active', np.int32, 6)])
        read_block('INFO', info_dtype, 4)

        # Read empty FMAX block
        fmax_dtype = np.dtype([('dummy', np.int64), ('fmax', np.float32)])
        read_block('FMAX', fmax_dtype, 2)

        # Read empty RMAX block
        rmax_dtype = np.dtype([('dummy', np.int64), ('rmax', np.int64)])
        read_block('RMAX', rmax_dtype, 2)

        # Read ID block in chunks (otherwise can be too much memory)
        logging.info("Reading particle IDs...")
        read_large_block('ID', np.int64, np.uint32, num_particles,)

        # Read POS block and save to file to reduce memory allocated
        logging.info("Reading particle positions...")
        pos_fname = read_large_block('POS', np.int64, np.float32, num_particles*3)

        # Read VEL block and save to file to reduce memory allocated
        logging.info("Reading particle velocities...")
        vel_fname  = read_large_block('VEL', np.int64, np.float32, num_particles*3)

    # Acces pos and vel with memory mapping
    posfile = h5py.File(pos_fname, 'r')
    velfile = h5py.File(vel_fname, 'r')
    pos = posfile['POS']
    vel = [velfile['VELX'], velfile['VELY'], velfile['VELZ']]

    # Calculate velocity field
    if supersampling is None:
        Ngrid = N
    else:
        Ngrid = N // supersampling
    logging.info(f'Running field construction on grid {Ngrid}...')
    rho, fvel = rho_and_vfield(
        pos, vel, L, Ngrid, 'CIC',
        omega_m=omega_m, h=h,
        chunk_size = 512**3)

    posfile.close()
    velfile.close()

    return rho, fvel, pos_fname, vel_fname


def save_pinocchio_nbody(outdir, rho, fvel, pos_fname, vel_fname, z, save_particles=False):

    if save_particles:
        posfile = h5py.File(pos_fname, 'r')
        velfile = h5py.File(vel_fname, 'r')
        pos = posfile['POS']
        vel = [velfile['VELX'], velfile['VELY'], velfile['VELZ']]
    else:
        pos, vel = None, None

    # Convert from comoving -> physical velocities
    fvel *= (1 + z)

    # Save nbody-type outputs
    af = 1 / (1 + z)
    save_nbody(outdir, af, rho, fvel, pos, vel, mode='a')

    if save_particles:
        posfile.close()
        velfile.close()

    # Delete temporary position and velocity files
    for fname in [pos_fname, vel_fname]:
        os.remove(fname)

    return



def process_halos(outdir, z, L, N, lhid):

    # Load halo data
    #    0) group ID
    #    1) group mass (Msun/h)
    # 2- 4) initial position (Mpc/h)
    # 5- 7) final position (Mpc/h)
    # 8-10) velocity (km/s)
    #   11) number of particles
    halo_filename = join(
        outdir, f'pinocchio.{z:.4f}.pinocchio-L{L}-'
                f'N{N}-{lhid}.catalog.out')
    hmass = np.log10(np.loadtxt(halo_filename, unpack=False, usecols=(1,)))
    hpos = np.loadtxt(halo_filename, unpack=False, usecols=(7, 6, 5))
    hvel = np.loadtxt(halo_filename, unpack=False, usecols=(10, 9, 8))

    # Save bias-type outputs
    af = 1 / (1 + z)
    outpath = join(outdir, 'halos.h5')
    logging.info('Saving cube to ' + outpath)
    save_snapshot(outdir, af, hpos, hvel, hmass)

    return


def main():

    from mpi4py import MPI

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get run information
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    # Split redshifts across ranks
    asave = np.array_split(data['asave'], size)
    nperrank = len(asave[0])
    asave = asave[rank]

    outdir = data['outdir']
    L = data['L']
    N = data['N']
    lhid = data['lhid']
    omega_m = data['omega_m']
    h = data['h']
    supersampling = data['supersampling']
    save_particles = data['save_particles']

    # Delete output files if they already exists
    for fname in ['halos.h5', 'nbody.h5']:
        outpath = join(outdir, fname)
        if os.path.isfile(outpath):
            os.remove(outpath)

    for i in range(nperrank):

        # Process particles
        if i < len(asave):
            a = asave[i]
            z = 1 / a - 1
            rho, fvel, pos_fname, vel_fname =  process_snapshot(
                outdir, z, L, N, lhid, omega_m, h, 
                supersampling=supersampling)
        comm.Barrier()

        # Print results rank-by-rank to avoid simultaneous
        # writing to the same file
        # Also process halos
        for r in range(size):
            if (r == rank) and (i < len(asave)):
                logging.info(f'Outputting results for a = {a:.6f}')
                save_pinocchio_nbody(outdir, rho, fvel, pos_fname, vel_fname, z, 
                    save_particles=save_particles)
                process_halos(outdir, z, L, N, lhid)
            comm.Barrier()

        logging.info(f'Completed a = {a}')

    comm.Barrier()

    return


if __name__ == "__main__":
    main()
