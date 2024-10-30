import numpy as np
import Pk_library as PKL
import MAS_library as MASL
import redshift_space_library as RSL


def get_redshift_space_pos(pos, vel, L, h, z, axis=0):
    pos, vel = map(np.ascontiguousarray, (pos, vel))
    RSL.pos_redshift_space(pos, vel, L, h*100, z, axis)
    return pos


def MA(pos, L, N, MAS='CIC'):
    pos = np.ascontiguousarray(pos)
    delta = np.zeros((N, N, N), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize=L, MAS=MAS)
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1
    return delta


def MAz(pos, vel, L, N, h, z, MAS='CIC', axis=0):
    pos, vel = map(np.ascontiguousarray, (pos, vel))
    RSL.pos_redshift_space(pos, vel, L, h*100, z, axis)
    return MA(pos, L, N, MAS)


def calcPk(delta, L, axis=0, MAS='CIC', threads=16):
    Pk = PKL.Pk(delta, L, axis, MAS, threads, verbose=False)
    k = Pk.k3D
    Pk = Pk.Pk
    return k, Pk
