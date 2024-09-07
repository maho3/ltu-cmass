import numpy as np

def load_white_noise(path_to_ic, N, quijote=False):
    """Loading in Fourier space."""
    print(f"Loading ICs from {path_to_ic}...")
    num_modes_last_d = N // 2 + 1
    with open(path_to_ic, 'rb') as f:
        if quijote:
            _ = np.fromfile(f, np.uint32, 1)[0]
        modes = np.fromfile(f, np.complex128, -1)
        modes = modes.reshape((N, N, num_modes_last_d))
    return modes

old_ic = load_white_noise('/home/x-dbartlett/project_dir/x-mho1/cmass-ili/quijote/wn/N128/wn_0.dat', 128, True)
new_ic = load_white_noise('/anvil/scratch/x-dbartlett/quijote/wn/N128/wn_0.dat', 128, True)
print('Are they the same?')
print(np.all(old_ic == new_ic))
