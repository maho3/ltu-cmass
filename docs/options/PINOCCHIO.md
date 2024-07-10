Installing Pinocchio
====================
This pipeline includes the option to use the [Pinocchio](https://github.com/pigimonaco/Pinocchio) 2LPT gravity solver and halo emulator. This is available in `cmass.nbody.pinocchio`. 

Pinocchio requires you to have an MPI library, `fftw3`, and `gsl` installed on your machine. You then download and compile Pinocchio for your system. The original installation instructions are [available here](https://github.com/pigimonaco/Pinocchio/blob/master/INSTALLATION), but we provide a working example for the Infinity@IAP cluster below:

## Compiling Pinocchio on Infinity@IAP
To build pinnochio on infinity, first load `openmpi`
```bash
module load openmpi/5.0.3-gnu
```
After cloning pinnochio from https://github.com/pigimonaco/Pinocchio, edit the Makefile to be compatible with infinity:
```bash
ifeq ($(SYSTYPE),"infinity")
CC          =  mpicc
CDEBUG      = -ggdb3 -Wall
COPTIMIZED  = -O3 -Wno-unused-result
FFTW_LIBR   = -L/softs/fftw3/3.3.10-gnu-mpi/lib -lfftw3_mpi -lfftw3
FFTW_INCL   = -I/softs/fftw3/3.3.10-gnu-mpi/include
MPI_LIBR    = -L/softs/openmpi/5.0.3-gnu/include
MPI_INCL    =
GSL_LIBR    = -lgsl -lgslcblas -lm
GSL_INCL    = -I/usr/include
endif
```
Also change the `SYSTYPE` argument to be `"infinity"` and ensure the "-DWHITENOISE" option is enabled. Building with `make clean; make` should then work. 

Lastly, in [your nbody configuration file](./cmass/conf/nbody/pinocchio.yaml) you need to specify the absolute path to your Pinocchio executable. This is the `pinnochio.x` file in the `src` directory of Pinocchio, generated during the `make`. For example, mine is:
```yaml
pinocchio_exec: /home/mattho/git/Pinocchio/src/pinocchio.x
```

Finally, before running the executable `pinnochio.x`, you must also run
```bash
export LD_LIBRARY_PATH=/softs/fftw3/3.3.10-gnu-mpi/lib:$LD_LIBRARY_PATH
```
to enable pinnochio to access the fftw3_mpi library. Then you should be able to run the default configuration:
```bash
python -m cmass.nbody.pinocchio nbody=pinocchio
```

## Installing camb, CLASS, or syren
Pinocchio then requires you to generate a linear power spectrum on your own. We provide integration with [camb](https://github.com/cmbant/CAMB), [CLASS](https://github.com/lesgourg/class_public), or [syren](https://github.com/DeaglanBartlett/symbolic_pofk).
```bash
# Install CAMB
pip install camb
# Install syren
pip install git+https://github.com/DeaglanBartlett/symbolic_pofk.git
```
To install CLASS's python wrapper, follow the instructions in the [CLASS repository](https://github.com/lesgourg/class_public/wiki/Python-wrapper).