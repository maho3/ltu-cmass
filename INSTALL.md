
Getting Started
===============

Welcome to the ltu-cmass installation instructions. First, some design principles:

- The repository is built as a collection of scripts designed to transform cosmological simulations into mocks of galaxy surveys. Each component script is designed to be run through command line on a computing cluster (e.g. `python -m cmass.nbody.pmwd`).
- I/O is done mostly through reading and saving `.npy` files on disk. The scripts expect a certain structure of data storage, located within the working directory specified in [`global.cfg`](global.cfg). An example of this data structure can be found in the `cmass-ili` folder of the [LtU OSN storage](https://github.com/maho3/ltu-ili/blob/main/DATA.md).
- We employ dynamic loading strategy, which means that the dependency modules are only loaded when needed. This is done to allow usage of the scripts when not all prerequisites can be installed simultaneously. However, this means all functions and classes must be loaded through relative imports (e.g. `from .tools import do_something` instead of `import tools; tools.do_something()`).
- The configuration options are given at the top of each script file in a `build_config()` function. There, you can find any command line arguments or user-defined settings which can affect the computation.

## Installation
ltu-cmass has quite a few dependencies, several of which are rather difficult to install. A partial list is given in [`requirements.txt`](requirements.txt) with the addition of the following unlisted dependencies: 
- nbodykit
- aquila-borg

Here we provide a step-by-step guide to installing the dependencies on a Linux system. For this configuration, everything can be installed into the same environment, though you may want to split up various components into different environments if you have trouble installing everything at once.

### Getting setup
First, clone the repository:
```bash
git clone git@github.com:maho3/ltu-cmass.git
```
Next, we recommend installing things in a fresh Python environment, such as anaconda. We've tested the following configuration with Python 3.10.
```bash
conda create -n cmass-env python=3.9
conda activate cmass-env
```

### Installing nbodykit
Installing nbodykit is quite tricky. First, it requires that you have a working MPI compiler and a C compiler installed which are compatible with cython and mpi4py. This is usually not so simple on a Mac machine. On the Linux cluster at Infinity@IAP, for example, you load these with:
```bash
module load gcc/13.2.0 openmpi/4.1.2-intel
```
Next, clone the nbodykit repository:
```bash
git clone https://github.com/bccp/nbodykit/tree/master
cd nbodykit
```
Then, install the dependencies. Note, numpy, cython, and mpi4py must be installed first because they are used to build other packages. We use the `--no-cache-dir` flag to force recompiling of cython and mpi4py, which must be built for your specific compilers.
```bash
pip install --no-cache-dir  numpy==1.24.4 cython==0.29.33 mpi4py
pip install -r requirements.txt
pip install -r requirements-extras.txt
```
Finally, install nbodykit itself:
```bash
pip install -e .
cd .. # return to the parent directory
```

### Installing BORG
Install the public version of borg with:
```bash
pip install --no-cache-dir aquila-borg
```
The build process for this package may take a while (~20 minutes). Note, this public version of BORG lacks several features, such as BORG-PM simulators. For access to these, consider joining the [Aquila consortium](https://www.aquila-consortium.org/) :).

### Installing other dependencies and ltu-cmass
The remaining dependencies can be installed with:
```bash
cd ltu-cmass
pip install -r requirements.txt
```
If you have access to jax_lpt (contact Axel Lapel), you can also install that with:
```bash
pip install -r requirements-extras.txt
```
Finally, install the ltu-cmass package itself:
```bash
pip install -e .
```

### Configure the working directory
Lastly, configure the json in `global.cfg` to point to the working directory where you want to store the data and the text file of cosmological parameters you want to index from. We recommend setting the working directory to somewhere in the scratch space of your computing cluster, as the data can take up many GB.

### Download some working data
Several steps of the forward model depend on external data that has been gathered and shared on the [LtU OSN storage](https://github.com/maho3/ltu-ili/blob/main/DATA.md). When getting started, we recommended you to download a copy of a pre-compiled and run working directory from the `cmass-ili` folder of the OSN repository. This will ensure your working directory is setup properly and has the prerequisite data.

## Running the code

After the packages are installed and your working directory is set up, a full forward pass of the model might look like this:
```bash
cd ltu-cmass
conda activate cmass-env

# simulate density field, particle positions and velocities with borglpt
python -m cmass.nbody.borglpt --lhid 0 --order 2

# sample halo positions, velocities, and masses from the density field
python -m cmass.biasing.rho_to_halo --lhid 0 --simtype borg2lpt

# reshape the halo catalog to a cuboid matching our survey volume
python -m cmass.survey.remap_cuboid --lhid 0 --simtype borg2lpt

# apply HOD to sample galaxies
python -m cmass.bias.apply_hod --lhid 0 --seed 1 --simtype borg2lpt

# apply survey mask to the galaxy catalog
python -m cmass.survey.ngc_selection --lhid 0 --seed 1 --simtype borg2lpt

# calculate power spectrum of the galaxy catalog
python -m cmass.summaries.Pk_nbkit --lhid 0 --seed 1 --simtype borg2lpt
```

This should save all intermediates and outputs to the working directory specified in `global.cfg`, which is `ltu-cmass/data` by default.
