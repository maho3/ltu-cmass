
Getting Started
===============

Welcome to the ltu-cmass installation instructions.

Primary installation:
1. [Clone the repository](#clone-the-repository)
2. [Install dependencies](#install-dependencies)
3. [Install `cmass`](#install-cmass)
4. [Configure the working directory](#configure-the-working-directory)
5. [Running the pipeline](#running-the-pipeline)

Additional functionality:
* [Configuring the pipeline](#configuring-the-pipeline)
* [Working with Quijote ICs](#generating-quijote-ics-and-refitting-halo-bias-models)

Some general design principles:

- The repository is built as a collection of scripts designed to transform cosmological simulations into mocks of galaxy surveys. Each component script is designed to be run through command line on a computing cluster (e.g. `python -m cmass.nbody.pmwd`).
- I/O is done mostly through reading and saving `.npy` files on disk. The scripts expect a certain structure of data storage, located within the working directory specified in [`global.cfg`](global.cfg). See an example directory tree in [Data Structure](#configure-the-working-directory).
- We employ dynamic loading strategy, which means that the dependency modules are only loaded when needed. This is done to allow usage of the scripts when not all prerequisites can be installed simultaneously. However, this means all functions and classes must be loaded through relative imports (e.g. `from .tools import do_something` instead of `import tools; tools.do_something()`).


## Clone the Repository 
Clone the repository from github:
```bash
git clone git@github.com:maho3/ltu-cmass.git
cd ltu-cmass
```

## Install dependencies
ltu-cmass has quite a few dependencies, several of which are rather difficult to install. A partial list is given in [`requirements.txt`](requirements.txt) with the addition of the following unlisted dependencies: 
- nbodykit
- aquila-borg

Here we provide a step-by-step guide to installing the dependencies on a Linux system. For this configuration, everything can be installed into the same environment, though you may want to split up various components into different environments if you have trouble installing everything at once.


### Activating a virtual environment
First, we recommend installing things in a fresh Python environment, such as anaconda. We've tested the following configuration with Python 3.10.
```bash
conda create -n cmass python=3.10
conda activate cmass
```

### Installing nbodykit
We use the [`nbodykit`](https://github.com/bccp/nbodykit) package to apply survey masks and calculate some summary statistics. If you don't want to use these features, you can skip this section.

Installing `nbodykit` is quite tricky. It requires that you have a working C compiler and MPI backend which are compatible with `cython` and `mpi4py`. Below are example instructions for installing `nbodykit` on Infinity and Rusty. 

#### Infinity@IAP
On the Infinity cluster at IAP, you can install nbodykit and its dependencies as:
```bash
# load the C-compiler and MPI modules
module load gcc/13.2.0 openmpi/4.1.2-intel

# clone nbodykit
git clone https://github.com/bccp/nbodykit.git
cd nbodykit

# install nbodykit dependencies
pip install --no-cache-dir  numpy==1.24.4 cython==0.29.33 mpi4py
pip install -r requirements.txt
pip install -r requirements-extras.txt

# install nbodykit itself
pip install -e .

# return to the parent directory
cd ..
```

#### Rusty@Flatiron
On Flatiron/Rusty, the correct versions of nbodykit and mpi4py are already installed. You can load them as follows:
```bash
module load python
module load openmpi
module load python-mpi
module load gcc/13.2.0
python -m venv --system-site-packages $VENVDIR/venv-name  # make your virtual env, and it has nbodykit and mpi4
```

#### Other machines
Currently, there is not a known convenient installation of `nbodykit` on Mac machines. If you manage to install `nbodykit` on a different machine, please let us know so we can add instructions here.


### Installing BORG
We use BORG solely to run the BORG-LPT and BORG-PM gravity solvers. If you don't want to use these features, you can skip this section.

Install the public version of borg with:
```bash
pip install --no-cache-dir aquila-borg
```
The build process for this package may take a while (~20 minutes). Note, this public version of BORG lacks several features, such as BORG-PM simulators. For access to these, consider joining the [Aquila consortium](https://www.aquila-consortium.org/) :).


## Install `cmass`
The remaining dependencies can be installed easily with:
```bash
cd ltu-cmass
pip install -r requirements.txt
pip install -e .
```
Now, the `cmass` package should be accessible in your Python environment. You can test this by running `python -c "import cmass"`.


## Configure the Working Directory
ltu-cmass expects a certain working directory structure to know how to move data around. First, pick a directory on your machine where you want to store the data. On computing clusters, this is usually in the scratch space. Then, change the global configuration in [`ltu-cmass/cmass/conf/global.yaml`](./cmass/conf/global.yaml) to point to this directory, as follows:
```yaml
meta:
    wdir: "/path/to/working/directory"
```
All data will be stored in this directory, and the pipeline will expect to find the data in this directory.

To run the pipeline in the following step, you will need some calibration files and observational masks. These can be downloaded from the `/learningtheuniverse/ltu-cmass-starter` directory on OSN ([See access instructions here](./DATA.md)). They should be stored in the working directory as following:
```yaml
+-- /path/to/working/directory
|   +-- calib_1gpch_z0.5   # name of calibration suite
|   |   +-- pmwd           # name of the gravity solver
|   |   |   +-- L1000-N128 # comoving size and resolution of the simulation
|   |   |   |   +-- 0      # number of the latin-hypercube ID
|   |   |   |   |   +-- halo_bias.npy    # bias model parameters
|   |   |   |   +-- 1
|   |   |   |   +-- ...
|   +-- obs                # CMASS observational masks and randoms
|   |   +-- allsky_bright_star_mask_pix.ply
|   |   +-- badfield_mask_postprocess_pixs8.ply
|   |   +-- ...
```

## Running the pipeline

After all the above steps are completed, you can run the pipeline. The below commands will run a 1 Gpc/h box with 128^3 resolution using the pmwd gravity solver.
```bash
# Run nbody density fields
python -m cmass.nbody.pmwd

# Populate density fields with halos
python -m cmass.bias.rho_to_halo

# Remap the cube into a cuboid to match the survey volume
python -m cmass.survey.remap_cuboid

# Apply the survey mask to the cuboid
python -m cmass.bias.apply_hod

# Apply the NGC survey mask
python -m cmass.survey.ngc_selection

# Measure the power spectrum of the galaxy catalog
python -m cmass.summaries.Pk_nbkit
```

After all the above steps are completed, you should see the data results in your working directory as follows:
```yaml
+-- /path/to/working/directory
|   +-- inf_1gpch          # name of the simulation suite
|   |   +-- pmwd           # name of the gravity solver
|   |   |   +-- L1000-N128 # comoving size and resolution of the simulation
|   |   |   |   +-- 0      # number of the latin-hypercube ID
|   |   |   |   |   +-- config.yaml    # record of the configuration file
|   |   |   |   |   +-- rho.npy        # density contrast field
|   |   |   |   |   +-- fvel.npy       # bulk velocity field
|   |   |   |   |   +-- halo_pos.npy   # halo positions (Mpc/h) - comoving
|   |   |   |   |   +-- halo_vel.npy   # halo velocities (km/s) - comoving
|   |   |   |   |   +-- halo_mass.npy  # halo masses (Msun/h)
|   |   |   |   |   +-- halo_cuboid_pos.npy  # halo positions in cuboid (Mpc/h) - comoving
|   |   |   |   |   +-- halo_cuboid_vel.npy  # halo velocities in cuboid (km/s) - comoving 
|   |   |   |   |   +-- hod            # galaxy positions/velocities, for HOD seed 0
|   |   |   |   |   |   +-- hod0_pos.npy - comoving
|   |   |   |   |   |   +-- hod0_vel.npy - comoving 
|   |   |   |   |   +-- obs           # ra (deg), dec (deg), redshift of galaxies after survey mask
|   |   |   |   |   |   +-- rdz0.npy
|   |   |   |   |   +-- Pk            # survey-space power spectrum
|   |   |   |   |   |   +-- Pk0.npz
```


## Configuring the pipeline
The default configurations of each stage of `ltu-cmass` are stored in [`cmass/conf`](./cmass/conf). We use [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) as a configuration framework. Hydra allows us to define a configuration schema and then override it with command-line arguments. This makes it easy to run the same code with different configurations.

For example, if I want to run `cmass/nbody/pmwd.py` twice using different latin hypercube (LH) cosmologies, I would run:
```bash
python -m cmass.nbody.pmwd nbody.lhid=3
python -m cmass.nbody.pmwd nbody.lhid=4
```
These will have the same configurations, except for the cosmology.

We can then define different suites of simulations based on large-scale configurations. For example, to run a simulation at the standard 2 Gpc/h configuration, we would run:
```bash
python -m cmass.nbody.pmwd nbody=2gpch
```
You can see other default configurations in [`cmass/conf`](./cmass/conf).

## Generating Quijote ICs and refitting halo bias models

You can also use the scripts in [`quijote_wn/NgenicWhiteNoise`](./quijote_wn/NgenicWhiteNoise) to generate initial white noise fields for the Quijote simulations. Then, these can be used to seed the ltu-cmass gravity solvers, and further to calibrate the halo biasing models.

To generate the Quijote ICs, you must first make the executable.
```bash
cd ltu-cmass/quijote_wn/NgenicWhiteNoise
make
```
Then, edit and run `gen_quijote_ic.sh` to generate the ICs. In the below command, this script will generate e.g. $128^3$ white noise fields for the first latin-hypercube cosmology and place them in my `quijote/wn/N128` directory.
```bash
sh gen_quijote_ic.sh 128 0
```

Then, you can run the gravity solvers in [`cmass.nbody`](./cmass/nbody) using the configuration flag `matchIC: 2` to match the Quijote ICs. 

Lastly, you can use these phase-matched density fields to refit the halo biasing parameters by first [downloading the Quijote halos](https://quijote-simulations.readthedocs.io/en/latest/halos.html) and using [cmass.bias.fit_halo_bias](./cmass/bias/fit_halo_bias.py) to fit the bias models. The configuration for this fitting is in [`cmass/conf/fit/quijote_HR.yaml`](./cmass/conf/fit/quijote_HR.yaml). The `path_to_qhalos` parameter specifies the relative path within the working directory to the Quijote halos. For example, my Quijote halos are stored as:
```yaml
+-- /path/to/working/directory
|   +-- quijote
|   |   +-- source
|   |   |   +-- Halos
|   |   |   |   +-- latin_hypercube_HR
|   |   |   |   |   +-- 0
|   |   |   |   |   +-- 1
|   |   |   |   |   +-- ...
```
and my `path_to_qhalos` would be `quijote/source/Halos/latin_hypercube_HR`.