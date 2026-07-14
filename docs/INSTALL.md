
Getting Started
===============

Welcome to the ltu-cmass installation instructions. This provides the minimal installation instructions to run the pipeline end-to-end. However, further functionality and configuration options are described in [Additional Functionality](#additional-functionality).

### Table of Contents
- [Design](#design)
- [Basic installation](#basic-installation)
- [Configure the Working Directory](#configure-the-working-directory)
- [Download observational masks and calibration files](#download-observational-masks-and-calibration-files)
- [Running the pipeline](#running-the-pipeline)
- [Additional Functionality](#additional-functionality)

## Design
- This is a collection of scripts designed to transform cosmological simulations into mocks of galaxy surveys. Each component script is designed to be run through command line on a computing cluster (e.g. `python -m cmass.nbody.pmwd`).
- I/O is done mostly through reading and saving `h5py` files on disk. The scripts expect a certain structure of data storage, located within the working directory specified in [`global.cfg`](../cmass/conf/global.cfg). See an example directory tree in [Running the Pipeline](#running-the-pipeline).
- Dependency modules are only loaded when needed. This is so that all prerequisites need not be installed simultaneously. However, this means everything must be loaded through relative imports (e.g. `from .tools import do_something` instead of `import tools; tools.do_something()`).


## Basic installation
Clone the repository from github:
```bash
git clone git@github.com:maho3/ltu-cmass.git
cd ltu-cmass
```
We recommend installing things in a fresh Python environment, such as anaconda. We've tested the following configuration with Python 3.10.
```bash
conda create -n cmass python=3.10
conda activate cmass
```
Install cmass and its dependencies (in [setup.cfg](../setup.cfg)) automatically using:
```bash
cd ltu-cmass
pip install -e '.[full]'
```
Now, the `cmass` package should be accessible in your Python environment. You can test this by running 
```bash
python -c "import cmass"
```

Some notes:
 - The default nbody simulator in the above installation is `pmwd`. However, due to [a bug in the public `pmwd` repo](https://github.com/eelregit/pmwd/issues/27), we include instead [this forked repository](https://github.com/maho3/pmwd).

 - If you want to use the GPU version of jax in pmwd, make sure you install the right version of jax and jaxlib [corresponding to your CUDA version](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier).

## Configure the Working Directory
`ltu-cmass` expects a certain working directory structure to know how to move data around. First, pick a directory on your machine where you want to store the data. On computing clusters, this is usually in the scratch space. 

Then, change the global configuration in [`ltu-cmass/cmass/conf/global.yaml`](../cmass/conf/global.yaml) to point to this directory, as follows:
```yaml
meta:
    wdir: "/path/to/working/directory"
```
All data will be stored in this directory, and the pipeline will expect to find the data in this directory.

Once you've changed this file, you can remove it from git tracking with:
```bash
git update-index --skip-worktree cmass/conf/global.yaml
```
This ensures that when you then go to push changes to the repository, you don't accidentally push your local configuration.


## Download observational masks and calibration files
To run the pipeline in the following step, you will need some calibration files and observational masks. These are already available for direct copying on the following machines:
- infinity @ IAP: `/automnt/data80/mattho/cmass-ili`
- anvil @ Purdue: `/anvil/projects/x-phy240043/x-mho1`
- narval @ ComputeCanada: `/lustre06/project/6044613/mattho/ltu-cmass`

Also, they can be downloaded from the `/learningtheuniverse/ltu-cmass-starter` directory on OSN ([See access instructions here](./DATA.md)).

They should be stored in the working directory as following:
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
python -m cmass.nbody.pmwd nbody=1gpch

# Populate density fields with halos
python -m cmass.bias.rho_to_halo nbody=1gpch

# Populate the halos with galaxies
python -m cmass.bias.apply_hod nbody=1gpch

# Construct the lightcone and apply the NGC survey mask
python -m cmass.survey.selection nbody=1gpch

# Measure summaries in the simulation box (halos, galaxies)
python -m cmass.diagnostics.summ nbody=1gpch

# Measure summaries in the survey space (galaxies on the lightcone)
python -m cmass.summary.Pk nbody=1gpch
```

After all the above steps are completed, you should see the data results in your working directory as follows:
```yaml
+-- /path/to/working/directory
|   +-- inf_1gpch          # name of the simulation suite
|   |   +-- pmwd           # name of the gravity solver
|   |   |   +-- L1000-N128 # comoving size and resolution of the simulation
|   |   |   |   +-- 0      # number of the latin-hypercube ID
|   |   |   |   |   +-- config.yaml   # record of the configuration file
|   |   |   |   |   +-- nbody.h5  # density and velocity fields
|   |   |   |   |   +-- halos.h5      # halo positions, velocity and masses
|   |   |   |   |   +-- galaxies
|   |   |   |   |   |   +-- hod00000.h5   # galaxy positions/velocities, for HOD seed 0
|   |   |   |   |   +-- ngc_lightcone           
|   |   |   |   |   |   +-- hod00000_aug00000.h5  # ra (deg), dec (deg), redshift of galaxies after survey mask
|   |   |   |   |   +-- diag  # simulation box summaries
|   |   |   |   |   |   +-- rho.h5
|   |   |   |   |   |   +-- halos.h5
|   |   |   |   |   |   +-- galaxies
|   |   |   |   |   |   |   +-- hod00000.h5
|   |   |   |   |   +-- ngc_summary  # survey-space summaries
|   |   |   |   |   |   +-- hod00000_aug00000.h5        # survey-space power spectrum
```


## Configuring the pipeline
The default configurations of each stage of `ltu-cmass` are stored in [`cmass/conf`](../cmass/conf). We use [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) as a configuration framework. Hydra allows us to define a configuration schema and then override it with command-line arguments. This makes it easy to run the same code with different configurations.

For example, if I want to run `cmass.nbody.pmwd` twice using different latin hypercube (LH) cosmologies, I would run:
```bash
python -m cmass.nbody.pmwd nbody.lhid=3
python -m cmass.nbody.pmwd nbody.lhid=4
```
These will have the same configurations, except for the cosmology.

We can then define different suites of simulations based on large-scale configurations. For example, to run a simulation at the standard 2 Gpc/h configuration, we would run:
```bash
python -m cmass.nbody.pmwd nbody=2gpch
```
You can see various default configurations in [`cmass/conf`](../cmass/conf). However, the two most commonly used are `sim` and `multisnapshot` as described in [`cmass/conf/config.yaml`](../cmass/conf/config.yaml). For example:
```bash
python -m cmass.bias.rho_to_halo nbody=1gpch sim=fastpm multisnapshot=True
```
`sim` tells the script which base nbody simulator (pmwd, fastpm, borglpt, borgpm, pinocchio) the script should use data products from. `multisnapshot` is a boolean flag that tells the script whether to use all simulation snapshots available, or only the final snapshot. The defaults are `sim=pmwd` and `multisnapshot=True`.

## Additional Functionality

We include various additional functionality beyond the minimal working example above. These are provided in separate documentation, including:
- [Installing `pmesh` and running pypower](./options/PMESH.md)
- [Installing ili-summarizer](./options/SUMMARIZER.md)
- [Running with Quijote initial conditions and fitting bias models](./options/QUIJOTE.md)
- [Installing and running FASTPM nbody simulators](./options/FASTPM.md)
- [Installing and running BORG nbody simulators](./options/BORG.md)
- [Building and running lightcone extrapolation](./options/LIGHTCONE.md)
- [Building and running the PINOCCHIO simulator](./options/PINOCCHIO.md)
- [Filtering and weighting galaxy positions](./options/FILTERING.md)
- [Installing pypower](./options/PYPOWER.md)