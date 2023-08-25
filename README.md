# ltu-cmass
A repository for storing code for the LtU Express Go Big pipeline. The scripts in this repository are designed to simulate and analyze mocks of the CMASS NGC galaxy sample from the BOSS survey.

## Organization
<!-- The repository is organized into three main directories: `simulation`, `summaries`, and `tools`. The `simulation` directory contains scripts for generating mock catalogs of the CMASS NGC sample. The `summaries` directory contains scripts for calculating informative summaries from the survey mocks. The `tools` directory contains software used to support various parts of the forward modeling pipeline. -->

Below, we list the functionality of each script in the repository as well as its major dependencies:

### cmass/nbody
  - `borg2lpt.py` - Simulate a cubic volume using Borg2LPT. Requires: `borg`.
  - `jax2lpt.py` - Simulate a cubic volume using Jax2LPT. Requires: `borg` and `jax_lpt`.
  - `pmwd.py` - Simulate a cubic volume using PM-WD. Requires: [`pmwd`](https://github.com/eelregit/pmwd/tree/master).

### cmass/survey
- `remap_as_cuboid.py` - Remap a periodic volume to a cuboid. Requires: [`cuboid_remap_jax`](https://github.com/maho3/cuboid_remap_jax).
- `apply_survey_cut.py` - Applies BOSS survey mask to a lightcone-shaped volume of galaxies. Requires: `nbodykit`, `pymangle`, and `astropy`.

### cmass/biasing
- `fit_bias.py` - Fit a halo biasing model to map density fields to halo counts. Requires: `astropy` and `scipy`.
- `rho_to_halo.py` - Sample halos from the density field using a pre-fit bias model. Requires: `scipy`.
- `apply_hod.py` - Sample an HOD realization from the halo catalog using the Zheng+(2007) model. Requires: `nbodykit`.

### cmass/summaries
- `calc_Pk_nbkit.py` - Measure the power spectrum of a galaxy catalog. Requires: `nbodykit`.

### Notebooks
- `preprocess.ipynb` -  Executes various preprocessing tasks prepare for mass simulation. Designed to be run once at the start of code development.
- `validation.ipynb` - Validates outputs at intermediate stages in the forward pipeline.
- `inference.ipynb` - Working notebook to perform implicit inference on power spectra from the CMASS forward model.
