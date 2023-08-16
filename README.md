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

A full forward pass of the model might look like this:
```bash
cd ltu-cmass

# simulate density field, particle positions and velocities with jax_lpt
python -m cmass.nbody.jax2lpt --lhid 0

# sample halo positions, velocities, and masses from the density field
python -m cmass.biasing.rho_to_halo --lhid 0 --simtype jax2lpt

# reshape the halo catalog to a cuboid matching our survey volume
python -m cmass.survey.remap_as_cuboid --lhid 0 --simtype jax2lpt

# apply HOD to sample galaxies
python -m cmass.biasing.apply_hod --lhid 0 --seed 42 --simtype jax2lpt

# apply survey mask to the galaxy catalog
python -m cmass.survey.apply_survey_cut --lhid 0 --seed 42 --simtype jax2lpt

# calculate power spectrum of the galaxy catalog
python -m cmass.summaries.calc_Pk_nbkit --lhid 0 --seed 42 --simtype jax2lpt
```

This should save all intermediates and outputs to `./data/`.

Links to major repositories:
- [`borg`](https://bitbucket.org/aquila-consortium/borg_public/src/e09486dfd098ffc4ccfbb167621a900034b4382e/?at=release%2F2.1)
- [`jax_lpt`](https://bitbucket.org/aquila-consortium/jax_lpt/src/main/)
- [`pmwd`](https://github.com/eelregit/pmwd/tree/master)
- [`cuboid_remap_jax`](https://github.com/maho3/cuboid_remap_jax)
- [`nbodykit`](https://github.com/bccp/nbodykit)
- [`pymangle`](https://github.com/esheldon/pymangle/tree/master)
