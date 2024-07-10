
Compiling lightcone extrapolation
=================================


We include a C++ implementation of multi-snapshot lightcone extrapolation in `cmass.lightcone`, developed by Leander Thiele and adapted from [this repository](https://github.com/leanderthiele/nuvoid_production). **This is necessary for running `cmass.survey.ngc_lightcone`**, but not necessary for its single-snapshot counterpart, `cmass.survey.ngc_selection`.

To compile the code, you need to have `gsl` and `gcc` installed on your machine. On Infinity@IAP, you can load these with:
```bash
module load gsl/2.7.1
module load gcc/13.3.0
```
Then, simply compile the code with:
```bash
cd ltu-cmass/cmass/lightcone
make
```
This will generate various `*.o` and `*.a` files in the `cmass/lightcone` directory, which can be accessed by Python. You can then test that it works by importing `cmass.lightcone.lc` in Python.
```bash
python -c "from cmass.lightcone import lc"
```
This will allow you to run `cmass.survey.ngc_lightcone`. Further installation and usage instructions can be found in [cmass/lightcone/README.md](../../cmass/lightcone/README.md).