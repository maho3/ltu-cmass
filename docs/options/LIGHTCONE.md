
Compiling lightcone extrapolation
=================================


We include a C++ implementation of multi-snapshot lightcone extrapolation in `cmass.lightcone`, developed by Leander Thiele and adapted from [this repository](https://github.com/leanderthiele/nuvoid_production). **This is necessary for running `cmass.survey.lightcone`**, but not necessary for its single-snapshot counterpart, `cmass.survey.selection`.

To compile the code, you need to have `gsl` and `gcc` installed on your machine. On Infinity@IAP or Anvil@Purdue, you can load these with:
```bash
# infinity
module load gsl/2.7.1
module load gcc/13.3.0
# anvil
module load gcc/11.2.0
module load gsl/2.4
```
Then, make sure your python environment has a version of pybind11 available
```
pip install pybind11
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
This will allow you to run `cmass.survey.lightcone`. Further installation and usage instructions can be found in [cmass/lightcone/README.md](../../cmass/lightcone/README.md).
