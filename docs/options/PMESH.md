Installing `pmesh` and running pypower
======================================

In order to run `cmass.summaries.Pk` to compute survey-level power spectrum, we require the use of meshing operators included in [`pmesh`](https://github.com/rainwoodman/pmesh). To install this, we first need `numpy`, `Cython`, and `mpi4py` installed for your given MPI configuration. On infinity @ IAP, we can install these with:

```bash
module load openmpi/4.1.2-intel gsl/2.7.1 gcc/13.3.0  # example configuration
pip install numpy mpi4py cython==0.29.33 --no-cache-dir
pip install pmesh
```

After this, you should be able to run:
```
python -m cmass.summaries.Pk
```
to measure survey-level power spectra.
