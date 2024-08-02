Installing FASTPM
=================

We use the [FastPM simulator](https://github.com/fastpm/fastpm/tree/master) to run PM simulations in `cmass.nbody.fastpm`. If you don't want to use this feature, you can skip this section.

To install FastPM, it requires an `mpi` and `gsl` distribution. On most supercomputing systems, these are already installed and easy to load with e.g. `module load openmpi gsl`.

For example, the following commands will install FastPM on Infinity@IAP:
```bash
# setup environment
module purge
module load openmpi/4.1.2-gnu gsl/2.7.1

# clone FastPM github
git clone git@github.com:fastpm/fastpm.git
cd fastpm

# copy local makefile
cp Makefile.local.example Makefile.local

# edit Makefile.local
vim Makefile.local  # Uncomment the line 'CC = mpicc'

# build FastPM
make

# check that it built correctly
./src/fastpm --help
```

Then, change your `nbody` configuration to the path of your FastPM executable:
```yaml
fastpm_exec: /path/to/git/fastpm/src/fastpm
```
An example of this is in [fastpm.yaml](../../cmass/conf/nbody/fastpm.yaml).
