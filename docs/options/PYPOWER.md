Installing PYPOWER
==================

Installing `pypower` is tricky because it relies on `pmesh`, which in turn relies on MPI. 

The environments I've gotten it to work in are:
- Anvil @ RCAC
```bash
x-mho1@login03.anvil:[ltu-cmass-run] $ module list

Currently Loaded Modules:
  1) gmp/6.2.1    3) mpc/1.1.0     5) gcc/11.2.0         7) numactl/2.0.14   9) xalt/2.10.45 (S)
  2) mpfr/4.0.2   4) zlib/1.2.11   6) libfabric/1.12.0   8) openmpi/4.0.6   10) modtree/cpu

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```
- Bridges2 @ PSC
```bash
(pmesh) [mho1@bridges2-login012 git]$ module list

Currently Loaded Modules:
  1) intel-advisor/2023.2.0       9) intel-dev-utilities/2021.10.0  17) intel-itac/2021.10.0
  2) intel-ccl/2021.10.0         10) intel-dnnl/2023.2.0            18) intel-mkl/2023.2.0
  3) intel-tbb/2021.10.0         11) intel-dpct/2023.2.0            19) intel-mpi/2021.10.0
  4) intel-compiler-rt/2023.2.1  12) intel-dpl/2022.2.0             20) intel-vtune/2023.2.0
  5) intel-oclfpga/2023.2.1      13) intel-icc/2023.2.1             21) intel-oneapi/2023.2.1
  6) intel-compiler/2023.2.1     14) intel-inspector/2023.2.0       22) intelmpi/2021.10.0
  7) intel-dal/2023.2.0          15) intel-ippcp-intel64/2021.8.0   23) gcc/13.3.1-p20240614
  8) intel-debugger/2023.2.0     16) intel-ipp-intel64/2021.9.0
```

I then create a new environment, and install `pmesh` and `pypower` from source, followed by `ltu-cmass` without the "full" extras:
```bash
conda create -n pmesh python=3.7
conda activate pmesh
pip install -e pmesh
pip install -e pypower
base_pkgs="numpy scipy matplotlib pandas seaborn jupyterlab ipykernel flake8 autopep8 tqdm Pylians h5py"
pip install $base_pkgs
pip install -e ltu-cmass
```
This will skip the problematic packages which clash with `pmesh`'s python==3.7 requirement.
