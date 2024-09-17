Installing Summarizer
====================

To compute summary statistics, this pipeline includes the option to use the [ili-summarizer](https://github.com/florpi/ili-summarizer/tree/main)
package. 

To install this, one must first clone, install and fix a specific version of `pycorr`, 
which requires a fix due to an issue related to whether the code runs on a GPU or not.

This can be done by first running
```bash
pip install git+https://github.com/adematti/Corrfunc@desi
```
and then navigating to the file `pycorr/corrfunc.py`. One must then comment out the lines 
https://github.com/cosmodesi/pycorr/blob/29df2c2d700df77781f20c125e982fbf3ffe9daf/pycorr/corrfunc.py#L113-L114
and replace them with
```python
kwargs['gpu'] = attrs.pop('gpu', False)
```

One this is done, one must load mpi and install the corresponding `mpi4py`
```bash
module load gcc/11.2.0
module load openmpi/4.1.6
module load gsl/2.4
env MPICC=$(which mpicc) pip3 install mpi4py --no-cache-dir
```

Now one can install the remaining dependencies
```bash
pip install cython==0.29.33 --no-cache-dir
pip install pmesh
```

and finally clone and install the `summarizer` package

```bash
git clone git@github.com:florpi/ili-summarizer.git
pip install -e ili-summarizer[all]
```

Before running the summarizer on anvil, one should also run
```bash
export LD_PRELOAD=/usr/lib64/libslurm.so
```

To check whether this has worked, one can simply run
```bash
python -c "import summarizer"
```

Note that this will currently not work with an interactive session on the anvil cluster,
but works for batch jobs.
