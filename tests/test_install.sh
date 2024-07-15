#PBS -N install_cmass
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/cmass-ili/logs_test/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=04:00:00
#PBS -l nodes=1:ppn=4,mem=8gb

# Configuration
INSTALLPATH=/home/mattho/git/ltu-cmass
TESTENV=test_cmass
source ~/.bashrc

# Move to directory
echo "Move to directory: $INSTALLPATH"
cd $INSTALLPATH

# Create test environment
echo "Create test environment: $TESTENV"
conda create -n $TESTENV python=3.10 -y
conda activate $TESTENV

# Install pmesh (for pypower)
echo "Install pmesh"
module purge
module load openmpi/4.1.2-intel gsl/2.7.1 gcc/13.3.0  # example configuration
pip install numpy mpi4py cython==0.29.33 --no-cache-dir
pip install pmesh

# Build lightcone
echo "Build lightcone"
cd $INSTALLPATH/cmass/lightcone
make clean
pip install pybind11
make
cd $INSTALLPATH

# Install ltu-cmass
echo "Install ltu-cmass"
pip install -e .

# Run tests
echo "Run tests"
python -c "import cmass.nbody.pmwd; print('pmwd ok')"
python -c "import cmass.bias.rho_to_halo; print('rho_to_halo ok')"
python -c "import cmass.bias.apply_hod; print('apply_hod ok')"
python -c "import cmass.survey.ngc_selection; print('ngc_selection ok')"
python -c "import cmass.survey.ngc_lightcone; print('ngc_lightcone ok')"
python -c "import cmass.filter.single_filter; print('single_filter ok')"
python -c "import cmass.summaries.Pk; print('Pk ok')"
python -c "import pmesh; print('pmesh ok')"  # only if pmesh installed correctly

# Deactivate environment
echo "Deactivate environment"
conda deactivate
module purge

# Remove test environment
echo "Remove test environment"
conda remove -n $TESTENV --all -y
