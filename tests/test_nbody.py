import subprocess
import shutil
import pytest
import os
from os.path import join


wdir = '/automnt/data80/mattho/cmass-ili'
rundir = '/home/mattho/git/ltu-cmass'


def check_outputs(sim):
    desired_files = ['config.yaml', 'nbody.h5', 'transfer.h5']
    simpath = join(wdir, 'test_nbody', sim, 'L1000-N128', '3')
    if not os.path.isdir(simpath):
        raise FileNotFoundError(f'{simpath} not found')
    for file in desired_files:
        if file not in os.listdir(simpath):
            raise FileNotFoundError(f'{file} not found in {simpath}')
    return True


@pytest.fixture(scope='module')
def setup():

    yield

    # Clean up after tests
    if os.path.isdir(join(wdir, 'test_nbody')):
        print('Cleaning up')
        shutil.rmtree(join(wdir, 'test_nbody'))


def run_bash(commands):
    result = subprocess.run(
        commands, shell=True,
        text=True, capture_output=True,
        check=False)
    if result.returncode != 0:
        print('STDOUT:\n' + result.stdout)
        print('STDERR:\n' + result.stderr)
    return result


def test_pmwd(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.nbody.pmwd nbody=testsmall nbody.suite=test_nbody
    """
    _ = run_bash(commands)
    assert check_outputs('pmwd')


def test_pinocchio(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    export LD_LIBRARY_PATH=/softs/fftw3/3.3.10-gnu-mpi/lib:$LD_LIBRARY_PATH
    python -m cmass.nbody.pinocchio nbody=testsmall nbody.suite=test_nbody nbody.transfer=CAMB
    """
    _ = run_bash(commands)
    assert check_outputs('pinocchio')


def test_borglpt(setup):
    commands = f"""
    module restore myborg
    source /data80/mattho/anaconda3/bin/activate
    conda activate borg310
    cd {rundir}
    python -m cmass.nbody.borglpt nbody=testsmall nbody.suite=test_nbody
    """
    _ = run_bash(commands)
    assert check_outputs('borg2lpt')


def test_borgpm(setup):
    commands = f"""
    module restore myborg
    source /data80/mattho/anaconda3/bin/activate
    conda activate borg310
    cd {rundir}
    python -m cmass.nbody.borgpm nbody=testsmall nbody.suite=test_nbody
    """
    _ = run_bash(commands)
    assert check_outputs('borgpm')
