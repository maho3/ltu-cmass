import shutil
import pytest
import os
from os.path import join
from tools import run_bash, check_outputs


wdir = '/automnt/data80/mattho/cmass-ili'
rundir = '/home/mattho/git/ltu-cmass'
nbody = 'testsmall'
suite = 'test_nbody'
lhid, L, N = 3, 1000, 128


@pytest.fixture(scope='module')
def setup():

    yield

    # Clean up after tests
    if os.path.isdir(join(wdir, 'test_nbody')):
        print('Cleaning up')
        shutil.rmtree(join(wdir, 'test_nbody'))


def test_pmwd(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.nbody.pmwd nbody={nbody} nbody.suite={suite}
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'pmwd', suite,
        ['config.yaml', 'nbody.h5', 'transfer.h5'],
        lhid, L, N
    )


def test_pinocchio(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    export LD_LIBRARY_PATH=/softs/fftw3/3.3.10-gnu-mpi/lib:$LD_LIBRARY_PATH
    python -m cmass.nbody.pinocchio nbody={nbody} nbody.suite={suite} nbody.transfer=CAMB
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'pinocchio', suite,
        ['config.yaml', 'nbody.h5'],  # pinocchio doesn't do transfer files
        lhid, L, N
    )


def test_borglpt(setup):
    commands = f"""
    module restore myborg
    source /data80/mattho/anaconda3/bin/activate
    conda activate borg310
    cd {rundir}
    python -m cmass.nbody.borglpt nbody={nbody} nbody.suite={suite}
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'borg2lpt', suite,
        ['config.yaml', 'nbody.h5', 'transfer.h5'],
        lhid, L, N
    )


def test_borgpm(setup):
    commands = f"""
    module restore myborg
    source /data80/mattho/anaconda3/bin/activate
    conda activate borg310
    cd {rundir}
    python -m cmass.nbody.borgpm nbody={nbody} nbody.suite={suite}
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'borgpm', suite,
        ['config.yaml', 'nbody.h5', 'transfer.h5'],
        lhid, L, N
    )
