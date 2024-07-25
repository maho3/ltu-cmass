
import shutil
import pytest
import os
from os.path import join
from tools import run_bash, check_outputs

wdir = '/automnt/data80/mattho/cmass-ili'
rundir = '/home/mattho/git/ltu-cmass'
nbody = 'testsmall'
suite = 'test_bias'
lhid, L, N = 3, 1000, 128


@pytest.fixture(scope="module")
def setup():
    # Set up before tests
    print('Running example nbody')
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.nbody.pmwd nbody={nbody} nbody.suite={suite}
    """
    _ = run_bash(commands)

    yield

    # Clean up after tests
    if os.path.isdir(join(wdir, suite)):
        print('Cleaning up')
        shutil.rmtree(join(wdir, suite))


def test_limd(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.bias.rho_to_halo nbody={nbody} nbody.suite={suite} bias.halo.model=LIMD
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'pmwd', suite,
        ['halos.h5'],
        lhid, L, N
    )


def test_charm(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.bias.rho_to_halo nbody={nbody} nbody.suite={suite} bias.halo.model=CHARM
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'pmwd', suite,
        ['halos.h5'],
        lhid, L, N
    )


def test_hod(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.bias.rho_to_halo nbody={nbody} nbody.suite={suite} bias.halo.model=LIMD
    python -m cmass.bias.apply_hod nbody={nbody} nbody.suite={suite}
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'pmwd', suite,
        ['galaxies/hod000.h5'],
        lhid, L, N
    )
