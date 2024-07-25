import subprocess
import shutil
import pytest
import os
from os.path import join

wdir = '/automnt/data80/mattho/cmass-ili'
rundir = '/home/mattho/git/ltu-cmass'
nbody = 'testsmall'
suite = 'test_bias'


def run_bash(commands):
    result = subprocess.run(
        commands, shell=True,
        text=True, capture_output=True,
        check=False)
    if result.returncode != 0:
        print('STDOUT:\n' + result.stdout)
        print('STDERR:\n' + result.stderr)
    return result


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


def check_outputs(sim):
    desired_files = ['halos.h5']
    simpath = join(wdir, suite, sim, 'L1000-N128', '3')
    if not os.path.isdir(simpath):
        raise FileNotFoundError(f'{simpath} not found')
    for file in desired_files:
        if file not in os.listdir(simpath):
            raise FileNotFoundError(f'{file} not found in {simpath}')
    return True


def test_limd(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.bias.rho_to_halo nbody={nbody} nbody.suite={suite} bias.halo.model=LIMD
    """
    _ = run_bash(commands)
    assert check_outputs('pmwd')


def test_charm(setup):
    commands = f"""
    module restore cmass
    source /data80/mattho/anaconda3/bin/activate
    conda activate cmass
    cd {rundir}
    python -m cmass.bias.rho_to_halo nbody={nbody} nbody.suite={suite} bias.halo.model=CHARM
    """
    _ = run_bash(commands)
    assert check_outputs('pmwd')
