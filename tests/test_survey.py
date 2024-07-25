
import shutil
import pytest
import os
from os.path import join
from tools import run_bash, check_outputs

wdir = '/automnt/data80/mattho/cmass-ili'
rundir = '/home/mattho/git/ltu-cmass'
nbody = 'testbig'
suite = 'test_survey'
lhid, L, N = 3, 1000, 128


@pytest.fixture(scope="module")
def setup():

    yield

    # Clean up after tests
    if os.path.isdir(join(wdir, suite)):
        print('Cleaning up')
        shutil.rmtree(join(wdir, suite))


def test_ngc_selection(setup):
    commands = f"""
    # Run nbody
    module restore myborg
    source /data80/mattho/anaconda3/bin/activate
    conda activate borg310
    cd {rundir}
    python -m cmass.nbody.borgpm nbody={nbody} nbody.suite={suite}

    # Run bias and selection
    module restore cmass
    conda activate cmass
    python -m cmass.bias.rho_to_halo nbody={nbody} nbody.suite={suite} bias.halo.model=LIMD
    python -m cmass.bias.apply_hod nbody={nbody} nbody.suite={suite}
    python -m cmass.survey.ngc_selection nbody={nbody} nbody.suite={suite}
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'borgpm', suite,
        ['lightcone/hod000_aug000.h5'],
        lhid, L, N
    )


def test_ngc_lightcone(setup):
    commands = f"""
    # Run nbody
    module restore myborg
    source /data80/mattho/anaconda3/bin/activate
    conda activate borg310
    cd {rundir}
    python -m cmass.nbody.borgpm_lc nbody={nbody} nbody.suite={suite}

    # Run bias and selection
    module restore cmass
    conda activate cmass
    python -m cmass.bias.rho_to_halo nbody={nbody} nbody.suite={suite} bias.halo.model=LIMD
    python -m cmass.bias.apply_hod nbody={nbody} nbody.suite={suite}
    python -m cmass.survey.ngc_lightcone nbody={nbody} nbody.suite={suite}
    """
    _ = run_bash(commands)
    assert check_outputs(
        wdir, 'borgpm', suite,
        ['lightcone/hod000_aug000.h5'],
        lhid, L, N
    )
