import os
from os.path import join
import subprocess


def run_bash(commands):
    result = subprocess.run(
        commands, shell=True,
        text=True, capture_output=True,
        check=False)
    if result.returncode != 0:
        print('STDOUT:\n' + result.stdout)
        print('STDERR:\n' + result.stderr)
    return result


def check_outputs(wdir, sim, suite, desired_files, lhid=3, L=1000, N=128):
    simpath = join(wdir, suite, sim, f'L{L}-N{N}', str(lhid))
    if not os.path.isdir(simpath):
        raise FileNotFoundError(f'{simpath} not found')
    for file in desired_files:
        if not os.path.isfile(join(simpath, file)):
            raise FileNotFoundError(f'{file} not found in {simpath}')
    return True
