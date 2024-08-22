import yaml
import os

def make_yaml(lhid, N,):

    suite = f'pin_1gpch_z0.5_id{lhid}_N{N}'
    pin_dir = '/home/x-dbartlett/Pinocchio/src/'

    # Define the data structure for the YAML file
    data = {
        'suite' : suite,
        'L': 1000,  # Mpc/h
        'N': N,  # meshgrid resolution
        'lhid': lhid,  # latin hypercube id
        'matchIC': 0,  # whether to match ICs to file (0 no, 1 yes, 2 quijote)
        'save_particles': False,  # whether to save particle data
        'save_transfer': True,  # whether to save transfer fn densities (for CHARM)
        'zi': 20,  # initial redshift
        'zf': 0.5,  # final redshift
        'transfer': 'CAMB',  # transfer function (EH, CLASS, CAMB or SYREN. Only EH or CLASS for borg)
        'mass_function': 'Watson_2013',  # which output HMF to use
        'pinocchio_exec': f'{pin_dir}/pinocchio.x'  # pinocchio executable path
    }

    outdir = '../cmass/conf/nbody/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    with open(f'{outdir}/{suite}.yaml', 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)    

    return suite


def make_run_script(all_suites):

    bash_script = """#!/bin/sh

# Cancel job if error raised
set -e

# Modules
module purge
module load gcc/11.2.0
module load openmpi/4.1.6
module load gsl/2.4
module load fftw/3.3.8
module load anaconda

# Environment
conda deactivate
conda activate cmass

cd ..

# Run Pinocchio"""

    for suite in all_suites:
        bash_script += f"\npython -m cmass.nbody.pinocchio nbody={suite}"

    bash_script += """\n\nconda deactivate
exit 0
"""

    with open('run_pinocchio_tests.sh', 'w') as f:
        print(bash_script, file=f)

    os.system('chmod u+x run_pinocchio_tests.sh')

    return


def main():

    all_N = [128, 256, 384,]
    all_N = [512]
    lhid = 3

    all_suites = []

    for N in all_N:
        all_suites.append(make_yaml(lhid, N,))

    make_run_script(all_suites)

    return


if __name__ == "__main__":
    main()

