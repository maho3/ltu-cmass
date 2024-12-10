import yaml
import random
import os

file_path = '/home/x-dbartlett/ltu-cmass/cmass/conf/infer/quijotelike.yaml'
fcn_hidden = [64, 32, 16] 
nmodel = 3
Nsim = 200
gravity = 'fastpm'

def generate_random_network_args():
    network_types = ['mdn', 'maf', 'nsf', 'made']
    selected_network = random.choice(network_types)

    if selected_network == 'mdn':
        args = {
            'model': 'mdn',
            'num_components': random.randint(2, 10),
            'hidden_features': random.randint(16, 128),
        }
    else:
        args = {
            'model': selected_network,
            'hidden_features': random.randint(16, 128),
            'num_transforms': random.randint(2, 6),
        }

    return args

# Set the seed for reproducibility
random.seed(42)

nets = [None] * nmodel

for i in range(nmodel):
    nets[i] = generate_random_network_args()
    print(i, nets[i])


data = {
        'save_dir': f'/anvil/scratch/x-dbartlett/cmass/quijotelike/{gravity}/models_nets{nmodel}_sims{Nsim}',  # where to save trained models (default e.g. wdir/quijotelike/fastpm/models)
        'halo': True,      # whether to train on halo summaries
        'galaxy': False,    # whether to train on galaxy summaries
        'ngc_lightcone': False,  # whether to train on ngc_lightcone summaries
        'sgc_lightcone': False,  # whether to train on sgc_lightcone summaries

        'Nmax': Nsim,  # maximum number of simulations to use for training (-1 for all)
        'val_frac': 0.1,  # fraction of data to use for validation
        'test_frac': 0.1,  # fraction of data to use for testing

        'prior': 'Uniform',  # prior to use for training (only uniform)

        'backend': 'sbi',  # backend to use for training (sbi/lampe/pydelfi)
        'engine': 'NPE',     # engine to use for training (NPE/NLE/NRE)
        'nets': nets,

        'fcn_hidden': fcn_hidden,  # hidden layer sizes for the FCN
        'batch_size': 64,  # batch size
        'learning_rate': 1e-3,  # learning rate

        'device': 'cuda',  # cpu or cuda

        'experiments': [
            {
                'summary': ['Pk0'],  # monopole
                'kmax': [0.1] #, 0.2, 0.3, 0.4]
            },
            # {
            #     'summary': ['Pk0', 'Pk2', 'Pk4'],  # monopole, quadrupole, hexadecapole
            #     'kmax': [0.1, 0.2, 0.3, 0.4]
            # },
            # {
            #     'summary': ['zPk0', 'zPk2', 'zPk4'],  # redshift-space monopole, quadrupole, hexadecapole
            #     'kmax': [0.1, 0.2, 0.3, 0.4]
            # }
        ]
    }

if not os.path.isdir(data['save_dir']):
    os.mkdir(data['save_dir'])

with open(file_path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

os.system(f'cp {file_path} {data["save_dir"]}')

