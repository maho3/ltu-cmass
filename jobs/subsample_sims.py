import numpy as np
import os
import yaml
import shutil

nbody = 'mtnglike'
sim = 'fastpm'
infer = 'default'
lightcone = 'ngc'
nsim = 3000
all_nsim = [500, 1000, 1500, 2000, 2500]

# Get working directory
with open("../cmass/conf/global.yaml", "r") as f:
    config = yaml.safe_load(f)
    orig_wdir = config['meta']['wdir']
    
wdir = os.path.join(orig_wdir, nbody, sim, 'models', f'{lightcone}_lightcone')

global_missing = None
global_partial = None
global_uniq = None

for summ in os.listdir(wdir):
    for kcut in os.listdir(os.path.join(wdir, summ)):
        dirname = os.path.join(wdir, summ, kcut)
        all_uniq = [] # sims which have at least one realisation
        all_full = [] # sims which have 5 realisations
        for s in ['train', 'val', 'test']:
            fname = os.path.join(dirname, f'ids_{s}.npy')
            ids = np.load(fname)
            all_uniq += list(set(ids))
            uniq, counts = np.unique(ids, return_counts=True)
            m = counts == 5
            all_full += list(uniq[m])
        all_uniq = np.array(all_uniq, dtype=int)
        all_full = np.array(all_full, dtype=int)
        missing = set(np.arange(nsim, dtype=int)) - set(all_uniq)
        partial = set(all_uniq) - set(all_full)
        print(summ, kcut, 'Number missing:', len(missing), 'Number partial:', len(partial))
        if global_missing is None:
            global_missing = list(missing)
        assert len(set(global_missing) - set(missing)) == 0
        if global_partial is None:
            global_partial = list(partial)
        if global_uniq is None:
            global_uniq = list(all_uniq)
        assert len(set(global_partial) - set(partial)) == 0
        n = 0
        for s in ['train', 'val', 'test']:
            fname = os.path.join(dirname, f'x_{s}.npy')
            x = np.load(fname)
            n += x.shape[0]
        print(n)

global_missing.sort()
global_partial.sort()
global_uniq.sort()
print(list(global_missing))
print(list(global_partial))
print(len(global_uniq))

global_uniq = np.array(global_uniq, dtype=int)
np.random.shuffle(global_uniq)

for summ in os.listdir(wdir):
    for kcut in os.listdir(os.path.join(wdir, summ)):
        dirname = os.path.join(wdir, summ, kcut)
        print('\n', summ, kcut)
        
        # Sub-sample training and validation
        for s in ['train', 'val']:
            fname = os.path.join(dirname, f'ids_{s}.npy')
            ids = np.load(fname)
            for new_nsim in all_nsim:
                new_dir = os.path.join(orig_wdir, nbody, sim, 'models', f'{lightcone}_lightcone_nsim_{new_nsim}', summ, kcut)
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir, exist_ok=True)
                uniq_ids = np.unique(ids)
                n_use = int(len(uniq_ids) / len(global_uniq) * new_nsim)
                print(f'Using {n_use} for {s}', new_nsim)
                m = np.isin(ids, uniq_ids[:n_use])
                x = np.load(os.path.join(dirname, f'x_{s}.npy'))[m,:]
                theta = np.load(os.path.join(dirname, f'theta_{s}.npy'))[m,:]
                np.save(os.path.join(new_dir, f'x_{s}.npy'), x)
                np.save(os.path.join(new_dir, f'theta_{s}.npy'), theta)
                np.save(os.path.join(new_dir, f'ids_{s}.npy'), ids[m])
                
        # Test using all simulations
        for v in ['x', 'theta', 'ids']:
            for new_nsim in all_nsim:
                new_dir = os.path.join(orig_wdir, nbody, sim, 'models', f'{lightcone}_lightcone_nsim_{new_nsim}', summ, kcut)
                shutil.copy(os.path.join(dirname, f'{v}_test.npy'), os.path.join(new_dir, f'{v}_test.npy'))
                
                



