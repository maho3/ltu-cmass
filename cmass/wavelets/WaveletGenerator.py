import numpy as np
import pandas as pd 
import argparse, math, os, sys, glob, multiprocessing
import torch

from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend \
    import TorchBackend3D
from kymatio.caching import get_cache_dir
from cmass.utils import get_source_path, timing_decorator
from cmass.diagnostics.tools import MA
from cmass.infer.loaders import get_cosmo, get_hod 
import glob

cosmo_params=["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]
hod_params=['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha']

def divide_into_batches(X, batch_size = 64):
    
    m = len(X)                  
    mini_batches = []
        
    num_complete_minibatches = math.floor(m/batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = X[k*batch_size:(k+1)*batch_size]
        
        mini_batches.append(mini_batch_X)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:

        mini_batch_X = X[num_complete_minibatches*batch_size:]
        
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch_X)
    
    return mini_batches

def Wavelets(dataset, out_dir, L, N, batchsize=128):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # The actual maximum width of the dilated wavelet is 2*sigma*2^J=128, when the pixels in a box side is 128;
    sigma = 1.0
    J = 6
    # Maximum angular frequency of harmonics: L=4;
    L = 6
    # The list of integral powers, which is the power applied after the modulus operation, i.e. Sum |f*g|^{integral_powers}
    integral_powers = [0.5, 1.0, 2.0, 3.0, 4.0]
    # Up to 2nd-order coefficient;
    max_order=2

    # Initialize the scattering calculation;
    scattering = HarmonicScattering3D(J=J, shape=(N, N, N),
                                      L=L, sigma_0=sigma,max_order=max_order,
                                      integral_powers=integral_powers)
    # To cuda;
    scattering.cuda()

    WPH=[]
    COSMO=np.array([]).reshape(0,5)
    HOD=np.array([]).reshape(0,5)

    ## Loop over the simulated boxes to calculate the coefficients;
    # Batch size of the calculation at a time;
    
    batches=divide_into_batches(dataset, batch_size=batchsize)    

    for batch in batches:
        torch.cuda.empty_cache()
        datafile = batch
        
        # preprocess your input if required here
        pos_file = [h5py.File(file)['0.666667']['pos'][()] for file in datafile]
        fields = [MA(pos, L, N, MAS='CIC').astype(np.float32) for pos in pos_file]
        batch_x=[(delta+1)/2 for delta in fields]
        
        # get_cosmo
        source_list = [file.split('/') for file in datafile]
        q = [file.pop(-1) for file in source_list]
        q = [file.pop(-1) for file in source_list]
        source_list = [os.path.join('/', *file) for file in source_list]
        cosmo = [get_cosmo(source_path) for source_path in source_list]
        COSMO = np.vstack([COSMO,cosmo])
        
        # get_hod
        diagfile_list = [file.split('/') for file in datafile]
        q = [file.insert(-2,'diag') for file in diagfile_list]
        diagfile_list = [os.path.join('/', *file) for file in diagfile_list]
        hod = [get_hod(diagfile) for diagfile in diagfile_list]
        HOD=np.vstack([HOD,hod])
               
        
        x=np.array(batch_x)
        x=torch.from_numpy(x)
        x_gpu = x.cuda()

        # 1st and 2nd-order coefficients;
        order12_gpu = scattering(x_gpu) 
        order12 = order12_gpu.cpu().numpy() 

        # Zeroth-order coefficients;
        order0_gpu = TorchBackend3D.compute_integrals(x,integral_powers)
        order0=order0_gpu.cpu().numpy()


        for i in np.arange(len(order12)):        

            # 1st and 2nd;
            filename = os.path.join(out_dir, datafile[i].split('/')[-3],'/diag/galaxies/WST/')
            if not os.path.exists(filename):
                os.makedirs(filename, exist_ok=True)
                
            filename12 = os.path.join(filename, 'S12_J6_L6_q0.5/')
            if not os.path.exists(filename12):
                os.mkdir(filename12)
                
            filename12 = filename12 + datafile[i].split('/')[-1].split('.')[0]+'.npy'
            np.save(filename12,order12[i])

            # Zeroth;
            filename0 = os.path.join(filename, 'S0_J6_L6_q0.5/')
            if not os.path.exists(filename0):
                os.mkdir(filename0)

            filename0 = filename0 + datafile[i].split('/')[-1].split('.')[0]+'.npy'
            np.save(filename0,order0[i])

            WPH.append([filename0, filename])

    return WPH, COSMO, HOD


def main(cfg: DictConfig) -> None:
    # cfg = parse_nbody_config(cfg)
    # cfg = parse_hod(cfg)
    
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, str('*')
    )
    
    dataset=glob.glob(source_path+'galaxies/*.h5')   # change for other tracers
        
    # output directory to store wavelets
    l = source_path.split('/')
    q = l.pop(-1)
    out_dir = os.path.join('/', *l)
        
    WPH, COSMO, HOD = Wavelets(dataset, out_dir, cfg.nbody.L, cfg.nbody.N, batchsize=128)
    dataframe=pd.DataFrame(columns=["S0", "S12", "Omega_m", "Omega_b", "h", "n_s", "sigma_8", 
                                        'logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha'])
    dataframe[["S0","S12"]]=WPH        
    dataframe[cosmo_params]=COSMO  
    dataframe[hod_params]=hod  
        
        # Output_dir_csv
    out_csv=os.path.join("Dataset", cfg.nbody.suite, cfg.sim, f'L{L}-N{N}')
    os.makedirs(out_csv, exist_ok=True)
    dataframe.to_csv(out_csv+"/"+"wst_q0.5.csv")
        