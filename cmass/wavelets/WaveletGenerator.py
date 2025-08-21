import numpy as np
import pandas as pd 
import argparse, math, os, sys, glob, multiprocessing
import torch
import hydra
from omegaconf import DictConfig

from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend \
    import TorchBackend3D
from kymatio.caching import get_cache_dir
from cmass.utils import get_source_path, timing_decorator, cosmo_to_astropy
from cmass.diagnostics.tools import MA, MAz
from cmass.infer.loaders import get_cosmo, get_hod 
from cmass.nbody.tools import parse_nbody_config
from cmass.bias.apply_hod import parse_hod
import glob, h5py

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

def Wavelets(dataset, out_dir, L, N, batchsize, tracer, use_rsd, add_noise):
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # The actual maximum width of the dilated wavelet is 2*sigma*2^J=128, when the pixels in a box side is 128;
    sigma = 1.0
    J = 6 #7
    # Maximum angular frequency of harmonics: L=4;
    l = 6 #4
    # The list of integral powers, which is the power applied after the modulus operation, i.e. Sum |f*g|^{integral_powers}
    integral_powers = [0.5, 1.0, 2.0, 3.0, 4.0]
    # Up to 2nd-order coefficient;
    max_order=2

    # Initialize the scattering calculation;
    scattering = HarmonicScattering3D(J=J, shape=(N, N, N),
                                      L=l, sigma_0=sigma,max_order=max_order,
                                      integral_powers=integral_powers)
    # To cuda;
    scattering.cuda()

    WPH = []
    COSMO = np.array([]).reshape(0,5)
    if tracer == "galaxies":
        HOD = np.array([]).reshape(0,5)
    pfx = 'z' if use_rsd else ''    

    ## Loop over the simulated boxes to calculate the coefficients;
    # Batch size of the calculation at a time;
    
    batches=divide_into_batches(dataset, batch_size=batchsize)    

    for batch in batches:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        datafile = batch
        
        # get_cosmo
        source_list = [file.split('/') for file in datafile]
        q = [file.pop(-1) for file in source_list]
        if tracer == "galaxies":
            q = [file.pop(-1) for file in source_list]
        source_list = [os.path.join('/', *file) for file in source_list]
        cosmo = [get_cosmo(source_path) for source_path in source_list]
        COSMO = np.vstack([COSMO,cosmo])
        
        # get_hod
        if tracer == "galaxies":
            diagfile_list = [file.split('/') for file in datafile]
            q = [file.insert(-2,'diag') for file in diagfile_list]
            diagfile_list = [os.path.join('/', *file) for file in diagfile_list]
            hod = [get_hod(diagfile) for diagfile in diagfile_list]
            HOD = np.vstack([HOD,hod])

        
        # preprocess your input if required here
        pos_file = [h5py.File(file)['0.666667']['pos'][()].astype(np.float32) for file in datafile]
        vel_file = [h5py.File(file)['0.666667']['vel'][()].astype(np.float32) for file in datafile]
        
        if add_noise:
            Lnoise = (1000/128)/np.sqrt(3)  # Set by CHARM resolution

            if use_rsd:
                fields = [MAz(pos + (np.random.randn(*pos.shape)*Lnoise).astype(np.float32), vel_file[i], L, N, 
                            cosmo_to_astropy(cosmo[i]), z=0.5, MAS='CIC', axis=0).astype(np.float32) for i, pos in enumerate(pos_file)]
            else:
                fields = [MA(pos + (np.random.randn(*pos.shape)*Lnoise).astype(np.float32), L, N, 
                            MAS='CIC').astype(np.float32) for pos in pos_file]
        else:
            if use_rsd:
                fields = [MAz(pos, vel_file[i], L, N, 
                            cosmo_to_astropy(cosmo[i]), z=0.5, MAS='CIC', axis=0).astype(np.float32) for i, pos in enumerate(pos_file)]
            else:
                fields = [MA(pos, L, N, MAS='CIC').astype(np.float32) for pos in pos_file]
        
        batch_x = [(delta+1)/2 for delta in fields]
        
        
        
        x=np.array(batch_x)
        x=torch.from_numpy(x)
        x_gpu = x.cuda()

        # 1st and 2nd-order coefficients;
        order12_gpu = scattering(x_gpu) 
        order12 = order12_gpu.cpu().numpy() 

        # Zeroth-order coefficients;
        order0_gpu = TorchBackend3D.compute_integrals(x,integral_powers)
        order0=order0_gpu.cpu().numpy()
        
        for i in np.arange(len(datafile)):       

            if tracer == "nbody" or tracer == "halos":
                ind = datafile[i].split('/')[-2]
                filename = os.path.join(out_dir, datafile[i].split('/')[-2],'diag/',
                                        datafile[i].split('/')[-1].split('.')[0]+'_wst.h5')
                                
            if tracer == "galaxies":
                ind = datafile[i].split('/')[-3]
                filename = os.path.join(out_dir, datafile[i].split('/')[-3],'diag/galaxies/',
                                        datafile[i].split('/')[-1].split('.')[0]+'_wst.h5')
                       
            with h5py.File(filename, "a") as dataset:
                
                dataset[pfx+'S12'] = order12[i]
                dataset[pfx+'S0'] = order0[i]

            WPH.append([ind, filename])
            


    if tracer == "galaxies":
        return WPH, COSMO, HOD
    else:
        return WPH, COSMO

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, str('*'), check=False
    )
    
    if cfg.diag.density:
        dataset = glob.glob(source_path+'/nbody.h5') 
        tracer = "nbody"
    
    elif cfg.diag.halo:
        dataset = glob.glob(source_path+'/halos.h5') 
        tracer = "halos"
        
    elif cfg.diag.galaxy:
        dataset = glob.glob(source_path+'/galaxies/*.h5')  
        tracer = "galaxies"
    
    # dataset=[]
    # for i in glob.glob("/anvil/scratch/x-abairagi/cmass-ili/abacuslike/fastpm/L2000-N256/*"):
    #     try:
    #         a=h5py.File(i+"/halos.h5")['0.666660'].keys()
    #         dataset.append(i+"/halos.h5")
    #     except:
    #         continue
        
        
    if cfg.diag.rsd:
        use_rsd=True
    else:
        use_rsd=False

    if cfg.diag.add_noise:
        add_noise=True
    else:
        add_noise=False
    
    pfx = 'z' if use_rsd else ''
    
    # output directory to store wavelets
    str_list = source_path.split('/')
    q = str_list.pop(-1)
    out_dir = os.path.join('/', *str_list)
    
    if cfg.diag.galaxy:
        WPH, COSMO, HOD = Wavelets(dataset, out_dir, cfg.nbody.L, cfg.nbody.N, cfg.diag.batchsize, tracer, use_rsd, add_noise) 
        dataframe = pd.DataFrame(columns=["Filename", "Omega_m", "Omega_b", "h", "n_s", "sigma_8", 
                                        'logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha'])
        dataframe[["id", "Filename"]]= WPH  
        dataframe[cosmo_params] = COSMO 
        dataframe[hod_params] = HOD 
    
    else:
        WPH, COSMO = Wavelets(dataset, out_dir, cfg.nbody.L, cfg.nbody.N, cfg.diag.batchsize, tracer, use_rsd, add_noise)
        dataframe = pd.DataFrame(columns=["Filename", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"]) 
        dataframe[["id", "Filename"]] = WPH 
        dataframe[cosmo_params] = COSMO  
     
        
    # Output_dir_csv
    out_csv=os.path.join("cmass/wavelets/Dataset", cfg.nbody.suite, cfg.sim, f'L{cfg.nbody.L}-N{cfg.nbody.N}')
    os.makedirs(out_csv, exist_ok=True)
    dataframe.to_csv(out_csv+"/"+pfx+tracer+".csv", index=False)
        
if __name__ == "__main__":
    main()