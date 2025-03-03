# Rough

import numpy as np
import pandas as pd 
import argparse, math, os, sys, glob, multiprocessing
import torch
from funcs import *
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend \
    import TorchBackend3D
from kymatio.caching import get_cache_dir

# parser = argparse.ArgumentParser(description="This takes the type of the simulation as input e.g. Om_p, fiducial")
# parser.add_argument("--nbody", type=str)
# parser.add_argument("--sim", '--names-list', nargs='+')
# args = parser.parse_args()
# nbody = args.nbody
# sim_list = args.sim

# param_dict = {
#               'fiducial':['/anvil/scratch/x-abairagi/cmass-ili/quijotelike-fid/fastpm/L1000-N128/Delta/128/galaxies/',[0.3175,0.049,0.6711,0.9624,0.834]],
# }

# grid = 128
# params=["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]

def compute_Wavelets(delta, N, out_dir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    # The actual maximum width of the dilated wavelet is 2*sigma*2^J=128, when the pixels in a box side is 128;
    sigma = 1.0
    J = 6
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
    torch.cuda.empty_cache()

    # preprocess your input if required here
    batch_x=[(delta+1)/2 ]        
    x=np.array(batch_x)
    x=torch.from_numpy(x)
    x_gpu = x.cuda()

    # 1st and 2nd-order coefficients;
    order12_gpu = scattering(x_gpu) 
    order12 = order12_gpu.cpu().numpy() 

    # Zeroth-order coefficients;
    order0_gpu = TorchBackend3D.compute_integrals(x,integral_powers)
    order0=order_0_gpu.cpu().numpy()

    return order0, order12

        for i in np.arange(len(order12)):        

            # 1st and 2nd;
            filename = os.path.join(out_dir, 'S12_J6_L6_q0.5/')
            if not os.path.exists(filename):
                os.mkdir(filename)

            filename = filename+datafile[i].split('/')[-2]+'/'

            if not os.path.exists(filename):
                os.mkdir(filename)
                
            filename = filename+'S12_J6_L6_q0.5_'+datafile[i].split('/')[-1].split('.')[0].split('_')[-1]+'.npy'
            np.save(filename,order12[i])

            # Zeroth;
            filename0 = os.path.join(out_dir, 'S0_J6_L6_q0.5/')
            if not os.path.exists(filename0):
                os.mkdir(filename0)

            filename0 = filename0+datafile[i].split('/')[-2]+'/'

            if not os.path.exists(filename0):
                os.mkdir(filename0)

            filename0 = filename0+'S0_J6_L6_q0.5'+datafile[i].split('/')[-1].split('.')[0].split('_')[-1]+'.npy'
            np.save(filename0,order_0[i])


            WPH.append([filename0, filename])

    return WPH


"""  
# Generating csv file to store paths and corresponding cosmologies (& HODs
for sim in sim_list:
    
    # modify if you are using data other than from param_dict
    if sim in ['LH', 'BSQ']: 
        path = 'Dataset/'+nbody+'/'+str(grid)+'/'+sim+'/Delta/'
        
        for i in glob.glob(path+'*'):   
            
            torch.cuda.empty_cache()
            df=pd.read_csv(i)
            dataset=df['Density_Field'].values
            values=df[params].values
        
            # output directory to store wavelets
            out_dir = '/data74/anirban/'+nbody+'/WST/'+str(grid)+'/'+sim+'/'+i.split('/')[-1].split('_')[0]+'/'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
   
            WPH = compute_Wavelets(dataset, out_dir)
            dataframe=pd.DataFrame(columns=["S0", "S12", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"])
            dataframe[["S0","S12"]]=WPH        
            dataframe[params]=values   
            # Output_dir_csv
            dataframe.to_csv("Dataset/"+nbody+"/"+str(grid)+"/"+sim+"/"+i.split('/')[-1].split('_')[0]+"_q0.5.csv")
        
    else:   
        path=param_dict[sim][0]
        dataset=DataLoader.load_df(path)
        values=param_dict[sim][1]
        
        # output directory to store wavelets
        out_dir = '/anvil/scratch/x-abairagi/cmass-ili/quijotelike-fid/'+nbody+'/L1000-N128/WST/'+str(grid)+'/'+sim+'/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)    
        WPH = compute_Wavelets(dataset, out_dir)
        dataframe=pd.DataFrame(columns=["S0", "S12", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"])
        dataframe[["S0","S12"]]=WPH        
        dataframe[params]=values   
        # Output_dir_csv
        dataframe.to_csv("Dataset/"+nbody+"/"+str(grid)+"/"+sim+"/"+sim+"_q0.5.csv")
"""
