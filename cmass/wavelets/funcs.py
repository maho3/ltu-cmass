import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, h5py
import math, random
from GPUtil import showUtilization as gpu_usage
import torch
from torch import nn
# from torchvision import models, transforms, datasets


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    print("GPU Usage after emptying the cache")
    gpu_usage()
    
def process_data(data, params):

    s0=np.array([h5py.File(i)['S0'][()].astype(np.float64).flatten() for i in data['Filename']])
    s12=np.array([h5py.File(i)['S12'][()].astype(np.float64).flatten() for i in data['Filename']])
    
    # s0=np.array([np.load(i).astype(np.float64).flatten() for i in data['S0']])
    # s12=np.array([np.load(i).astype(np.float64).flatten() for i in data['S12']])
    
    x=np.concatenate([s0,s12],1)
    x=np.log(x)
    theta=data[params].values
    
    return x, theta

def normalize(train_x, test_x):

    mean=train_x.mean(axis=0)
    std=train_x.std(axis=0)
    test_x=(test_x-mean)/std
        
    return test_x

def generate_sample(posterior, data, num_samples=1000, device='cuda'):
    samples = posterior.sample((num_samples,), torch.Tensor(data).to(device))
    return samples.cpu().numpy()

def evaluate_posterior(posterior, x, theta):
    log_prob = posterior.log_prob(theta=theta, x=x)
    return log_prob.mean().item()

# @hydra.main(version_base=None, config_path="../conf", config_name="config")
class DataLoader:
    def load_data(nbody="quijotelike", sim="fastpm", tracer="halo", summaries=[],):
        cfg = parse_nbody_config(cfg)
        cfg = parse_hod(cfg)
        for summary_name in summaries:
            filename = os.path.join(cfg.meta.wdir,nbody,sim,"models",tracer,summary_name)
            x_train = np.load(filename+"x_train.npy")
            x_val = np.load(filename+"x_val.npy")
            x_test = np.load(filename+"x_test.npy")
            theta_train = np.load(filename+"theta_train.npy")
            theta_val = np.load(filename+"theta_val.npy")
            theta_test = np.load(filename+"theta_test.npy")
        
    def load_df(path):
    
        return glob.glob(path+"*/*.npy")
    
    def load_param(path):
        param=open(path,'r+')
        param_list=[]
        for i, line in enumerate(param.readlines()):
            if i!=0:
                Omega_m, Omega_b, h, n_s, sigma_8=map(eval,line.split(' '))  #change when required
                param_list.append([Omega_m, Omega_b, h, n_s, sigma_8])       # 

        return param_list
    
    def load_LH_datasets(path_of_data,path_of_param):
        dataframe=pd.DataFrame()
        dataframe['Density_Field']=DataLoader.load_df(path_of_data)
        dataframe[["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]]=DataLoader.load_param(path_of_param)

        return dataframe


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




def cov(output):        #covariance from predicted parameters 
    
    dim = output.shape
    mean = np.mean(output,axis=0).reshape(1, dim[1])
    arr = output - mean
    mat = np.matmul(arr.reshape(*dim,1),np.moveaxis(arr.reshape(*dim,1),-1,1))        
    mat = np.array(mat)
    
    return np.mean(mat,axis=0)

def cov_from_fid_wst(path=""):        #covariance from predicted parameters 
    
#     path = "Dataset/128/Fiducial/fid_wst.csv"
    df = pd.read_csv(path)
    data0 = df['S0'].values
    data12 = df['S12'].values
#     summary = np.concatenate([[np.load(i).astype(np.float128).flatten() for i in data0], [np.load(j).astype(np.float128).flatten() for j in data12]], axis=1)
    
    summary = np.concatenate([[np.load(i)[:3].astype(np.float128).flatten() for i in data0], [np.load(j)[:,:,:3].astype(np.float128).flatten() for j in data12]], axis=1)
    
    return cov(summary)


def del_mu_from_direct_WST(parameters, order, J=6):  #this will give transpose
    
    param_pm_dict = {'Omega_m':[['Dataset/128/Om_p/Om_p_wst_q0.5.csv',[0.3275,0.049,0.6711,0.9624,0.834]],['Dataset/128/Om_m/Om_m_wst_q0.5.csv',[0.3075,0.049,0.6711,0.9624,0.834]]],
                'Omega_b':[['Dataset/128/Ob2_p/Ob2_p_wst_q0.5.csv',[0.3175,0.050,0.6711,0.9624,0.834]],['Dataset/128/Ob2_m/Ob2_m_wst_q0.5.csv',[0.3175,0.048,0.6711,0.9624,0.834]]],
                 'h':[['Dataset/128/h_p/h_p_wst_q0.5.csv',[0.3175,0.049,0.6911,0.9624,0.834]],['Dataset/128/h_m/h_m_wst_q0.5.csv',[0.3175,0.049,0.6511,0.9624,0.834]]],
                 'n_s':[['Dataset/128/ns_p/ns_p_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9824,0.834]],['Dataset/128/ns_m/ns_m_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9424,0.834]]], 
                 'sigma_8':[['Dataset/128/s8_p/s8_p_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9624,0.849]],['Dataset/128/s8_m/s8_m_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9624,0.819]]]}
       
    
    mat= []
    for num, element in enumerate(parameters):
            
        path_p=param_pm_dict[element][0][0]
        data_p=pd.read_csv(path_p)
        value_p=param_pm_dict[element][0][1]             
        
        path_m=param_pm_dict[element][1][0]
        data_m=pd.read_csv(path_m)
        value_m=param_pm_dict[element][1][1]

        if order==0:
            data_p=data_p['S0'].values
            data_m=data_m['S0'].values
            
#             output_p = np.array([np.load(i).astype(np.float128) for i in data_p])
#             output_m = np.array([np.load(i).astype(np.float128) for i in data_m])        
            
            output_p = np.array([np.load(i)[[0,2,3,4]].astype(np.float64).flatten() for i in data_p])
            output_m = np.array([np.load(i)[[0,2,3,4]].astype(np.float64).flatten() for i in data_m])        
        
        
        elif order==1:
            data_p=data_p['S12'].values
            data_m=data_m['S12'].values
        
#             output_p = np.array([np.load(i).astype(np.float128)[:(J+1)].flatten() for i in data_p])
#             output_m = np.array([np.load(i).astype(np.float128)[:(J+1)].flatten() for i in data_m]) 
        
            output_p_0 = np.array([np.load(i)[:J,0,[0,2,3,4]].astype(np.float64).flatten() for i in data_p])
            output_p_1 = np.array([np.load(i)[:J,1:,:].astype(np.float64).flatten() for i in data_p])
            output_p_2 = np.array([np.load(i)[J,1,:].astype(np.float64).flatten() for i in data_p])
            
            output_m_0 = np.array([np.load(i)[:J,0,[0,2,3,4]].astype(np.float64).flatten() for i in data_m])
            output_m_1 = np.array([np.load(i)[:J,1:,:].astype(np.float64).flatten() for i in data_m])
            output_m_2 = np.array([np.load(i)[J,1,:].astype(np.float64).flatten() for i in data_m])
            
            output_p=np.concatenate([output_p_0,output_p_1,output_p_2], axis=1)
            output_m=np.concatenate([output_m_0,output_m_1,output_m_2], axis=1)
        
        elif order==2:
            data_p=data_p['S12'].values
            data_m=data_m['S12'].values
        
#             output_p = np.array([np.load(i).astype(np.float128)[(J+1):].flatten() for i in data_p])
#             output_m = np.array([np.load(i).astype(np.float128)[(J+1):].flatten() for i in data_m])

            ex=np.arange(J,0,-1)
            ex[0]=ex[0]-1
            for i in range(1,J):
                ex[i]=ex[i]+ex[i-1]
                
            output_p_0= np.array([(np.delete(np.load(i)[(J+1):,0,[0,2,3,4]],ex,0)).astype(np.float64).flatten() for i in data_p])
            output_p_1 = np.array([np.load(i)[(J+1):,1:,:].astype(np.float64).flatten() for i in data_p])
            
            output_m_0= np.array([(np.delete(np.load(i)[(J+1):,0,[0,2,3,4]],ex,0)).astype(np.float64).flatten() for i in data_m])
            output_m_1 = np.array([np.load(i)[(J+1):,1:,:].astype(np.float64).flatten() for i in data_m]) 
            
            output_p=np.concatenate([output_p_0,output_p_1], axis=1)
            output_m=np.concatenate([output_m_0,output_m_1], axis=1)
        
            
        else:
            print("Please modify the function del_mu_from_direct_WST in oder to incorporate the higher order") 
        
            
        mean_p = np.mean(output_p, axis=0)
        mean_m = np.mean(output_m, axis=0)
      
        derivative=(mean_p - mean_m)/(value_p[num]-value_m[num])
       
        mat.append(derivative)
        
    mat=np.array(mat)
        
        
    return mat


def del_mu_individual_seed_from_direct_WST(parameters, order, J=6):  #this will give transpose
    
    param_pm_dict = {'Omega_m':[['Dataset/128/Om_p/Om_p_wst_q0.5.csv',[0.3275,0.049,0.6711,0.9624,0.834]],['Dataset/128/Om_m/Om_m_wst_q0.5.csv',[0.3075,0.049,0.6711,0.9624,0.834]]],
                'Omega_b':[['Dataset/128/Ob2_p/Ob2_p_wst_q0.5.csv',[0.3175,0.050,0.6711,0.9624,0.834]],['Dataset/128/Ob2_m/Ob2_m_wst_q0.5.csv',[0.3175,0.048,0.6711,0.9624,0.834]]],
                 'h':[['Dataset/128/h_p/h_p_wst_q0.5.csv',[0.3175,0.049,0.6911,0.9624,0.834]],['Dataset/128/h_m/h_m_wst_q0.5.csv',[0.3175,0.049,0.6511,0.9624,0.834]]],
                 'n_s':[['Dataset/128/ns_p/ns_p_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9824,0.834]],['Dataset/128/ns_m/ns_m_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9424,0.834]]], 
                 'sigma_8':[['Dataset/128/s8_p/s8_p_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9624,0.849]],['Dataset/128/s8_m/s8_m_wst_q0.5.csv',[0.3175,0.049,0.6711,0.9624,0.819]]]}
       
    
    mat= {}
    for num, element in enumerate(parameters):
            
        path_p=param_pm_dict[element][0][0]
        data_p=pd.read_csv(path_p)
        value_p=param_pm_dict[element][0][1]             
        
        path_m=param_pm_dict[element][1][0]
        data_m=pd.read_csv(path_m)
        value_m=param_pm_dict[element][1][1]

        if order==0:
            data_p=data_p['S0'].values
            data_m=data_m['S0'].values
            
            output_p = np.array([np.load(i).astype(np.float128) for i in data_p])
            output_m = np.array([np.load(i).astype(np.float128) for i in data_m])        
            
        
        
        elif order==1:
            data_p=data_p['S12'].values
            data_m=data_m['S12'].values
        
            output_p = np.array([np.load(i).astype(np.float128)[:(J+1)].flatten() for i in data_p])
            output_m = np.array([np.load(i).astype(np.float128)[:(J+1)].flatten() for i in data_m]) 
        
        
        elif order==2:
            data_p=data_p['S12'].values
            data_m=data_m['S12'].values
        
            output_p = np.array([np.load(i).astype(np.float128)[(J+1):].flatten() for i in data_p])
            output_m = np.array([np.load(i).astype(np.float128)[(J+1):].flatten() for i in data_m])
            
            
        else:
            print("Please modify the function del_mu_from_direct_WST in oder to incorporate the higher order") 
        
            
      
        dx_dtheta=(output_p - output_m)/(value_p[num]-value_m[num])
       
        mat[element]=dx_dtheta        
          
    return mat



def del_mu_from_ili(parameters, posterior_ensemble):  #this will give transpose
    
    param_pm_dict = {'Omega_m':[['Dataset/128/Om_p/Om_p.csv',[0.3275,0.049,0.6711,0.9624,0.834]],['Dataset/128/Om_m/Om_m.csv',[0.3075,0.049,0.6711,0.9624,0.834]]],
                'Omega_b':[['Dataset/128/Ob2_p/Ob2_p.csv',[0.3175,0.050,0.6711,0.9624,0.834]],['Dataset/128/Ob2_m/Ob2_m.csv',[0.3175,0.048,0.6711,0.9624,0.834]]],
                 'h':[['Dataset/128/h_p/h_p.csv',[0.3175,0.049,0.6911,0.9624,0.834]],['Dataset/128/h_m/h_m.csv',[0.3175,0.049,0.6511,0.9624,0.834]]],
                 'n_s':[['Dataset/128/ns_p/ns_p.csv',[0.3175,0.049,0.6711,0.9824,0.834]],['Dataset/128/ns_m/ns_m.csv',[0.3175,0.049,0.6711,0.9424,0.834]]], 
                 'sigma_8':[['Dataset/128/s8_p/s8_p.csv',[0.3175,0.049,0.6711,0.9624,0.849]],['Dataset/128/s8_m/s8_m.csv',[0.3175,0.049,0.6711,0.9624,0.819]]]}
       
    
    mat= []
    for num, element in enumerate(parameters):
#         sum_p = 0
#         sum_m = 0
            
        path_p=param_pm_dict[element][0][0]
        data_p=pd.read_csv(path_p)
        value_p=param_pm_dict[element][0][1]             
        
        path_m=param_pm_dict[element][1][0]
        data_m=pd.read_csv(path_m)
        value_m=param_pm_dict[element][1][1]

        
        s0=np.array([np.load(i) for i in data_p['S0']])
        s12=np.array([np.load(i) for i in data_p['S12']])
        out_p=np.concatenate([np.log(s0[:,1:]),np.log(s12.reshape(s12.shape[0],-1))], axis=1)
        
        s0=np.array([np.load(i) for i in data_m['S0']])
        s12=np.array([np.load(i) for i in data_m['S12']])
        out_m=np.concatenate([np.log(s0[:,1:]),np.log(s12.reshape(s12.shape[0],-1))], axis=1)
        
        metric = PosteriorSamples(
                    num_samples=1000, sample_method='direct', 
                    )

        samps_p = metric(
                        posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
                        x=out_p, theta=value_p
                      )

        samps_m = metric(
                        posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
                        x=out_m, theta=value_m
                      )
        
        mean_p = np.mean(samps_p, axis=(0,1))
        mean_m = np.mean(samps_m, axis=(0,1))
      
        derivative=(mean_p - mean_m)/(value_p[num]-value_m[num])
       
        mat.append(np.array(derivative))
        
    mat=np.array(mat)
        
        
    return mat

def Fisher_contour(fisher_matrix='', label=''):
    fid_cosmology=[0.3175,0.049,0.6711,0.9624,0.834]
    c = ChainConsumer()
    for num,fisher in enumerate(fisher_matrix):
        
        c.add_covariance(fid_cosmology, np.linalg.inv(fisher), parameters=["$\Omega_m$", "$\Omega_b$", "h", "$n_s$", "$\sigma_8$"], name=label[num])
        
    c.configure(usetex=False, serif=False)
    fig = c.plotter.plot(filename="chainconsumer.pdf", figsize="column", truth=fid_cosmology)
    fig.set_size_inches(3 + fig.get_size_inches())  