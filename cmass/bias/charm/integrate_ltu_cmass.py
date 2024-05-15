import sys, os
import numpy as np
import torch
# dev = torch.device("cuda")
if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
import torch.optim as optim
# root_dir = '/mnt/home/spandey/ceph/ltu-cmass/cmass/bias/charm/'
# os.chdir(root_dir)
import sys, os
# sys.path.append(root_dir)
from .combined_models import COMBINED_Model
from .all_models import *
from .utils_data_prep_cosmo import *
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}
cosmo = cosmology.setCosmology('myCosmo', **params)
# get halo mass function:
from colossus.lss import mass_function
from tqdm import tqdm
    
import yaml
import pickle as pk
# autoreload modules
import matplotlib
import matplotlib.pyplot as pl
import os  # noqa

import numpy as np
import logging
import hydra
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf, open_dict
from os.path import join as pjoin
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
# from .tools.halo_models import TruncatedPowerLaw
# from .tools.halo_sampling import (pad_3d, sample_3d,
#                                   sample_velocities_density,
#                                   sample_velocities_kNN,
#                                   sample_velocities_CIC)
# sys.path.append("../../../utils/")
import pathlib
curr_path = pathlib.Path(__file__).parent.resolve()
print(curr_path)
# from utils import get_source_path, timing_decorator, load_params


def parse_config(cfg):
    with open_dict(cfg):
        cfg.nbody.cosmo = load_params(cfg.nbody.lhid, cfg.meta.cosmofile)
    return cfg

class get_model_interface:

    def __init__(self, run_config_name='config_v0.yaml'):
        with open(f"{curr_path}/configs/{run_config_name}","r") as file_object:
            config=yaml.load(file_object,Loader=yaml.SafeLoader)


        config_sims = config['sim_settings']
        ji_array = np.arange(int(config_sims['nsims']))
        num_cosmo_params = int(config_sims['num_cosmo_params'])
        ns_d = config_sims['ns_d']
        nb = config_sims['nb']
        nf = config_sims['nf']
        self.nf = nf
        layers_types = config_sims['layers_types']
        z_inference = config_sims['z_inference']
        nc = 0
        for jl in range(len(layers_types)):
            if layers_types[jl] == 'cnn':
                nc += 1
            elif layers_types[jl] == 'res':
                nc += 2
            else:
                raise ValueError("layer type not supported")
        self.nc = nc

        z_all = config_sims['z_all']
        z_all_FP = config_sims['z_all_FP']
        self.z_all_FP = z_all_FP
        ns_h = config_sims['ns_h']
        self.ns_h = ns_h
        nax_h = ns_h // nb
        self.nax_h = nax_h
        cond_sim = config_sims['cond_sim']

        mass_type = config_sims['mass_type']
        lgMmin = config_sims['lgMmin']
        lgMmax = config_sims['lgMmax']
        self.lgMmin = lgMmin
        self.lgMmax = lgMmax
        stype = config_sims['stype']
        rescale_sub = config_sims['rescale_sub']

        try:
            Nmax = config_sims['Nmax']
        except:
            Nmax = 4

        config_net = config['network_settings']
        hidden_dim_MAF = config_net['hidden_dim_MAF']
        learning_rate = config_net['learning_rate']
        K_M1 = config_net['K_M1']
        B_M1 = config_net['B_M1']
        nflows_M1_NSF = config_net['nflows_M1_NSF']

        K_Mdiff = config_net['K_Mdiff']
        B_Mdiff = config_net['B_Mdiff']
        nflows_Mdiff_NSF = config_net['nflows_Mdiff_NSF']

        base_dist_Ntot = config_net['base_dist_Ntot']
        if base_dist_Ntot == 'None':
            base_dist_Ntot = None
        base_dist_M1 = config_net['base_dist_M1']
        base_dist_Mdiff = config_net['base_dist_Mdiff']
        ngauss_M1 = config_net['ngauss_M1']

        ksize = nf
        nfeature_cnn = config_net['nfeature_cnn']
        nout_cnn = 4 * nfeature_cnn
        if cond_sim == 'fastpm':
            ninp = len(z_all_FP)
        elif cond_sim == 'quijote':
            ninp = len(z_all)
        else:
            raise ValueError("cond_sim not supported")

        num_cond = nout_cnn + ninp + num_cosmo_params


        lgM_array = np.linspace(lgMmin, lgMmax, 1000)
        M_array = 10**lgM_array
        if '200c' in mass_type:
            hmf = mass_function.massFunction(M_array, float(z_inference), mdef = '200c', model = 'tinker08', q_out = 'dndlnM')
        if 'vir' in mass_type:
            hmf = mass_function.massFunction(M_array, float(z_inference), mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')    
        if 'fof' in mass_type:
            hmf = mass_function.massFunction(M_array, float(z_inference), mdef = 'fof', model = 'bhattacharya11', q_out = 'dndlnM')
        lgM_rescaled = rescale_sub + (lgM_array - lgMmin)/(lgMmax-lgMmin)

        int_val = sp.integrate.simps(hmf, lgM_rescaled)
        hmf_pdf = hmf/int_val
        # define the cdf of the halo mass function
        hmf_cdf = np.zeros_like(hmf_pdf)
        for i in range(len(hmf_cdf)):
            hmf_cdf[i] = sp.integrate.simps(hmf_pdf[:i+1], lgM_rescaled[:i+1])

        ndim_diff =  Nmax - 1
        self.ndim_diff = ndim_diff

        # with open("/mnt/home/spandey/ceph/AR_NPE/run_configs/CMASS_test/" + run_config_name,"r") as file_object:
            # config=yaml.load(file_object,Loader=yaml.SafeLoader)

        # config_train = config['train_settings']
        # save_string = config_train['save_string']

        # save_bestfit_model_dir = '/mnt/home/spandey/ceph/AR_NPE/' + \
        #                         'TEST_VARY_COSMO/HRES_SUMGAUSS_subsel_random_MULT_GPU_NO_VELOCITY_ns_' + \
        #                             str(len(ji_array)) + \
        #                             '_cond_sim_' + cond_sim  + '_ns_' + str(ns_h) \
        #                             + '_nc' + str(nc) + '_mass_' + mass_type + \
        #                             '_KM1_' + str(K_M1) + \
        #                             '_stype_' + stype + \
        #                             '_Nmax' + str(Nmax) + save_string


        if 'sigv' in config_net:
            sigv = config_net['sigv']
        else:
            sigv = 0.05
        mu_all = np.arange(Nmax + 1) + 1
        sig_all = sigv * np.ones_like(mu_all)
        ngauss_Nhalo = Nmax + 1


        num_cond_Ntot = num_cond

        model_BinaryMask = SumGaussModel(
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_Ntot,
            ngauss=2,
            mu_all=mu_all[:2],
            sig_all=sig_all[:2],
            base_dist=base_dist_Ntot   
            )

        model_BinaryMask.to(dev)


        model_multiclass = SumGaussModel(
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_Ntot,
            ngauss=ngauss_Nhalo - 1,
            mu_all=mu_all[1:] - 1,
            sig_all=sig_all[1:],
            base_dist=base_dist_Ntot   
            )


        model_multiclass.to(dev)

        num_cond_M1 = num_cond + 1

        model_M1 = NSF_M1_CNNcond(
            K=K_M1,
            B=B_M1,
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_M1,
            nflows=nflows_M1_NSF,
            base_dist=base_dist_M1,
            ngauss=ngauss_M1,
            lgM_rs_tointerp=lgM_rescaled,
            hmf_pdf_tointerp=hmf_pdf,
            hmf_cdf_tointerp=hmf_cdf    
            )

        num_cond_Mdiff = num_cond + 2
        model_Mdiff = NSF_Mdiff_CNNcond(
            dim=ndim_diff,
            K=K_Mdiff,
            B=B_Mdiff,
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_Mdiff,
            nflows=nflows_Mdiff_NSF,
            base_dist=base_dist_Mdiff,
            mu_pos=True
            )

        ndim = ndim_diff + 1
        model = COMBINED_Model(
            None,
            model_Mdiff,
            model_M1,
            model_BinaryMask,
            model_multiclass,
            ndim,
            ksize,
            ns_d,
            ns_h,
            1,    
            ninp,
            nfeature_cnn,
            nout_cnn,
            layers_types=layers_types,
            act='tanh',
            padding='valid',
            sep_Binary_cond=True,
            sep_MultiClass_cond=True,
            sep_M1_cond=True,
            sep_Mdiff_cond=True,
            num_cond_Binary = num_cond_Ntot,
            num_cond_MultiClass = num_cond_Ntot,
            num_cond_M1 = num_cond_M1,
            num_cond_Mdiff = num_cond_Mdiff
            )

        model= torch.nn.DataParallel(model)
        model.to(dev)
        print()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_min = 1e20
        epoch_tot_counter = 0
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=1000, verbose=True, cooldown=1000, min_lr=1e-5)

        jf = 1
        save_bestfit_model_name = f'{curr_path}/trained_models/flow_{jf}'

        print('loading bestfit model')
        bestfit_model = torch.load(save_bestfit_model_name, map_location=device)
        model.load_state_dict(bestfit_model['state_dict'])
        optimizer.load_state_dict(bestfit_model['optimizer'])
        scheduler.load_state_dict(bestfit_model['scheduler'])
        loss_min = bestfit_model['loss_min']
        loss = bestfit_model['loss']
        # lr = bestfit_model['lr']
        epoch_tot_counter = bestfit_model['epoch_tot_counter']
        self.model = model
        print(loss_min, epoch_tot_counter)


    def process_input_density(self, rho_m_zg=None, rho_m_zIC=None, cosmology_array=None, BoxSize=1000, test_LH_id=None, 
                              load_test_LH_dir='/mnt/ceph/users/spandey/Quijote/data_NGP_self_fastpm_LH', 
                              LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt',
                              verbose=False):
        '''
        cosmology_array: array of cosmological parameters, should be in the order [Omega_m, Omega_b, h, n_s, sigma_8]
        rho_m_zg: density field at galaxy redshift (here z=0.5 for CMASS)
        rho_m_zIC: density field of lagrangian densities at high redshift (here z=99 for trained model)
        BoxSize: BoxSize for which to run the model. Here fixed to 1000 by default.
        test_LH_id: If not providing the cosmologies and densities, provide the LH id for testing purposes so that the densities can be loaded
        '''
        n_dim_red = (self.nf - 1) // 2
        n_pad = n_dim_red * self.nc

        if rho_m_zg is None:
            df_zg = pk.load(open(f'{load_test_LH_dir}/{test_LH_id}/density_HR_full_m_res_128_z=0.5_nbatch_8_nfilter_3_ncnn_0.pk','rb'))
            df_test_zg = df_zg['density_cic_unpad_combined']
        else:
            df_test_zg = rho_m_zg
        df_test_pad_zg = np.pad(df_test_zg, n_pad, 'wrap')
        if verbose:
            print(f"loaded density at zg=0.5 with shape {df_test_pad_zg.shape}")

        if rho_m_zIC is None:
            df_zIC = pk.load(open(f'{load_test_LH_dir}/{test_LH_id}/density_HR_full_m_res_128_z=99_nbatch_8_nfilter_3_ncnn_0.pk','rb'))            
            df_test_zIC = df_zIC['density_cic_unpad_combined']
        else:
            df_test_zIC = rho_m_zIC
        df_test_pad_zIC = np.pad(df_test_zIC, n_pad, 'wrap')
        if verbose:
            print(f"loaded density at IC zIC=99 with shape {df_test_pad_zIC.shape}")


        z_REDSHIFT_diff_sig_VALUE = self.z_all_FP[-1]
        VALUE_SIG = float(z_REDSHIFT_diff_sig_VALUE.split('_')[4])
        density_smoothed = gaussian_filter(df_test_pad_zg, sigma=VALUE_SIG)
        df_test_pad_constrast_zg = density_smoothed - df_test_pad_zg

        df_test_all_pad = np.stack([np.log(1 + df_test_pad_zg + 1e-10), np.log(1 + df_test_pad_zIC+ 1e-10), df_test_pad_constrast_zg], axis=0)[None,None,:]


        density_smoothed = gaussian_filter(df_test_zg, sigma=VALUE_SIG)
        df_test_constrast_zg = density_smoothed - df_test_zg

        df_test_all_unpad = np.stack([np.log(1 + df_test_zg + 1e-10), np.log(1 + df_test_zIC + 1e-10), df_test_constrast_zg], axis=0)[None,None,:]

        cond_nsh_test = np.moveaxis(df_test_all_unpad, 2, 5)
        nsims_test = cond_nsh_test.shape[1]
        nax_h_test = cond_nsh_test.shape[2]
        ninp_test = cond_nsh_test.shape[-1]
        cond_tensor_nsh_test = torch.Tensor(np.copy(cond_nsh_test.reshape(1,nsims_test * (nax_h_test ** 3), ninp_test))).to(dev)    

        if cosmology_array is None:
            LH_cosmo_val_all = np.loadtxt(LH_cosmo_val_file)
            cosmology_array = LH_cosmo_val_all[test_LH_id]


        cosmo_val_test = np.tile(cosmology_array, (cond_tensor_nsh_test.shape[1] ,1))[None,:]
        
        df_test_all_pad = torch.tensor(df_test_all_pad, dtype=torch.float32).to(dev)
        df_test_all_unpad = torch.tensor(cond_tensor_nsh_test, dtype=torch.float32).to(dev)
        cosmo_val_test = torch.tensor(cosmo_val_test, dtype=torch.float32).to(dev)

        train_Ntot, train_M1, train_Mdiff = 1, 1, 1
        train_binary, train_multi = 1, 1
        if verbose:
            print(f"Running the model")

        Ntot_samp_test, M1_samp_test, M_diff_samp_test, mask_tensor_M1_samp_test, mask_tensor_Mdiff_samp_test, _ = self.model.module.inverse(
            cond_x=df_test_all_pad,
            cond_x_nsh=df_test_all_unpad,
            cond_cosmo=cosmo_val_test,
            use_truth_Nhalo=1-train_Ntot,
                use_truth_M1=1-train_M1,
                use_truth_Mdiff=1-train_Mdiff, 
            mask_Mdiff_truth=None,
            mask_M1_truth=None,
            Nhalos_truth=None,
            M1_truth=None,
            Mdiff_truth=None,
            train_binary=train_binary,
            train_multi=train_multi,   
            train_M1=train_M1,
            train_Mdiff=train_Mdiff,
            )
        if verbose:
            print(f"Ran the model")


        Ntot_samp_test = Ntot_samp_test[0][:,np.newaxis]
        save_subvol_Nhalo = Ntot_samp_test.reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test)
        save_subvol_M1 = (M1_samp_test[0] * mask_tensor_M1_samp_test[0][:,0]
                            ).cpu().detach().numpy().reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test, 1)
        save_subvol_Mdiff = (M_diff_samp_test[0] * mask_tensor_Mdiff_samp_test[0]
                                ).cpu().detach().numpy().reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test, self.ndim_diff)

        mask_subvol_Mtot1 = mask_tensor_M1_samp_test[0].cpu().detach().numpy().reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test)[...,None]
        mask_subvol_Mtot2 = mask_tensor_Mdiff_samp_test[0].cpu().detach().numpy().reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test, self.ndim_diff)
        mask_subvol_Mtot = np.concatenate([mask_subvol_Mtot1, mask_subvol_Mtot2], axis=-1)

        save_subvol_Mtot = np.zeros((nsims_test, nax_h_test, nax_h_test, nax_h_test, self.ndim_diff + 1))
        # Mmin, Mmax = return_dict_test['Mmin'], return_dict_test['Mmax']
        for jd in range(self.ndim_diff + 1):
            if jd == 0:
                save_subvol_Mtot[..., jd] = (save_subvol_M1[..., 0] + 0.5) * (self.lgMmax - self.lgMmin) + self.lgMmin
            else:
                save_subvol_Mtot[...,
                                jd] = (save_subvol_Mtot[..., jd - 1]) - (save_subvol_Mdiff[..., jd - 1]) * (self.lgMmax - self.lgMmin)


        save_subvol_Mtot *= mask_subvol_Mtot

        Nhalos = save_subvol_Nhalo[0,...]
        M_halos = save_subvol_Mtot[0,...]
                    
        # create the meshgrid
        xall = (np.linspace(0, BoxSize, self.ns_h + 1))
        xarray = 0.5 * (xall[1:] + xall[:-1])
        yarray = np.copy(xarray)
        zarray = np.copy(xarray)
        x_cy, y_cy, z_cy = np.meshgrid(xarray, yarray, zarray, indexing='ij')


        x_h_mock, y_h_mock, z_h_mock, lgM_mock = [], [], [], []
        # Nmax_sel = 3
        k = 0
        for jx in range(self.ns_h):
            for jy in range(self.ns_h):
                for jz in range(self.ns_h):
                        Nh_vox = int(Nhalos[jx, jy, jz])
                        if Nh_vox > 0:
                            x_h_mock.append(x_cy[jx, jy, jz]*np.ones(Nh_vox))
                            y_h_mock.append(y_cy[jx, jy, jz]*np.ones(Nh_vox))
                            z_h_mock.append(z_cy[jx, jy, jz]*np.ones(Nh_vox))
                            
                            lgM_mock.append((M_halos[jx, jy, jz, :Nh_vox]))
                            k += Nh_vox

        # convert to numpy arrays
        x_h_mock = np.concatenate(x_h_mock)
        y_h_mock = np.concatenate(y_h_mock)
        z_h_mock = np.concatenate(z_h_mock)
        pos_h_mock = np.vstack((x_h_mock, y_h_mock, z_h_mock)).T
        lgMass_mock = np.concatenate(lgM_mock)
        # convert to float data type
        pos_h_mock = pos_h_mock.astype('float32')
        lgMass_mock = lgMass_mock.astype('float32')

        return pos_h_mock, lgMass_mock
                        

# @timing_decorator
# @hydra.main(version_base=None, config_path="../conf", config_name="config")
# def main(cfg: DictConfig) -> None:
#     # Filtering for necessary configs
#     cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody'])

#     # Build run config
#     cfg = parse_config(cfg)
#     logging.info(f"Working directory: {os.getcwd()}")
#     logging.info(
#         "Logging directory: " +
#         hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
#     logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

#     run_config_name = 'MULTGPU_cond_fastpm_ns128_run_Ntot_M1_Mdiff_subselrand_gumbel.yaml'
#     charm_interface = get_model_interface(run_config_name)    

#     test_LH_id = 0
#     pos_h_mock, lgMass_mock = charm_interface.process_input_density(test_LH_id)

#     # Setup
#     # pmconf, pmcosmo = configure_pmwd(cfg)

#     # # Get ICs
#     # wn = get_ICs(cfg)

#     # # Run
#     # rho, pos, vel = run_density(wn, pmconf, pmcosmo, cfg)

#     # # Calculate velocity field
#     # fvel = None
#     # if cfg.nbody.save_velocities:
#     #     fvel = vfield_CIC(pos, vel, cfg)
#     #     # convert from comoving -> peculiar velocities
#     #     fvel *= (1 + cfg.nbody.zf)

#     # # Save
#     # outdir = get_source_path(cfg, "pmwd", check=False)
#     # save_nbody(outdir, rho, fvel, pos, vel,
#     #            cfg.nbody.save_particles, cfg.nbody.save_velocities)
#     # with open(pjoin(outdir, 'config.yaml'), 'w') as f:
#     #     OmegaConf.save(cfg, f)
#     logging.info("Done!")


# if __name__ == '__main__':
#     main()
