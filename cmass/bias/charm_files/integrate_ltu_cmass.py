"""
Module to integrate ltu-cmass with CHARM halo emulators.

Much of this code is not production-quality, but is kept in its original form
in case of further development.

Note from Matt, the only things I changed in this code were:
 * applying autopep formatting
 * allowing to specify the padded inputs in `process_input_density`
 * commenting out unnecessary imports (to avoid further dependencies)
TODO: Clean up and simplify
"""

import pathlib
from omegaconf import DictConfig, OmegaConf, open_dict
from copy import deepcopy
import hydra
import logging
import pickle as pk
import yaml
from tqdm import tqdm
from colossus.lss import mass_function
import sys
import os
import numpy as np
import scipy as sp
import torch
if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
import torch.optim as optim
import sys
import os
import pickle as pk
from scipy.interpolate import RegularGridInterpolator

from charm.combined_models import *
from charm.all_models import *
from charm.utils_data_prep_cosmo_vel import *
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 67.11, 'Om0': 0.3175,
          'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}
cosmo = cosmology.setCosmology('myCosmo', **params)
# get halo mass function:

# autoreload modules
import os  # noqa

curr_path = pathlib.Path(__file__).parent.resolve()
# print(curr_path)


def parse_config(cfg):
    with open_dict(cfg):
        cfg.nbody.cosmo = load_params(cfg.nbody.lhid, cfg.meta.cosmofile)
    return cfg


class get_model_interface:

    def __init__(self, run_config_massNtot_name='train_massNtot_config.yaml', run_config_vel_name='train_vel_config.yaml'):
        with open(f"{curr_path}/configs/{run_config_massNtot_name}", "r") as file_object:
            config = yaml.load(file_object, Loader=yaml.SafeLoader)

        config_sims = config['sim_settings']
        ji_array = np.arange(int(config_sims['nsims']))
        nsubvol_per_ji = int(config_sims['nsubvol_per_ji'])
        nsubvol_fid = int(config_sims['nsubvol_fid'])
        subsel_criteria = config_sims['subsel_criteria']
        num_cosmo_params = int(config_sims['num_cosmo_params'])
        ns_d = config_sims['ns_d']
        nb = config_sims['nb']
        nax_d =  ns_d // nb
        self.nf = config_sims['nf']
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
        self.z_all_FP = config_sims['z_all_FP']
        self.ns_h = config_sims['ns_h']
        self.nax_h = self.ns_h // nb
        cond_sim = config_sims['cond_sim']

        nsims_per_batch = config_sims['nsims_per_batch']
        nbatches_train = config_sims['nbatches_train']

        mass_type = config_sims['mass_type']
        self.lgMmin = config_sims['lgMmin']
        self.lgMmax = config_sims['lgMmax']
        stype = config_sims['stype']
        rescale_sub = config_sims['rescale_sub']
        lgMmincutstr = config_sims['lgMmincutstr']
        is_HR = config_sims['is_HR']

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

        changelr = config_net['changelr']
        ksize = self.nf
        nfeature_cnn = config_net['nfeature_cnn']
        nout_cnn = 4 * nfeature_cnn
        if cond_sim == 'fastpm':
            if any('v' in str(string) for string in self.z_all_FP):
                ninp = len(self.z_all_FP) + 2
            else:
                ninp = len(self.z_all_FP)

        elif cond_sim == 'quijote':
            ninp = len(z_all)
        else:
            raise ValueError("cond_sim not supported")

        num_cond = nout_cnn + ninp + num_cosmo_params


        self.ndim_diff = Nmax - 1

        lgM_array = np.linspace(self.lgMmin, self.lgMmax, 1000)
        M_array = 10**lgM_array
        if '200c' in mass_type:
            hmf = mass_function.massFunction(M_array, float(z_inference), mdef = '200c', model = 'tinker08', q_out = 'dndlnM')
        if 'vir' in mass_type:
            hmf = mass_function.massFunction(M_array, float(z_inference), mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')    
        if 'fof' in mass_type:
            hmf = mass_function.massFunction(M_array, float(z_inference), mdef = 'fof', model = 'bhattacharya11', q_out = 'dndlnM')
        lgM_rescaled = rescale_sub + (lgM_array - self.lgMmin)/(self.lgMmax-self.lgMmin)

        int_val = sp.integrate.simps(hmf, lgM_rescaled)
        hmf_pdf = hmf/int_val
        # define the cdf of the halo mass function
        hmf_cdf = np.zeros_like(hmf_pdf)
        for i in range(len(hmf_cdf)):
            hmf_cdf[i] = sp.integrate.simps(hmf_pdf[:i+1], lgM_rescaled[:i+1])

        if 'sigv' in config:
            sigv = config['sigv']
        else:
            sigv = 0.05
        num_cond_Ntot = num_cond
        mu_all = np.arange(Nmax + 1) + 1
        sig_all = sigv * np.ones_like(mu_all)
        ngauss_Nhalo = Nmax + 1

        model_BinaryMask = SumGaussModel(
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_Ntot,
            ngauss=2,
            mu_all=mu_all[:2],
            sig_all=sig_all[:2],
            base_dist=base_dist_Ntot,
            device=dev
            )


        model_multiclass = SumGaussModel(
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_Ntot,
            ngauss=ngauss_Nhalo - 1,
            mu_all=mu_all[1:] - 1,
            sig_all=sig_all[1:],
            base_dist=base_dist_Ntot,
            device=dev
            )

        num_cond_M1 = num_cond + 1

        model_M1 = NSF_1var_CNNcond(
            K=K_M1,
            B=B_M1,
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_M1,
            nflows=nflows_M1_NSF,
            base_dist=base_dist_M1,
            ngauss=ngauss_M1,
            lgM_rs_tointerp=lgM_rescaled,
            hmf_pdf_tointerp=hmf_pdf,
            hmf_cdf_tointerp=hmf_cdf,
            device=dev 
            )

        num_cond_Mdiff = num_cond + 2
        model_Mdiff = NSF_Autoreg_CNNcond(
            dim=self.ndim_diff,
            K=K_Mdiff,
            B=B_Mdiff,
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_Mdiff,
            nflows=nflows_Mdiff_NSF,
            base_dist=base_dist_Mdiff,
            mu_pos=True
            )


        ndim = self.ndim_diff + 1
        model = COMBINED_Model(
            None,
            model_Mdiff,
            model_M1,
            model_BinaryMask,
            model_multiclass,
            ndim,
            ksize,
            ns_d,
            self.ns_h,
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
            ).to(dev)


        model = torch.nn.DataParallel(model)

        save_bestfit_model_name = f'{curr_path}/trained_models/charm_model_massNtot_bestfit_v2.pth'

        checkpoint = torch.load(save_bestfit_model_name, map_location=dev)
        # print(iter)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model_mass = model


        with open(f"{curr_path}/configs/{run_config_vel_name}", "r") as file_object:
            config_vel = yaml.load(file_object, Loader=yaml.SafeLoader)

        config_net_vel = config_vel['network_settings']
        K_vel = config_net_vel['K_vel']
        B_vel = config_net_vel['B_vel']
        nflows_vel_NSF = config_net_vel['nflows_vel_NSF']
        self.cond_Mass_for_vel = config_net_vel['cond_Mass_for_vel']
        base_dist_vel = config_net_vel['base_dist_vel']

        ndim_mass = Nmax
        ndim_vel = 3*Nmax
            
        if self.cond_Mass_for_vel:
            num_cond_vel = num_cond + ndim_mass
        else:
            num_cond_vel = num_cond
            
        model_vel = NSF_Autoreg_CNNcond(
            dim=ndim_vel,
            K=K_vel,
            B=B_vel,
            hidden_dim=hidden_dim_MAF,
            num_cond=num_cond_vel,
            nflows=nflows_vel_NSF,
            base_dist=base_dist_vel,
            mu_pos=False
            )


        model_vel = COMBINED_Model_vel_only(
            None,
            model_vel,
            ndim_vel,
            ksize,
            ns_d,
            self.ns_h,
            1,
            ninp,
            nfeature_cnn,
            nout_cnn,
            layers_types=layers_types,
            act='tanh',
            padding='valid',
            ).to(dev)

        model_vel = torch.nn.DataParallel(model_vel)

        save_bestfit_model_name = f'{curr_path}/trained_models/charm_model_vel_bestfit_v2.pth'

        checkpoint = torch.load(save_bestfit_model_name, map_location=dev)
        model_vel.load_state_dict(checkpoint['state_dict'])
        model_vel.eval()
        self.model_vel = model_vel






    def process_input_density(self, rho_m_zg=None, rho_m_vel_zg=None,
                              rho_m_pad_zg=None, rho_m_vel_pad_zg=None,
                              cosmology_array=None, BoxSize=1000, test_LH_id=None,
                              load_test_LH_dir='/mnt/ceph/users/spandey/Quijote/data_NGP_self_fastpm_LH',
                              LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt',
                              verbose=False, fac_norm_vel=1.):
        '''
        cosmology_array: array of cosmological parameters, should be in the order [Omega_m, Omega_b, h, n_s, sigma_8]
        rho_m_zg: density field at galaxy redshift (here z=0.5 for CMASS)
        rho_m_zIC: density field of lagrangian densities at high redshift (here z=99 for trained model)
        BoxSize: BoxSize for which to run the model. Here fixed to 1000 by default.
        test_LH_id: If not providing the cosmologies and densities, provide the LH id for testing purposes so that the densities can be loaded
        '''
        n_dim_red = (self.nf - 1) // 2
        n_pad = n_dim_red * self.nc

        z_REDSHIFT = float(self.z_all_FP[-1].split('_')[1])
        if z_REDSHIFT == 0.0:
            z_REDSHIFT = 0


        if rho_m_zg is None:
            # load the z=0.5 density, if unspecified
            df_zg = pk.load(open(
                f'{load_test_LH_dir}/{test_LH_id}/density_HR_full_m_res_128_z={z_REDSHIFT}_nbatch_8_nfilter_3_ncnn_0.pk', 'rb'))
            rho_m_zg = df_zg['density_cic_unpad_combined']

        if rho_m_pad_zg is None:
            rho_m_pad_zg = np.pad(rho_m_zg, n_pad, 'wrap')


        if rho_m_vel_zg is None:
            df_load = pk.load(open(
                f'{load_test_LH_dir}/{test_LH_id}/velocity_HR_full_m_res_128_z={z_REDSHIFT}_nbatch_8_nfilter_3_ncnn_0.pk', 'rb')
                )

            rho_m_vel_zg = df_load['velocity_cic_unpad_combined']/fac_norm_vel
            
        if rho_m_vel_pad_zg is None:
            rho_m_vel_pad_zg = np.stack([np.pad(rho_m_vel_zg[j,...], n_pad, 'wrap') for j in range(3)], axis=0)


        if verbose:
            print(
                f"loaded density at zg=0.5 with shape {rho_m_vel_pad_zg.shape}")

        df_test_all_pad = np.concatenate([np.log(1 + rho_m_pad_zg + 1e-10)[None,...], rho_m_vel_pad_zg], axis=0)[None, None,:]
        df_test_all_unpad = np.concatenate([np.log(1 + rho_m_zg + 1e-10)[None,...], rho_m_vel_zg], axis=0)[None, None,:]        

        cond_nsh_test = np.moveaxis(df_test_all_unpad, 2, 5)
        nsims_test = cond_nsh_test.shape[1]
        nax_h_test = cond_nsh_test.shape[2]
        ninp_test = cond_nsh_test.shape[-1]
        cond_tensor_nsh_test = torch.Tensor(np.copy(cond_nsh_test.reshape(
            1, nsims_test * (nax_h_test ** 3), ninp_test))).to(dev)

        if cosmology_array is None:
            LH_cosmo_val_all = np.loadtxt(LH_cosmo_val_file)
            cosmology_array = LH_cosmo_val_all[test_LH_id]

        cosmo_val_test = np.tile(
            cosmology_array, (cond_tensor_nsh_test.shape[1], 1))[None, :]

        df_test_all_pad = torch.tensor(
            df_test_all_pad, dtype=torch.float32).to(dev)
        df_test_all_unpad = torch.tensor(
            cond_tensor_nsh_test, dtype=torch.float32).to(dev)
        cosmo_val_test = torch.tensor(
            cosmo_val_test, dtype=torch.float32).to(dev)

        train_Ntot, train_M1, train_Mdiff = 1, 1, 1
        train_binary, train_multi = 1, 1
        if verbose:
            print(f"Running the model")

        # run the model
        Ntot_samp_test, M1_samp_test, M_diff_samp_test, mask_tensor_M1_samp_test, mask_tensor_Mdiff_samp_test, _ = self.model_mass.module.inverse(
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
            print("Ran the model")

        # reshape outputs
        Ntot_samp_test = Ntot_samp_test[0][:, np.newaxis]
        save_subvol_Nhalo = Ntot_samp_test.reshape(
            nsims_test, nax_h_test, nax_h_test, nax_h_test)
        save_subvol_M1 = (M1_samp_test[0] * mask_tensor_M1_samp_test[0][:, 0]
                          ).cpu().detach().numpy().reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test, 1)
        save_subvol_Mdiff = (M_diff_samp_test[0] * mask_tensor_Mdiff_samp_test[0]
                             ).cpu().detach().numpy().reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test, self.ndim_diff)

        mask_subvol_Mtot1 = mask_tensor_M1_samp_test[0].cpu().detach().numpy().reshape(
            nsims_test, nax_h_test, nax_h_test, nax_h_test)[..., None]
        mask_subvol_Mtot2 = mask_tensor_Mdiff_samp_test[0].cpu().detach().numpy(
        ).reshape(nsims_test, nax_h_test, nax_h_test, nax_h_test, self.ndim_diff)
        mask_subvol_Mtot = np.concatenate(
            [mask_subvol_Mtot1, mask_subvol_Mtot2], axis=-1)

        # compute the mass of halos from output
        save_subvol_Mtot = np.zeros(
            (nsims_test, nax_h_test, nax_h_test, nax_h_test, self.ndim_diff + 1))
        # Mmin, Mmax = return_dict_test['Mmin'], return_dict_test['Mmax']
        for jd in range(self.ndim_diff + 1):
            if jd == 0:
                save_subvol_Mtot[..., jd] = (
                    save_subvol_M1[..., 0] + 0.5) * (self.lgMmax - self.lgMmin) + self.lgMmin
            else:
                save_subvol_Mtot[...,
                                 jd] = (save_subvol_Mtot[..., jd - 1]) - (save_subvol_Mdiff[..., jd - 1]) * (self.lgMmax - self.lgMmin)

        save_subvol_Mtot *= mask_subvol_Mtot

        Nhalos = save_subvol_Nhalo[0, ...]  # histogram of halos in each voxel
        M_halos = save_subvol_Mtot[0, ...]  # mass of halos in each voxel

        # create the meshgrid
        xall = (np.linspace(0, BoxSize, self.ns_h + 1))
        xarray = 0.5 * (xall[1:] + xall[:-1])
        yarray = np.copy(xarray)
        zarray = np.copy(xarray)
        x_cy, y_cy, z_cy = np.meshgrid(xarray, yarray, zarray, indexing='ij')

        # record discrete halo positions and masses
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


        Nhalos_truth_recomb_tensor = torch.Tensor(Ntot_samp_test[None,...]).cuda(dev)
        if self.cond_Mass_for_vel:
            Mhalos_truth_recomb_tensor = torch.Tensor(M_halos_sort_norm_condvel[None,...]).cuda(dev)
        else:
            Mhalos_truth_recomb_tensor = None

        vel_samp_out = self.model_vel.module.inverse(cond_x=df_test_all_pad,
                                    cond_x_nsh=df_test_all_unpad,
                                    cond_cosmo=cosmo_val_test,
                                    mask_vel_truth=None,
                                    Nhalos_truth=Nhalos_truth_recomb_tensor,
                                    Mhalos_truth=Mhalos_truth_recomb_tensor,
                                    # Mhalos_truth=None,                            
                                    vel_truth=None)

        vx_mesh_load = (1000./fac_norm_vel)*rho_m_vel_zg[0,...]
        vy_mesh_load = (1000./fac_norm_vel)*rho_m_vel_zg[1,...]
        vz_mesh_load = (1000./fac_norm_vel)*rho_m_vel_zg[2,...]

        vx_all_3D_interp_l = RegularGridInterpolator((xarray, yarray, zarray), vx_mesh_load, bounds_error=False, fill_value=None)
        vy_all_3D_interp_l = RegularGridInterpolator((xarray, yarray, zarray), vy_mesh_load, bounds_error=False, fill_value=None)
        vz_all_3D_interp_l = RegularGridInterpolator((xarray, yarray, zarray), vz_mesh_load, bounds_error=False, fill_value=None)

        vx_eval_interp_l = vx_all_3D_interp_l(pos_h_mock)
        vy_eval_interp_l = vy_all_3D_interp_l(pos_h_mock)
        vz_eval_interp_l = vz_all_3D_interp_l(pos_h_mock)


        vmax = 1000
        vmin = -1000
        v_halos_diff_recomb = np.reshape(vel_samp_out, (1, 128, 128, 128, (self.ndim_diff + 1)*3))[0,...]
        v_halos_diff_recomb = np.reshape(v_halos_diff_recomb, (128, 128, 128, (self.ndim_diff + 1), 3))

        vx_diff_mock = []
        vy_diff_mock = []
        vz_diff_mock = []

        Nhalos_pred_recomb = Ntot_samp_test[...,0].reshape(self.ns_h, self.ns_h, self.ns_h)
        for jx in range(self.ns_h):
            for jy in range(self.ns_h):
                for jz in range(self.ns_h):
                        Nh_vox = int(Nhalos_pred_recomb[jx, jy, jz])
                        if Nh_vox > 0:
                            vx_diff_mock.append(((v_halos_diff_recomb[jx, jy, jz, :Nh_vox, 0])*((vmax - vmin)) + vmin))

                            vy_diff_mock.append(((v_halos_diff_recomb[jx, jy, jz, :Nh_vox, 1])*((vmax - vmin)) + vmin))

                            vz_diff_mock.append(((v_halos_diff_recomb[jx, jy, jz, :Nh_vox, 2])*((vmax - vmin)) + vmin))

        vx_total_mock = vx_eval_interp_l - np.concatenate(vx_diff_mock)

        vy_total_mock = vy_eval_interp_l - np.concatenate(vy_diff_mock)

        vz_total_mock = vz_eval_interp_l - np.concatenate(vz_diff_mock)



        vel_h_mock = np.vstack((vx_total_mock, vy_total_mock, vz_total_mock)).T
        vel_h_mock = vel_h_mock.astype('float32')

        return pos_h_mock, lgMass_mock, vel_h_mock



# if __name__ == '__main__':
#     model_interface = get_model_interface()
#     pos_h_mock, lgMass_mock, vel_h_mock = model_interface.process_input_density(test_LH_id = 1, verbose=True)
#     import pdb; pdb.set_trace()
