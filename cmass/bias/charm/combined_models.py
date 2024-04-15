import torch
import torch.nn as nn
import numpy as np
from cnn_3d_stack import CNN3D_stackout
# check if cuda is available:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self.network(x)


class COMBINED_Model(nn.Module):
    """
    Combined model for the AR_NPE.
    """

    def __init__(
        self,
        priors_all,
        Mdiff_model,
        M1_model,
        BinaryMask_model,
        MultiClass_model,
        ndim,
        ksize,
        nside_in,
        nside_out,
        nbatch,
        ninp,
        nfeature,
        nout,
        layers_types=['cnn', 'res', 'res', 'res'],
        act='tanh',
        padding='valid',
        sep_Binary_cond=False,
        sep_MultiClass_cond=False,
        sep_M1_cond=False,
        sep_Mdiff_cond=False,        
        num_cond_Binary=None,
        num_cond_MultiClass=None,
        num_cond_M1=None,
        num_cond_Mdiff=None,
        M1reg_model=None,
        ):
        super().__init__()
        self.priors_all = priors_all
        self.M1_model = M1_model
        self.BinaryMask_model = BinaryMask_model
        self.MultiClass_model = MultiClass_model
        self.Mdiff_model = Mdiff_model
        self.M1reg_model = M1reg_model
        self.nbatch = nbatch
        self.nout = nout
        self.ninp = ninp

        self.conv_layers = CNN3D_stackout(
            ksize,
            nside_in,
            nside_out,
            nbatch,
            ninp,
            nfeature,
            nout,
            layers_types=layers_types,
            act=act,
            padding=padding
            )
        self.ndim = ndim
        self.sep_Binary_cond = sep_Binary_cond
        self.sep_MultiClass_cond = sep_MultiClass_cond
        self.sep_M1_cond = sep_M1_cond
        self.sep_Mdiff_cond = sep_Mdiff_cond

        if self.sep_Binary_cond:
            if num_cond_Binary is None:
                num_cond_Binary = nout + ninp
            self.cond_Binary_layer = FCNN(num_cond_Binary, num_cond_Binary, num_cond_Binary)

        if self.sep_MultiClass_cond:
            if num_cond_MultiClass is None:
                num_cond_MultiClass = nout + ninp
            self.cond_MultiClass_layer = FCNN(num_cond_MultiClass, num_cond_MultiClass, num_cond_MultiClass)

        if self.sep_M1_cond:
            if num_cond_M1 is None:
                num_cond_M1 = nout + ninp + 1
            # self.cond_M1_layer = FCNN(nout + ninp + 1, nout + ninp + 1, nout + ninp + 1)
            self.cond_M1_layer = FCNN(num_cond_M1, num_cond_M1, num_cond_M1)            
        if self.sep_Mdiff_cond:
            if num_cond_Mdiff is None:
                num_cond_Mdiff = nout + ninp + 2
            self.cond_Mdiff_layer = FCNN(num_cond_Mdiff, num_cond_Mdiff, num_cond_Mdiff)

    def forward(
        self,
        x_Mdiff,
        x_M1,
        x_binary_mask,
        x_multiclass,
        cond_x=None,
        cond_x_nsh=None,
        cond_cosmo=None,
        indsel_Nhalo_gt1=None,
        mask_Ntot_all=None,
        mask_Mdiff_truth_all=None,
        mask_M1_truth_all=None,
        Nhalos_truth_all=None,
        use_Ntot_samples=False,
        use_M1_samples=False,
        reg_M1=False,
        train_binary=False,
        train_multi=False,        
        train_M1=False,
        train_Mdiff=False,
        x_Mdiff_FP=None,
        x_M1_FP=None,
        x_Ntot_FP=None,
        Nhalos_truth_all_FP=None,
        mask_Mdiff_truth_all_FP=None,
        mask_M1_truth_all_FP=None,
        L2norm_M1hist=False,
        L2norm_Ntothist=False,
        ):
        
        nbatches = cond_x.shape[0]
        loss_binarymask = torch.zeros(1, device=device)
        loss_multiclass = torch.zeros(1, device=device)        
        loss_M1 = torch.zeros(1, device=device)
        loss_M1reg = torch.zeros(1, device=device)
        loss_Mdiff = torch.zeros(1, device=device)
        M1_L2_loss_tot = torch.zeros(1, device=device)
        Ntot_L2_loss_tot = torch.zeros(1, device=device)
        for jb in range(nbatches):
            cond_out = self.conv_layers(cond_x[jb])
            cond_out = torch.cat((cond_out, cond_x_nsh[jb]), dim=1)
            if cond_cosmo is not None:
                cond_out = torch.cat((cond_out, cond_cosmo[jb]), dim=1)

            if train_binary:
                if mask_Ntot_all is not None:
                    mask_sel_Ntot = mask_Ntot_all[jb].to(device)            
                else:
                    mask_sel_Ntot = torch.arange(cond_out.shape[0]).to(device)
                if self.sep_Binary_cond:
                    cond_out_Binary = self.cond_Binary_layer(cond_out[mask_sel_Ntot])
                else:
                    cond_out_Binary = cond_out[mask_sel_Ntot]

                if jb == 0:
                    loss_binarymask = torch.mean(self.BinaryMask_model.forward(x_binary_mask[jb][mask_sel_Ntot], cond_out_Binary))
                else:
                    loss_binarymask += torch.mean(self.BinaryMask_model.forward(x_binary_mask[jb][mask_sel_Ntot], cond_out_Binary))

            #     Nhalos_truth = Nhalos_truth_all[jb].to(device)
            #     if reg_M1 or train_M1:   
            #         mask_M1_truth = mask_M1_truth_all[jb].to(device)
            #     if train_Mdiff:
            #         mask_Mdiff_truth = mask_Mdiff_truth_all[jb].to(device)
            # else:
            #     Nhalos_truth = Nhalos_truth_all[jb].to(device)
            #     if reg_M1 or train_M1:                
            #         mask_M1_truth = mask_M1_truth_all[jb].to(device)
            #     if train_Mdiff:
            #         mask_Mdiff_truth = mask_Mdiff_truth_all[jb].to(device)

            # mask_sel_MultiClass = Nhalos_truth_all[jb][:,0].nonzero().squeeze()


            if train_multi:
                mask_sel_MultiClass = mask_M1_truth_all[jb].to(device)            
                if self.sep_MultiClass_cond:
                    cond_out_MultiClass = self.cond_MultiClass_layer(cond_out[mask_sel_MultiClass])
                else:
                    cond_out_MultiClass = cond_out[mask_sel_MultiClass]
                # import pdb; pdb.set_trace()
                if jb == 0:
                    loss_multiclass = torch.mean(self.MultiClass_model.forward(x_multiclass[jb][mask_sel_MultiClass], cond_out_MultiClass))
                else:
                    loss_multiclass += torch.mean(self.MultiClass_model.forward(x_multiclass[jb][mask_sel_MultiClass], cond_out_MultiClass))                



            if train_M1:
                Nhalos_truth = Nhalos_truth_all[jb].to(device)
                mask_sel_M1 = mask_M1_truth_all[jb].to(device)                      
                cond_inp_M1 = torch.cat([Nhalos_truth, cond_out], dim=1)
                if x_M1_FP is not None:
                    cond_inp_M1 = torch.cat([cond_inp_M1, x_M1_FP[jb]], dim=1)
                if mask_M1_truth_all_FP is not None:
                    cond_inp_M1 = torch.cat([cond_inp_M1, mask_M1_truth_all_FP[jb][:,None]], dim=1)
                # import pdb; pdb.set_trace()
                if self.sep_M1_cond:
                    cond_inp_M1 = self.cond_M1_layer(cond_inp_M1[mask_sel_M1])
                else:
                    cond_inp_M1 = cond_inp_M1[mask_sel_M1]
            
                if jb == 0:
                    loss_M1 = torch.mean(-(self.M1_model.forward(x_M1[jb][mask_sel_M1], cond_inp_M1)))
                else:
                    loss_M1 += torch.mean(-(self.M1_model.forward(x_M1[jb][mask_sel_M1], cond_inp_M1)))
                M1_truth = x_M1[jb]
            else:
                if train_Mdiff:
                    M1_truth = x_M1[jb]
                    Nhalos_truth = Nhalos_truth_all[jb].to(device)

            if train_Mdiff:
                # mask_sel_Mdiff = torch.where(Nhalos_truth_all[jb][:,0] > 1)
                # mask_sel_tensor1 = Nhalos_truth_all[jb][:,0] - 1.0
                # mask_sel_tensor = torch.nn.ReLU(mask_sel_tensor1)
                # mask_sel_Mdiff = mask_sel_tensor.nonzero().squeeze()


                Nhalos_truth = Nhalos_truth_all[jb].to(device)
                # indsel_Nhalos = 
                mask_sel_Mdiff = indsel_Nhalo_gt1[jb].to(device)
                # mask_Mdiff_truth = mask_Mdiff_truth_all[jb].to('cuda')
                # import pdb; pdb.set_trace()
                cond_inp_Mdiff = torch.cat([Nhalos_truth, M1_truth, cond_out], dim=1)
                if self.sep_Mdiff_cond:
                    cond_inp_Mdiff = self.cond_Mdiff_layer(cond_inp_Mdiff[mask_sel_Mdiff])
                else:
                    cond_inp_Mdiff = cond_inp_Mdiff[mask_sel_Mdiff]
                # import pdb; pdb.set_trace()
                if jb == 0:
                    loss_Mdiff = torch.mean(-self.Mdiff_model.forward(x_Mdiff[jb][mask_sel_Mdiff], cond_inp_Mdiff, mask_Mdiff_truth_all[jb][mask_sel_Mdiff]))
                else:
                    loss_Mdiff += torch.mean(-self.Mdiff_model.forward(x_Mdiff[jb][mask_sel_Mdiff], cond_inp_Mdiff, mask_Mdiff_truth_all[jb][mask_sel_Mdiff]))
        loss = (loss_binarymask + loss_multiclass + loss_M1 + loss_Mdiff)
        return loss

    def inverse(
        self,
        cond_x=None,
        cond_x_nsh=None,
        cond_cosmo=None,
        use_truth_Nhalo=False,
        use_truth_M1=False,
        use_truth_Mdiff=False,
        mask_Mdiff_truth=None,
        mask_M1_truth=None,
        Nhalos_truth=None,
        M1_truth=None,
        Mdiff_truth=None,
        train_binary=False,
        train_multi=False,        
        train_M1=False,
        train_Mdiff=False,
        reg_M1=False,
        x_Mdiff_FP=None,
        x_M1_FP=None,
        x_Ntot_FP=None,
        Nhalos_truth_all_FP=None,
        mask_Mdiff_truth_all_FP=None,
        mask_M1_truth_all_FP=None        
        ):
        nbatches = cond_x.shape[0]
        Ntot_samp_out, M1_samp_out, M_diff_samp_out = [], [], []
        mask_tensor_M1_samp_out, mask_tensor_Mdiff_samp_out = [], []
        cond_inp_M1_out = []
        for jb in range(nbatches):
            cond_out = self.conv_layers(cond_x[jb])
            cond_out = torch.cat((cond_out, cond_x_nsh[jb]), dim=1)
            if cond_cosmo is not None:
                cond_out = torch.cat((cond_out, cond_cosmo[jb]), dim=1)

            if self.sep_Binary_cond:
                cond_out_Binary = self.cond_Binary_layer(cond_out)
            else:
                cond_out_Binary = cond_out

            Ntot_samp = torch.zeros(cond_out.shape[0])
            # print(cond_out_Ntot.shape)
            if train_binary:
                # binary_mask_tensor = self.BinaryMask_model.inverse(cond_out_Binary) - 1
                # binary_mask_tensor_out = torch.zeros_like(binary_mask_tensor)
                # ind_gt_0p5 = torch.where(binary_mask_tensor >= 0.5)
                # binary_mask_tensor_out[ind_gt_0p5] = 1
                binary_mask_tensor = self.BinaryMask_model.inverse(cond_out_Binary) - 1
                binary_mask_tensor_out = torch.zeros_like(binary_mask_tensor)
                ind_gt_0p5 = torch.where(binary_mask_tensor >= 0.5)
                binary_mask_tensor_out[ind_gt_0p5] = 1

                # Ntot_samp = np.maximum(np.round(Ntot_samp_tensor.cpu().detach().numpy()) - 1, 0).astype(int)
            else:
                # Ntot_samp = torch.Tensor(Nhalos_truth)
                # Ntot_samp = np.maximum(np.round(Nhalos_truth[jb,...].cpu().detach().numpy()) - 1, 0).astype(int)
                binary_mask_tensor_out = torch.zeros_like(Nhalos_truth[jb,...])
                ind_gt = torch.where(Nhalos_truth[jb,...] > 0)
                binary_mask_tensor_out[ind_gt] = 1
                

            # mask_sel_MultiClass = torch.where(binary_mask_tensor_out > 0)
            mask_sel_MultiClass = torch.where(binary_mask_tensor_out > 0)[0]
            if self.sep_MultiClass_cond:
                cond_out_MultiClass = self.cond_MultiClass_layer(cond_out[mask_sel_MultiClass])
            else:
                cond_out_MultiClass = cond_out[mask_sel_MultiClass]

            if train_multi:
                multi_samp_tensor = (self.MultiClass_model.inverse(cond_out_MultiClass))
                multi_samp_tensor = torch.maximum(torch.round(multi_samp_tensor), torch.Tensor([0]).to(device)).detach().cpu()
                # import pdb; pdb.set_trace()
                Ntot_samp[mask_sel_MultiClass] = multi_samp_tensor
            else:
                Ntot_samp = torch.Tensor(Nhalos_truth)

            Ntot_samp = Ntot_samp.cpu().detach().numpy()
                # Ntot_samp = Nhalos_truth.cpu().detach().numpy()
            Ntot_samp_out.append(Ntot_samp)
            # nvox_batch = 64 // 8
            nvox_batch = self.nout // self.nbatch
            # import pdb; pdb.set_trace()
            nvox_batch = 1
            Ntot_samp_rs = Ntot_samp.reshape(-1, nvox_batch**3)
            # print(cond_out_Ntot.shape, Ntot_samp.shape, Ntot_samp_rs.shape)
            # print(Ntot_samp_rs.shape, nvox_batch)
            # Ntot_samp = np.maximum(np.round(self.Ntot_model.inverse(cond_out_Ntot).detach().numpy() - 1), 0).astype(int)
            nsim, nvox = Ntot_samp_rs.shape[0], Ntot_samp_rs.shape[1]
            mask_samp_all = np.zeros((nsim, nvox, self.ndim))
            idx = np.arange(self.ndim)[None, None, :]
            mask_samp_all[np.arange(nsim)[:, None, None],
                          np.arange(nvox)[None, :, None], idx] = (idx < Ntot_samp_rs[..., None])

            Ntot_samp_diff = Ntot_samp_rs - 1
            Ntot_samp_diff[Ntot_samp_diff < 0] = 0
            mask_samp_M_diff = np.zeros((nsim, nvox, self.ndim - 1))
            idx = np.arange(self.ndim - 1)[None, None, :]
            mask_samp_M_diff[np.arange(nsim)[:, None, None],
                             np.arange(nvox)[None, :, None], idx] = (idx < Ntot_samp_diff[..., None])

            mask_samp_M1 = mask_samp_all[:, :, 0]
            # print(mask_samp_all.shape)
            mask_samp_M_diff = mask_samp_M_diff.reshape(nsim * nvox, self.ndim - 1)
            mask_samp_M1 = mask_samp_M1.reshape(nsim * nvox, 1)

            # mask_samp_M1 = mask_samp_M1.reshape(nbatches, nsim * nvox // nbatches, 1)[jb, ...]
            # mask_samp_M_diff = mask_samp_M_diff.reshape(nbatches, nsim * nvox // nbatches, self.ndim - 1)[jb, ...]
            # import pdb; pdb.set_trace()

            if use_truth_M1:
                mask_tensor_M1_samp = (mask_M1_truth)[jb, ...][None, ...].T
                mask_tensor_M1_samp = mask_tensor_M1_samp.float().cuda()

            else:
                # mask_tensor_M1_samp = torch.Tensor(np.array([mask_samp_all[:, 0]]).T)
                mask_tensor_M1_samp = torch.from_numpy(mask_samp_M1)
                mask_tensor_M1_samp = mask_tensor_M1_samp.float().cuda()
            mask_tensor_M1_samp_out.append(mask_tensor_M1_samp)
            # print(mask_tensor_M1_samp.shape, mask_samp_M1.shape, mask_samp_all.shape)
            if use_truth_Mdiff:
                mask_tensor_Mdiff_samp = (mask_Mdiff_truth[jb])
            else:
                # mask_tensor_Mdiff_samp = torch.Tensor(np.copy(mask_samp))
                mask_tensor_Mdiff_samp = torch.from_numpy(mask_samp_M_diff)
                mask_tensor_Mdiff_samp = mask_tensor_Mdiff_samp.float().cuda()
            mask_tensor_Mdiff_samp_out.append(mask_tensor_Mdiff_samp)

            if use_truth_Nhalo:
                Nhalo_conditional = Nhalos_truth[jb, ...]
            else:
                if train_multi:
                    Nhalo_conditional = torch.Tensor(np.array([Ntot_samp]).T)
                    Nhalo_conditional = Nhalo_conditional.float().cuda()
                else:
                    raise ValueError('Must use truth Nhalo if not training Ntot')

            cond_inp_M1 = torch.cat([Nhalo_conditional, cond_out], dim=1)
            if x_M1_FP is not None:
                cond_inp_M1 = torch.cat([cond_inp_M1, x_M1_FP[jb]], dim=1)
            if mask_M1_truth_all_FP is not None:
                cond_inp_M1 = torch.cat([cond_inp_M1, mask_M1_truth_all_FP[jb][:,None]], dim=1)            
            
            if reg_M1 or train_M1:
                if self.sep_M1_cond:
                    cond_inp_M1 = self.cond_M1_layer(cond_inp_M1)
                cond_inp_M1_out.append(cond_inp_M1)
                
            # print(Ntot_samp.shape,cond_inp_M1.shape, mask_tensor_M1_samp.shape)
            if reg_M1:
                M1_samp = self.M1reg_model.inverse(cond_inp_M1, mask_tensor_M1_samp)
            # import pdb; pdb.set_trace()
            if train_M1:
                M1_samp, _ = self.M1_model.inverse(cond_inp_M1, mask_tensor_M1_samp)
            else:
                # M1_samp = None
                if not reg_M1:
                    M1_samp = M1_truth[jb, ...][:,0]
            M1_samp_out.append(M1_samp)

            if use_truth_M1:
                M1_conditional = M1_truth[jb, ...]
            else:
                if train_M1:
                    M1_conditional = torch.unsqueeze(M1_samp, 0).T
                else:
                    raise ValueError('Must use truth M1 if not training M1')

            if train_Mdiff:
                cond_inp_Mdiff = torch.cat([Nhalo_conditional, M1_conditional, cond_out], dim=1)
                if self.sep_Mdiff_cond:
                    cond_inp_Mdiff = self.cond_Mdiff_layer(cond_inp_Mdiff)
                M_diff_samp, _ = self.Mdiff_model.inverse(cond_inp_Mdiff, mask_tensor_Mdiff_samp)
            else:
                M_diff_samp = Mdiff_truth[jb, ...]
            M_diff_samp_out.append(M_diff_samp)

            # import pdb; pdb.set_trace()

        # if not train_M1:
        # M1_samp_out = M1_truth

        # return Ntot_samp_out, M1_samp_out, M_diff_samp_out, mask_tensor_M1_samp_out, mask_tensor_Mdiff_samp_out
        return Ntot_samp_out, M1_samp_out, M_diff_samp_out, mask_tensor_M1_samp_out, mask_tensor_Mdiff_samp_out, cond_inp_M1_out
