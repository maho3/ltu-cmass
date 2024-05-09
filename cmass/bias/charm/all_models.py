import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .utils import unconstrained_RQS
from torch.distributions import HalfNormal, Weibull, Gumbel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[:,1:] - fp[:,:-1]) / (xp[:,1:] - xp[:,:-1])  #slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]) )

    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1  #torch.ge:  x[i] >= xp[i] ? true: false
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.linspace(0, indicies.shape[0], 1, device=indicies.device).to(torch.long)
    line_idx = line_idx.expand(indicies.shape)
    # idx = torch.cat([line_idx, indicies] , 0)
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]




class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, activation="tanh"):
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


class BinaryMaskModel(nn.Module):
    """
    This function gets the mask for the halo field. That is conditional on the features, it predicts if the voxel has a halo or not
    """

    def __init__(self,
            dim=1,
            hidden_dim=8,
            base_network=FCNN,
            num_cond=0,
        ):
        super().__init__()
        self.dim = dim
        self.num_cond = num_cond

        self.layer_init = base_network(self.num_cond, 1, hidden_dim)

        if self.num_cond == 0:
            self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, cond_inp=None):
        out = self.layer_init(cond_inp)
        loss = nn.BCEWithLogitsLoss(reduction='none')
        lossv = loss(out.reshape(-1,1), x.reshape(-1, 1))
        return lossv[:,0]

    def inverse(self, cond_inp=None):
        out = self.layer_init(cond_inp)
        out = torch.sigmoid(out)
        return out
    
class MultiClassMaskModel(nn.Module):
    """
    This function gets the number of halos in the voxels that have atleast one halo. That is conditional on the features, it predicts the number of halos in the voxel
    """

    def __init__(self,
            dim=1,
            hidden_dim=8,
            base_network=FCNN,
            num_cond=0,
            num_classes=2
        ):
        super().__init__()
        self.dim = dim
        self.num_cond = num_cond
        self.num_classes = num_classes

        self.layer_init = base_network(self.num_cond, self.num_classes, hidden_dim)

        if self.num_cond == 0:
            self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, cond_inp=None):
        out = self.layer_init(cond_inp)
        loss = nn.CrossEntropyLoss(reduction='none')
        lossv = loss(out, x[:,0].type(torch.long))
        return lossv

    def inverse(self, cond_inp=None, mask=None):
        out = self.layer_init(cond_inp)
        out = torch.softmax(out, dim=1)
        if mask is not None:
            out *= mask[:, 0]
        return out


class SumGaussModel(nn.Module):
    """
    This function is for the quantization of the halo field. That is it models the probability of 
    observing number of halos in a given voxel as a sum of gausians.
    """

    def __init__(
            self,
            dim=1,
            hidden_dim=8,
            base_network=FCNN,
            num_cond=0,
            ngauss=1,
            mu_all=None,
            sig_all=None,
            base_dist='normal'
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.num_cond = num_cond
        self.ngauss = ngauss
        self.base_dist = base_dist
        if mu_all is not None:
            self.mu_all = torch.tensor(mu_all, device=device)
        else:
            self.mu_all = mu_all
        if sig_all is not None:
            self.sig_all = torch.tensor(sig_all, device=device)
            self.var_all = torch.tensor(sig_all**2, device=device)
        else:
            self.sig_all = sig_all
            self.var_all = None

        if (self.mu_all is None) or (self.sig_all is None):
            self.layer_init = base_network(self.num_cond, 3 * self.ngauss, hidden_dim)
        else:
            if base_dist == 'normal':
                self.layer_init = base_network(self.num_cond, self.ngauss, hidden_dim)
            elif base_dist == 'pl_exp':
                self.layer_init = base_network(self.num_cond, self.ngauss + 2, hidden_dim)
            else:
                raise ValueError("base_dist not supported")

        if self.num_cond == 0:
            self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, cond_inp=None):
        out = self.layer_init(cond_inp)
        if (self.mu_all is None) or (self.sig_all is None):
            mu_all, alpha_all, pw_all = (
                out[:, 0:self.ngauss],
                out[:, self.ngauss:2 * self.ngauss],
                out[:, 2 * self.ngauss:3 * self.ngauss],
                )
            mu_all = (1 + nn.Tanh()(mu_all)) / 2
            var_all = torch.exp(alpha_all)
            pw_all = nn.Softmax(dim=1)(pw_all)
            Li_all = torch.zeros(mu_all.shape[0])
            Li_all = Li_all.to(device)
            for i in range(self.ngauss):
                Li_all += (
                    pw_all[:, i] * (1 / torch.sqrt(2 * np.pi * var_all[:, i])) *
                    torch.exp(-0.5 * ((x[:, 0] - mu_all[:, i])**2) / (var_all[:, i]))
                    )
            logP = torch.log(Li_all)
        else:
            mu_all, var_all = self.mu_all, self.var_all
            if self.base_dist == 'normal':
                pw_all_inp = self.layer_init(cond_inp)
            elif self.base_dist == 'pl_exp':
                # in this we have a power law and an exponential as a base distribution
                out = self.layer_init(cond_inp)
                out = torch.exp(out)
                pw_all_orig = out[:, 0:self.ngauss]
                al = out[:, self.ngauss]
                # put al between 0 and 2
                al = 0. * nn.Tanh()(al)
                # al = 5. * nn.Tanh()(al)
                # al = 5*(1 + nn.Tanh()(al))
                bt = out[:, self.ngauss + 1] + 1.
                # bt = out[:, self.ngauss + 1]
                # bt = 5*(1 + nn.Tanh()(bt))                
                # we first predict the base distirbution given the alpha and beta of the form mu**alpha * exp(-beta*mu)
                base_pws = torch.zeros(x.shape[0], self.ngauss)
                base_pws = base_pws.to(device)
                for i in range(self.ngauss):
                    base_pws[:, i] = torch.pow(mu_all[i], al) * torch.exp(-bt * mu_all[i])
                pw_all_inp = torch.mul(pw_all_orig, base_pws)
                pw_all_inp = torch.log(pw_all_inp)
            else:
                raise ValueError("base_dist not supported")

            pw_all = nn.Softmax(dim=1)(pw_all_inp)
            # pw_all = nn.Softmax(dim=1)(torch.log(pw_all_inp))
            Li_all = torch.zeros(x.shape[0])
            Li_all = Li_all.to(device)
            for i in range(self.ngauss):
                Li_all += (
                    pw_all[:, i] * (1 / torch.sqrt(2 * np.pi * var_all[i])) *
                    torch.exp(-0.5 * ((x[:, 0] - mu_all[i])**2) / (var_all[i]))
                    )

            neglogP = -torch.log(Li_all + 1e-30)
        return neglogP

    def inverse(self, cond_inp=None):
        if (self.mu_all is None) or (self.sig_all is None):
            out = self.layer_init(cond_inp)
            mu_all, alpha_all, pw_all = (
                out[:, 0:self.ngauss],
                out[:, self.ngauss:2 * self.ngauss],
                out[:, 2 * self.ngauss:3 * self.ngauss],
                )
            mu_all = (1 + nn.Tanh()(mu_all)) / 2
            pw_all = nn.Softmax(dim=1)(pw_all)
            var_all = torch.exp(alpha_all)
            counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=pw_all).sample()
            counts = counts.to(device)
            # loop over gaussians
            z = torch.empty(0, device=counts.device)
            for k in range(self.ngauss):
                # find indices where count is non-zero for kth gaussian
                ind = torch.nonzero(counts[:, k])
                # if there are any indices, sample from kth gaussian
                if ind.shape[0] > 0:
                    z_k = (mu_all[ind, k][:, 0] + torch.randn(ind.shape[0]) * torch.sqrt(var_all[ind, k])[:, 0])
                    z = torch.cat((z, z_k), dim=0)

        else:
            # print(self.base_dist)
            # import pdb; pdb.set_trace()
            if self.base_dist == 'normal':
                pw_all_inp = self.layer_init(cond_inp)
            elif self.base_dist == 'pl_exp':
                # in this we have a power law and an exponential as a base distribution
                out = self.layer_init(cond_inp)
                out = torch.exp(out)
                pw_all_orig = out[:, 0:self.ngauss]
                al = out[:, self.ngauss]
                al = 0. * nn.Tanh()(al)
                # al = 5. * nn.Tanh()(al)
                # al = 5*(1 + nn.Tanh()(al))
                bt = out[:, self.ngauss + 1] + 1.
                
                # bt = out[:, self.ngauss + 1]
                # bt = 5*(1 + nn.Tanh()(bt))
                # we first predict the base distirbution given the alpha and beta of the form mu**alpha * exp(-beta*mu)
                base_pws = torch.zeros(out.shape[0], self.ngauss)
                base_pws = base_pws.to(device)
                for i in range(self.ngauss):
                    base_pws[:, i] = torch.pow(self.mu_all[i], al) * torch.exp(-bt * self.mu_all[i])
                pw_all_inp = torch.mul(pw_all_orig, base_pws)
                pw_all_inp = torch.log(pw_all_inp)
                

            pw_all = nn.Softmax(dim=1)(pw_all_inp)
            
            # pw_all = nn.Softmax(dim=1)(torch.log(pw_all_inp))

            var_all = self.var_all
            mu_all = self.mu_all

            counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=pw_all).sample()
            counts = counts.to(device)
            # import pdb; pdb.set_trace()
            # loop over gaussians
            # z = torch.empty(0, device=counts.device)
            z_out = torch.empty(counts.shape[0], device=counts.device)
            for k in range(self.ngauss):
                # find indices where count is non-zero for kth gaussian
                ind = torch.nonzero(counts[:, k])
                # if there are any indices, sample from kth gaussian
                # import pdb; pdb.set_trace()    
                if ind.shape[0] > 0:
                    z_k = (mu_all[k] + torch.randn(ind.shape[0], device=device) * torch.sqrt(var_all[k]))
                    z_out[ind[:, 0]] = z_k
                    # z = torch.cat((z, z_k), dim=0)
                            
        return z_out

    def sample(self, cond_inp=None, mask=None):
        x = self.inverse(cond_inp, mask)
        return x





class NSF_M1_CNNcond(nn.Module):
    """
    This function models the probability of observing the heaviest halo mass given the density field.
    """

    def __init__(
        self,
        dim=1,
        K=5,
        B=3,
        hidden_dim=8,
        base_network=FCNN,
        num_cond=0,
        nflows=1,
        ngauss=1,
        base_dist="gauss",
        mu_all=None,
        mu_pos=False,
        base_dist_pwall=None,
        lgM_rs_tointerp=None,
        hmf_pdf_tointerp=None,
        hmf_cdf_tointerp=None        
        ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.num_cond = num_cond
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        self.num_cond = num_cond
        self.base_dist_pwall = base_dist_pwall
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
            else:
                if self.base_dist_pwall == 'pl_exp':
                    if mu_all is not None:
                        self.mu_all = torch.tensor(mu_all, device=device)
                        self.layer_init_gauss = base_network(self.num_cond, 2 * self.ngauss+2, hidden_dim)
                    else:
                        self.mu_all = None
                        self.layer_init_gauss = base_network(self.num_cond, 3 * self.ngauss+2, hidden_dim)
                else:
                    if mu_all is not None:
                        self.mu_all = torch.tensor(mu_all, device=device)
                        self.layer_init_gauss = base_network(self.num_cond, 2 * self.ngauss, hidden_dim)
                    else:
                        self.mu_all = None
                        self.layer_init_gauss = base_network(self.num_cond, 3 * self.ngauss, hidden_dim)
        elif self.base_dist == 'weibull':
            self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
        elif self.base_dist == 'gumbel':
            self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
        elif self.base_dist == 'physical_hmf':
            self.lgM_rs_tointerp = torch.Tensor(np.array([lgM_rs_tointerp])).to(device)
            self.hmf_pdf_tointerp = torch.log(torch.Tensor(np.array([hmf_pdf_tointerp])).to(device))
            self.hmf_cdf_tointerp = torch.Tensor(np.array([hmf_cdf_tointerp])).to(device)
        else:
            print('base_dist not recognized')
            raise ValueError

        self.layers = nn.ModuleList()
        for jf in range(nflows):
            self.layers += [base_network(self.num_cond, 3 * K - 1, hidden_dim)]

        try:
            self.reset_parameters(self.initial_param, -1, 1)
        except:
            pass

    def reset_parameters(self, params, min=0, max=1):
        init.uniform_(params, min, max)

    def get_gauss_func_mu_alpha(self, cond_inp=None):
        out = self.layer_init_gauss(cond_inp)
        if self.ngauss == 1:
            mu, alpha = out[:, 0], out[:, 1]
            if self.mu_pos:
                mu = (1 + nn.Tanh()(mu)) / 2
            var = torch.exp(alpha)
            return mu, var
        else:
            if self.mu_all is not None:
                # alpha_all, pw_all_orig = (
                #     out[:, 0:self.ngauss], out[:, self.ngauss:2 * self.ngauss]
                #     )
                alpha_all = out[:, 0:self.ngauss]
                pw_all_orig = out[:, self.ngauss:2 * self.ngauss]
                mu_all = self.mu_all.reshape(1, self.ngauss).repeat(out.shape[0], 1)
                # import pdb; pdb.set_trace()
            else:
                mu_all, alpha_all, pw_all_orig = (
                    out[:, 0:self.ngauss], out[:, self.ngauss:2 * self.ngauss], out[:, 2 * self.ngauss:3 * self.ngauss]
                    )
            if self.mu_pos:
                mu_all = (1 + nn.Tanh()(mu_all)) / 2
            if self.base_dist_pwall == 'pl_exp':
                # in this we have a power law and an exponential as a base distribution
                pw_all_orig = torch.exp(pw_all_orig)
                al = torch.exp(out[:, 3*self.ngauss])
                # put al between 0 and 2
                al = 0. * nn.Tanh()(al)
                bt = torch.exp(out[:, 3*self.ngauss + 1]) + 1.
                # we first predict the base distirbution given the alpha and beta of the form mu**alpha * exp(-beta*mu)
                base_pws = torch.zeros(out.shape[0], self.ngauss)
                base_pws = base_pws.to(device)
                for i in range(self.ngauss):
                    base_pws[:, i] = torch.pow(mu_all[:,i], al) * torch.exp(-bt * mu_all[:,i])
                
                pw_all = torch.mul(pw_all_orig, base_pws)
            else:
                pw_all = pw_all_orig

            pw_all = nn.Softmax(dim=1)(pw_all)
            var_all = torch.exp(alpha_all)
            # import pdb; pdb.set_trace()
            return mu_all, var_all, pw_all

    def forward(self, x, cond_inp=None):
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                mu, var = self.get_gauss_func_mu_alpha(cond_inp)
            else:
                mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha(cond_inp)
        elif self.base_dist in ['weibull', 'gumbel']:
            out = self.layer_init_gauss(cond_inp)
            mu, alpha = out[:, 0], out[:, 1]
            if self.base_dist == 'weibull':
                scale, conc = torch.exp(mu), torch.exp(alpha)
            else:
                if self.mu_pos:
                    # mu = torch.exp(mu)
                    mu = (1 + nn.Tanh()(mu)) / 2
                sig = torch.exp(alpha)
        elif self.base_dist == 'physical_hmf':
            pass                
        else:
            print('base_dist not recognized')
            raise ValueError
        # import pdb; pdb.set_trace()
        if len(x.shape) > 1:
            x = x[:, 0]
        log_det_all = torch.zeros_like(x)
        for jf in range(self.nflows):
            out = self.layers[jf](cond_inp)
            z = torch.zeros_like(x)
            # log_det_all = torch.zeros(z.shape)
            W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            if isinstance(self.B, float) or isinstance(self.B, int):
                W, H = 2 * self.B * W, 2 * self.B * H
            else:
                W, H = (self.B[1] - self.B[0]) * W, (self.B[1] - self.B[0]) * H
            # W, H = 2 * self.B * W, 2 * self.B * H
            # D = F.softplus(D)
            D = 2. * F.sigmoid(D)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)
            log_det_all += ld
            x = z

        if self.base_dist == 'gauss':
            if self.ngauss == 1:
                logp = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(var) - 0.5 * (x - mu)**2 / var
            else:
                Li_all = torch.zeros(mu_all.shape[0])
                Li_all = Li_all.to(device)
                for i in range(self.ngauss):
                    Li_all += (
                        pw_all[:, i] * (1 / torch.sqrt(2 * np.pi * var_all[:, i])) *
                        torch.exp(-0.5 * ((x - mu_all[:, i])**2) / (var_all[:, i]))
                        )
                logp = torch.log(Li_all)

        elif self.base_dist == 'halfgauss':
            if self.ngauss == 1:
                x = torch.exp(x - mu)
                hf = HalfNormal((torch.sqrt(var)))
                logp = hf.log_prob(x)

        elif self.base_dist == 'weibull':
            hf = Weibull(scale, conc)
            logp = hf.log_prob(x)
            # if there are any nans of infs, replace with -100
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100
        elif self.base_dist == 'gumbel':
            hf = Gumbel(mu, sig)
            logp = hf.log_prob(x)
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100
        elif self.base_dist == 'physical_hmf':
            # use the interpolation function to get the logp
            # add another axis to x
            # import pdb; pdb.set_trace()
            logp = (interpolate(x[None,:], self.lgM_rs_tointerp, self.hmf_pdf_tointerp))[0,:]
            # import pdb; pdb.set_trace()
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100

        else:
            raise ValueError("Base distribution not supported")

        logp = log_det_all + logp
        return logp

    def inverse(self, cond_inp=None, mask=None):
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                mu, var = self.get_gauss_func_mu_alpha(cond_inp)
            else:
                mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha(cond_inp)
        elif self.base_dist in ['weibull', 'gumbel']:
            out = self.layer_init_gauss(cond_inp)
            mu, alpha = out[:, 0], out[:, 1]
            if self.base_dist == 'weibull':
                scale, conc = torch.exp(mu), torch.exp(alpha)
            else:
                if self.mu_pos:
                    # mu = torch.exp(mu)
                    mu = (1 + nn.Tanh()(mu)) / 2
                sig = torch.exp(alpha)
        elif self.base_dist == 'physical_hmf':
            pass                
        else:
            print('base_dist not recognized')
            raise ValueError
        if self.base_dist == 'gauss':
            if self.ngauss == 1:
                x = mu + torch.randn(cond_inp.shape[0]) * torch.sqrt(var)
            else:
                counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=pw_all).sample()
                counts = counts.to(device)
                # loop over gaussians
                x = torch.empty(0, device=counts.device)
                for k in range(self.ngauss):
                    # find indices where count is non-zero for kth gaussian
                    ind = torch.nonzero(counts[:, k])
                    # if there are any indices, sample from kth gaussian
                    if ind.shape[0] > 0:
                        x_k = (mu_all[ind, k][:, 0] + torch.randn(ind.shape[0]).to(device) * torch.sqrt(var_all[ind, k])[:, 0])
                        x = torch.cat((x, x_k), dim=0)

        elif self.base_dist == 'halfgauss':
            if self.ngauss == 1:
                x = torch.log(mu + torch.abs(torch.randn(cond_inp.shape[0])) * torch.sqrt(var))

        elif self.base_dist == 'weibull':
            hf = Weibull(scale, conc)
            x = hf.sample()
        elif self.base_dist == 'gumbel':
            hf = Gumbel(mu, sig)
            x = hf.sample()
        elif self.base_dist == 'physical_hmf':
            u = torch.rand(cond_inp.shape[0])
            u = u.to(device)
            # import pdb; pdb.set_trace()
            # x = interpolate(torch.log(u)[None,:], torch.log(self.hmf_cdf_tointerp[:,1:]), self.lgM_rs_tointerp[:,1:])[0,:]
            x = interpolate((u)[None,:], (self.hmf_cdf_tointerp[:,1:]), self.lgM_rs_tointerp[:,1:])[0,:]

        log_det_all = torch.zeros_like(x)
        for jf in range(self.nflows):
            ji = self.nflows - jf - 1
            out = self.layers[ji](cond_inp)
            z = torch.zeros_like(x)
            W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            # W, H = 2 * self.B * W, 2 * self.B * H
            if isinstance(self.B, float) or isinstance(self.B, int):
                W, H = 2 * self.B * W, 2 * self.B * H
            else:
                W, H = (self.B[1] - self.B[0]) * W, (self.B[1] - self.B[0]) * H            
            # D = F.softplus(D)
            D = 2. * F.sigmoid(D)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=self.B)
            log_det_all += ld
            x = z

        if mask is not None:
            x *= mask[:, 0]
        return x, log_det_all

    def sample(self, cond_inp=None, mask=None):
        x, _ = self.inverse(cond_inp, mask)
        return x


class NSF_Mdiff_CNNcond(nn.Module):
    """
    This function models the probability of observing all the lower halo masses
    """

    def __init__(
        self,
        dim=None,
        K=5,
        B=3,
        hidden_dim=8,
        base_network=FCNN,
        num_cond=0,
        nflows=1,
        ngauss=1,
        base_dist="gumbel",
        mu_pos=False,
        base_dist_pwall = 'pl_exp'
        ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.num_cond = num_cond
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        self.num_cond = num_cond
        self.base_dist_pwall = base_dist_pwall
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        self.layers_all_dim = nn.ModuleList()
        self.layers_all_dim_init = nn.ModuleList()
        # self.layers_all_dim = []
        # self.layers_all_dim_init = []
        for jd in range(dim):
            if self.base_dist in ["gauss", "halfgauss"]:
                if self.ngauss == 1:
                    layer_init_gauss = base_network(self.num_cond + jd, 2, hidden_dim)
                else:
                    if self.base_dist_pwall == 'pl_exp':
                        layer_init_gauss = base_network(self.num_cond + jd, 3 * self.ngauss + 2, hidden_dim)
                    else:
                        layer_init_gauss = base_network(self.num_cond + jd, 3 * self.ngauss, hidden_dim)
            elif self.base_dist == 'weibull':
                layer_init_gauss = base_network(self.num_cond + jd, 2, hidden_dim)
            elif self.base_dist == 'gumbel':
                layer_init_gauss = base_network(self.num_cond + jd, 2, hidden_dim)
            else:
                print('base_dist not recognized')
                raise ValueError
            self.layers_all_dim_init += [layer_init_gauss]

            layers = nn.ModuleList()
            for jf in range(nflows):
                layers += [base_network(self.num_cond + jd, 3 * K - 1, hidden_dim)]
            self.layers_all_dim += [layers]

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, -1 / 2, 1 / 2)

    def get_gauss_func_mu_alpha(self, jd, cond_inp=None):
        out = self.layers_all_dim_init[jd](cond_inp)
        if self.ngauss == 1:
            mu, alpha = out[:, 0], out[:, 1]
            if self.mu_pos:
                mu = (1 + nn.Tanh()(mu)) / 2
            var = torch.exp(alpha)
            return mu, var
        else:
            mu_all, alpha_all, pw_all_orig = (
                out[:, 0:self.ngauss], out[:, self.ngauss:2 * self.ngauss], out[:, 2 * self.ngauss:3 * self.ngauss]
                )
            if self.mu_pos:
                mu_all = (1 + nn.Tanh()(mu_all)) / 2
            else:
                mu_all = (nn.Tanh()(mu_all))
            if self.base_dist_pwall == 'pl_exp':
                # in this we have a power law and an exponential as a base distribution
                pw_all_orig = torch.exp(pw_all_orig)
                al = torch.exp(out[:, 3*self.ngauss])
                # put al between 0 and 2
                al = 0. * nn.Tanh()(al)
                bt = torch.exp(out[:, 3*self.ngauss + 1]) + 1.
                # we first predict the base distirbution given the alpha and beta of the form mu**alpha * exp(-beta*mu)
                base_pws = torch.zeros(out.shape[0], self.ngauss)
                base_pws = base_pws.to(device)
                for i in range(self.ngauss):
                    base_pws[:, i] = torch.pow(mu_all[:,i], al) * torch.exp(-bt * mu_all[:,i])
                
                pw_all = torch.mul(pw_all_orig, base_pws)
            else:
                pw_all = pw_all_orig


            pw_all = nn.Softmax(dim=1)(pw_all)
            var_all = torch.exp(alpha_all)
            return mu_all, var_all, pw_all

    def forward(self, x_inp, cond_inp=None, mask=None):
        logp = torch.zeros_like(x_inp)
        logp = logp.to(device)
        x_inp = x_inp.to(device)
        for jd in range(self.dim):
            # print(cond_inp.shape)
            if jd > 0:
                cond_inp_jd = torch.cat([cond_inp, x_inp[:, :jd]], dim=1)
            else:
                cond_inp_jd = cond_inp
            # print(cond_inp.shape)
            if self.base_dist in ["gauss","halfgauss"]:
                if self.ngauss == 1:
                    mu, var = self.get_gauss_func_mu_alpha(jd, cond_inp_jd)
                else:
                    mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha(jd, cond_inp_jd)

            elif self.base_dist in ['weibull', 'gumbel']:
                out = self.layers_all_dim_init[jd](cond_inp_jd)
                mu, alpha = out[:, 0], out[:, 1]
                if self.base_dist == 'weibull':
                    scale, conc = torch.exp(mu), torch.exp(alpha)
                else:
                    if self.mu_pos:
                        # mu = torch.exp(mu)
                        mu = (1 + nn.Tanh()(mu)) / 2
                    sig = torch.exp(alpha)
            else:
                print('base_dist not recognized')
                raise ValueError

            # if len(x.shape) > 1:
            #     x = x[:, 0]
            log_det_all_jd = torch.zeros(x_inp.shape[0])
            log_det_all_jd = log_det_all_jd.to(device)
            for jf in range(self.nflows):
                if jf == 0:
                    x = x_inp[:, jd]
                    x = x.to(device)
                out = self.layers_all_dim[jd][jf](cond_inp_jd)
                # z = torch.zeros_like(x)
                # log_det_all = torch.zeros(z.shape)
                W, H, D = torch.split(out, self.K, dim=1)
                W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
                # W, H = 2 * self.B * W, 2 * self.B * H
                D = F.softplus(D)
                if isinstance(self.B, float) or isinstance(self.B, int):
                    W, H = 2 * self.B * W, 2 * self.B * H
                    z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)
                else:
                    if isinstance(self.B[0], float) or isinstance(self.B[0], int):
                        W, H = (self.B[1] - self.B[0]) * W, (self.B[1] - self.B[0]) * H  
                        z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)         
                    else:
                        try:
                            W, H = (self.B[jd][1] - self.B[jd][0]) * W, (self.B[jd][1] - self.B[jd][0]) * H
                            z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=[self.B[jd][0], self.B[jd][1]])
                        except:
                            W, H = (self.B[jd-1][1] - self.B[jd-1][0]) * W, (self.B[jd-1][1] - self.B[jd-1][0]) * H
                            z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=[self.B[jd-1][0], self.B[jd-1][1]])
                # z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)
                log_det_all_jd += ld
                x = z

            if self.base_dist == 'gauss':
                if self.ngauss == 1:
                    logp_jd = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(var) - 0.5 * (x - mu)**2 / var
                else:
                    Li_all = torch.zeros(mu_all.shape[0])
                    Li_all = Li_all.to(device)
                    for i in range(self.ngauss):
                        Li_all += (
                            pw_all[:, i] * (1 / torch.sqrt(2 * np.pi * var_all[:, i])) *
                            torch.exp(-0.5 * ((x - mu_all[:, i])**2) / (var_all[:, i]))
                            )
                    logp_jd = torch.log(Li_all)


            elif self.base_dist == 'halfgauss':
                if self.ngauss == 1:
                    x = torch.exp(x - mu)
                    hf = HalfNormal((torch.sqrt(var)))
                    logp_jd = hf.log_prob(x)

            elif self.base_dist == 'weibull':
                hf = Weibull(scale, conc)
                logp_jd = hf.log_prob(x)
                # if there are any nans of infs, replace with -100
                logp_jd[torch.isnan(logp_jd) | torch.isinf(logp_jd)] = -100
            elif self.base_dist == 'gumbel':
                hf = Gumbel(mu, sig)
                logp_jd = hf.log_prob(x)
                logp_jd[torch.isnan(logp_jd) | torch.isinf(logp_jd)] = -100
            else:
                raise ValueError("Base distribution not supported")

            logp[:, jd] = log_det_all_jd + logp_jd
        if mask is not None:
            logp *= mask
        logp = torch.sum(logp, dim=1)
        # print(logp.shape, mask.shape)
        return logp

    def inverse(self, cond_inp=None, mask=None):
        z_out = torch.zeros((cond_inp.shape[0], self.dim))
        z_out = z_out.to(device)
        for jd in range(self.dim):
            if jd > 0:
                cond_inp_jd = torch.cat([cond_inp, z_out[:, :jd]], dim=1)
            else:
                cond_inp_jd = cond_inp
            if self.base_dist in ["gauss","halfgauss"]:
                if self.ngauss == 1:
                    mu, var = self.get_gauss_func_mu_alpha(jd, cond_inp_jd)
                else:
                    mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha(jd, cond_inp_jd)


            elif self.base_dist in ['gumbel', 'weibull']:
                out = self.layers_all_dim_init[jd](cond_inp_jd)
                mu, alpha = out[:, 0], out[:, 1]
                if self.base_dist == 'weibull':
                    scale, conc = torch.exp(mu), torch.exp(alpha)
                else:
                    if self.mu_pos:
                        # mu = torch.exp(mu)
                        mu = (1 + nn.Tanh()(mu)) / 2
                    sig = torch.exp(alpha)
            else:
                print('base_dist not recognized')
                raise ValueError

            if self.base_dist == 'gauss':
                if self.ngauss == 1:
                    x = mu + torch.randn(cond_inp_jd.shape[0], device=device) * torch.sqrt(var)
                else:
                    counts = torch.distributions.multinomial.Multinomial(total_count=1, probs=pw_all).sample()
                    counts = counts.to(device)
                    # loop over gaussians
                    x = torch.empty(0, device=counts.device)
                    for k in range(self.ngauss):
                        # find indices where count is non-zero for kth gaussian
                        ind = torch.nonzero(counts[:, k])
                        # if there are any indices, sample from kth gaussian
                        if ind.shape[0] > 0:
                            x_k = (mu_all[ind, k][:, 0] + torch.randn(ind.shape[0]).to(device) * torch.sqrt(var_all[ind, k])[:, 0])
                            x = torch.cat((x, x_k), dim=0)


            elif self.base_dist == 'halfgauss':
                if self.ngauss == 1:
                    x = torch.log(mu + torch.abs(torch.randn(cond_inp_jd.shape[0], device=device)) * torch.sqrt(var))

            elif self.base_dist == 'weibull':
                hf = Weibull(scale, conc)
                x = hf.sample()
            elif self.base_dist == 'gumbel':
                hf = Gumbel(mu, sig)
                x = hf.sample()
                # print(x.shape)
                # print(mu, sig)
            else:
                raise ValueError("Base distribution not supported")

            log_det_all = torch.zeros_like(x)
            for jf in range(self.nflows):
                ji = self.nflows - jf - 1
                out = self.layers_all_dim[jd][ji](cond_inp_jd)
                z = torch.zeros_like(x)
                W, H, D = torch.split(out, self.K, dim=1)
                W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
                D = F.softplus(D)
                # W, H = 2 * self.B * W, 2 * self.B * H
                if isinstance(self.B, float) or isinstance(self.B, int):
                    W, H = 2 * self.B * W, 2 * self.B * H
                    z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=self.B)                    
                else:
                    if isinstance(self.B[0], float) or isinstance(self.B[0], int):
                        W, H = (self.B[1] - self.B[0]) * W, (self.B[1] - self.B[0]) * H    
                        z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=self.B)        
                    else:
                        try:
                            W, H = (self.B[jd][1] - self.B[jd][0]) * W, (self.B[jd][1] - self.B[jd][0]) * H
                            z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=[self.B[jd][0], self.B[jd][1]])
                        except:
                            W, H = (self.B[jd-1][1] - self.B[jd-1][0]) * W, (self.B[jd-1][1] - self.B[jd-1][0]) * H
                            z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=[self.B[jd-1][0], self.B[jd-1][1]])

                log_det_all += ld
                x = z

            x *= mask[:, jd]
            z_out[:, jd] = x
        return z_out, log_det_all

    def sample(self, cond_inp=None, mask=None):
        x, _ = self.inverse(cond_inp, mask)
        return x


class NSF_M_all_uncond(nn.Module):
    """
    This function models the probability of observing the heaviest halo mass given the density field.
    """

    def __init__(
        self,
        dim=1,
        K=5,
        B=3,
        nflows=1,
        ngauss=1,
        base_dist="gauss",
        mu_pos=False,
        base_dist_pwall = 'pl_exp',
        lgM_rs_tointerp=None,
        hmf_pdf_tointerp=None,
        hmf_cdf_tointerp=None
        ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        # self.num_cond = num_cond
        self.nflows = nflows
        self.ngauss = ngauss
        self.base_dist = base_dist
        self.mu_pos = mu_pos
        # self.num_cond = num_cond
        self.base_dist_pwall = base_dist_pwall
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                self.initial_param = nn.Parameter(torch.Tensor(2))
                # self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
            else:
                if self.base_dist_pwall == 'pl_exp':
                    self.initial_param = nn.Parameter(torch.Tensor(3 * self.ngauss+2))
                    # self.layer_init_gauss = base_network(self.num_cond, 3 * self.ngauss+2, hidden_dim)
                else:
                    self.initial_param = nn.Parameter(torch.Tensor(3 * self.ngauss))                    
                    # self.layer_init_gauss = base_network(self.num_cond, 3 * self.ngauss, hidden_dim)
        elif self.base_dist == 'weibull':
            self.initial_param = nn.Parameter(torch.Tensor(2))
            # self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
        elif self.base_dist == 'gumbel':
            self.initial_param = nn.Parameter(torch.Tensor(2))
            # self.layer_init_gauss = base_network(self.num_cond, 2, hidden_dim)
        elif self.base_dist == 'physical_hmf':
            self.lgM_rs_tointerp = torch.Tensor(np.array([lgM_rs_tointerp])).to(device)
            self.hmf_pdf_tointerp = torch.log(torch.Tensor(np.array([hmf_pdf_tointerp])).to(device))
            self.hmf_cdf_tointerp = torch.Tensor(np.array([hmf_cdf_tointerp])).to(device)
        else:
            print('base_dist not recognized')
            raise ValueError

        self.layers = nn.ParameterList()
        # self.NSF_params = []
        for jf in range(nflows):
            params_jf = nn.Parameter(torch.Tensor(3 * K - 1))
            self.reset_parameters(params_jf, -10, 10)
            self.layers += [params_jf]
            # self.layers += [base_network(self.num_cond, 3 * K - 1, hidden_dim)]
        try:
            self.reset_parameters(self.initial_param, -1, 1)
        except:
            pass

    def reset_parameters(self, params, min=0, max=1):
        init.uniform_(params, min, max)

    def get_gauss_func_mu_alpha(self):
        out = self.initial_param
        if self.ngauss == 1:
            mu, alpha = out[0], out[1]
            if self.mu_pos:
                mu = (1 + nn.Tanh()(mu)) / 2
            var = torch.exp(alpha)
            return mu, var
        else:
            mu_all, alpha_all, pw_all_orig = (
                out[0:self.ngauss], out[self.ngauss:2 * self.ngauss], out[2 * self.ngauss:3 * self.ngauss]
                )
            if self.mu_pos:
                mu_all = (1 + nn.Tanh()(mu_all)) / 2
            if self.base_dist_pwall == 'pl_exp':
                # in this we have a power law and an exponential as a base distribution
                pw_all_orig = torch.exp(pw_all_orig)
                al = torch.exp(out[3*self.ngauss])
                # put al between 0 and 2
                al = 0. * nn.Tanh()(al)
                bt = torch.exp(out[3*self.ngauss + 1]) + 1.
                # we first predict the base distirbution given the alpha and beta of the form mu**alpha * exp(-beta*mu)
                base_pws = torch.zeros(self.ngauss)
                base_pws = base_pws.to(device)
                for i in range(self.ngauss):
                    base_pws[i] = torch.pow(mu_all[i], al) * torch.exp(-bt * mu_all[i])
                
                pw_all = torch.mul(pw_all_orig, base_pws)
            else:
                pw_all = pw_all_orig

            pw_all = nn.Softmax(dim=0)(pw_all)
            var_all = torch.exp(alpha_all)
            return mu_all, var_all, pw_all

    def forward(self, x):
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                mu, var = self.get_gauss_func_mu_alpha()
            else:
                mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha()
        elif self.base_dist in ['weibull', 'gumbel']:
            out = self.initial_param
            mu, alpha = out[0], out[1]
            if self.base_dist == 'weibull':
                scale, conc = torch.exp(mu), torch.exp(alpha)
            else:
                if self.mu_pos:
                    # mu = torch.exp(mu)
                    mu = (1 + nn.Tanh()(mu)) / 2
                sig = torch.exp(alpha)
        elif self.base_dist == 'physical_hmf':
            pass
        else:
            print('base_dist not recognized')
            raise ValueError
        if len(x.shape) > 1:
            x = x[:, 0]
        log_det_all = torch.zeros_like(x)
        for jf in range(self.nflows):
            out = self.layers[jf]
            z = torch.zeros_like(x)
            # log_det_all = torch.zeros(z.shape)
            W, H, D = torch.split(out, self.K)
            W, H = torch.softmax(W, dim=0), torch.softmax(H, dim=0)
            W, H = 2 * self.B * W, 2 * self.B * H
            # D = F.softplus(D)
            D = 2. * F.sigmoid(D)
            # import pdb; pdb.set_trace()
            W = W.unsqueeze(0).repeat(x.shape[0], 1)
            H = H.unsqueeze(0).repeat(x.shape[0], 1)
            D = D.unsqueeze(0).repeat(x.shape[0], 1)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=False, tail_bound=self.B)
            log_det_all += ld
            x = z
            # x = nn.Sigmoid()(x)
        # x = nn.Sigmoid()(x)
        # x = torch.exp(x)
        if self.base_dist == 'gauss':
            if self.ngauss == 1:
                logp = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(var) - 0.5 * (x - mu)**2 / var
            else:
                Li_all = torch.zeros(x.shape)
                Li_all = Li_all.to(device)
                for i in range(self.ngauss):
                    Li_all += (
                        pw_all[i] * (1 / torch.sqrt(2 * np.pi * var_all[i])) *
                        torch.exp(-0.5 * ((x - mu_all[i])**2) / (var_all[i]))
                        )
                logp = torch.log(Li_all)

        elif self.base_dist == 'halfgauss':
            if self.ngauss == 1:
                x = torch.exp(x - mu)
                hf = HalfNormal((torch.sqrt(var)))
                logp = hf.log_prob(x)

        elif self.base_dist == 'weibull':
            hf = Weibull(scale, conc)
            logp = hf.log_prob(x)
            # if there are any nans of infs, replace with -100
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100
        elif self.base_dist == 'gumbel':
            hf = Gumbel(mu, sig)
            logp = hf.log_prob(x)
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100
        elif self.base_dist == 'physical_hmf':
            # use the interpolation function to get the logp
            # add another axis to x
            logp = (interpolate(x[None,:], self.lgM_rs_tointerp, self.hmf_pdf_tointerp)[0,:])
            # import pdb; pdb.set_trace()
            logp[torch.isnan(logp) | torch.isinf(logp)] = -100
        else:
            raise ValueError("Base distribution not supported")

        # logp = log_det_all + logp
        return logp, log_det_all, x

    def inverse(self, ntot:int):
        if self.base_dist in ["gauss", "halfgauss"]:
            if self.ngauss == 1:
                mu, var = self.get_gauss_func_mu_alpha()
            else:
                mu_all, var_all, pw_all = self.get_gauss_func_mu_alpha()
        elif self.base_dist in ['weibull', 'gumbel']:
            out = self.initial_param
            mu, alpha = out[0], out[1]
            if self.base_dist == 'weibull':
                scale, conc = torch.exp(mu), torch.exp(alpha)
            else:
                if self.mu_pos:
                    # mu = torch.exp(mu)
                    mu = (1 + nn.Tanh()(mu)) / 2
                sig = torch.exp(alpha)
        elif self.base_dist == 'physical_hmf':
            pass
        else:
            print('base_dist not recognized')
            raise ValueError

        if self.base_dist == 'gauss':
            if self.ngauss == 1:
                x = mu + torch.randn(ntot) * torch.sqrt(var)
            else:
                counts = torch.distributions.multinomial.Multinomial(total_count=ntot, probs=pw_all).sample()
                counts = counts.to(device)
                # loop over gaussians
                x = torch.empty(0, device=counts.device)
                for k in range(self.ngauss):
                    # find indices where count is non-zero for kth gaussian
                    # ind = torch.nonzero(counts[k])
                    count = counts[k]
                    # if there are any indices, sample from kth gaussian
                    if count > 0:
                        # import pdb; pdb.set_trace()
                        x_k = (mu_all[k] + torch.randn(int(count)).to(device) * torch.sqrt(var_all[k]))
                        x = torch.cat((x, x_k), dim=0)

        elif self.base_dist == 'halfgauss':
            if self.ngauss == 1:
                x = torch.log(mu + torch.abs(torch.randn(ntot)) * torch.sqrt(var))

        elif self.base_dist == 'weibull':
            hf = Weibull(scale, conc)
            x = hf.sample([ntot])
        elif self.base_dist == 'gumbel':
            hf = Gumbel(mu, sig)
            x = hf.sample([ntot])
        elif self.base_dist == 'physical_hmf':
            u = torch.rand(ntot)
            u = u.to(device)
            # import pdb; pdb.set_trace()
            # x = interpolate(torch.log(u)[None,:], torch.log(self.hmf_cdf_tointerp[:,1:]), self.lgM_rs_tointerp[:,1:])[0,:]
            x = interpolate((u)[None,:], (self.hmf_cdf_tointerp[:,1:]), self.lgM_rs_tointerp[:,1:])[0,:]
            # x = x.to(device)



        log_det_all = torch.zeros_like(x)
        for jf in range(self.nflows):
            ji = self.nflows - jf - 1
            out = self.layers[ji]
            z = torch.zeros_like(x)
            W, H, D = torch.split(out, self.K)
            W, H = torch.softmax(W, dim=0), torch.softmax(H, dim=0)
            W, H = 2 * self.B * W, 2 * self.B * H
            # D = F.softplus(D)
            D = 2. * F.sigmoid(D)
            # import pdb; pdb.set_trace()
            W = W.unsqueeze(0).repeat(x.shape[0], 1)
            H = H.unsqueeze(0).repeat(x.shape[0], 1)
            D = D.unsqueeze(0).repeat(x.shape[0], 1)
            z, ld = unconstrained_RQS(x, W, H, D, inverse=True, tail_bound=self.B)
            log_det_all += ld
            x = z

        # x *= mask[:, 0]
        return x, log_det_all

    def sample(self, ntot):
        x, _ = self.inverse(ntot)
        return x



class M1_reg_model(nn.Module):
    def __init__(
    self,
    dim=1,
    hidden_dim=8,
    num_cond=0):
        super().__init__()
        self.dim = dim
        self.num_cond = num_cond
        # self.layer_reg = base_network(self.num_cond, 1, hidden_dim)
        in_dim = self.num_cond
        out_dim = 1
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.Tanh(),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.LeakyReLU(negative_slope=0.5),                        
            nn.Linear(hidden_dim, hidden_dim//2),
            # nn.Tanh(),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            # nn.Tanh(),            
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(hidden_dim//4, out_dim),
            )

        # self.reset_parameters()

    
    def forward(self, cond_inp=None):
        out = self.network(cond_inp)
        return out

    def inverse(self, cond_inp=None, mask=None):
        out = self.network(cond_inp)
        out *= mask
        return out[:,0]
