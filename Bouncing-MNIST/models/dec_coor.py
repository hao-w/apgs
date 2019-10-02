import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import probtorch
import math

class Dec_coor():
    """
    Real generative model for time dynamics
    z_1 ~ N (0, Sigma_0)
    z_t | z_t-1 ~ N (A z_t-1, Sigma_t)
    where A is the transformation matrix
    """
    def __init__(self, D, Sigma0, CUDA, device):
        super(self.__class__, self)

        self.prior_mu0 = torch.zeros(D)
        self.prior_Sigma0 = torch.ones(D) * Sigma0
        self.prior_Sigmat = torch.ones(D) * 0.1
        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu0  = self.prior_mu0 .cuda()
                self.prior_Sigma0 = self.prior_Sigma0.cuda()
                self.prior_Sigmat = self.prior_Sigmat.cuda()
        # self.prior_Sigmat = nn.Parameter(self.prior_Sigmat)
    def forward(self, z_where, disp):
        S, B, T, K, D = z_where.shape
        log_P = []
        p0 = Normal(self.prior_mu0, self.prior_Sigma0)
        log_p0 = p0.log_prob(z_where[:, :, 0, :, :]) # S * B * K * D
        log_v0 = p0.log_prob(disp[:, :, 0, :, :])
        log_P.append(log_p0.unsqueeze(2) + log_v0.unsqueeze(2)) # S * B * 1 * K * D
        new_z_where = z_where[:,:,:T-1,:, :] + disp
        new_disp = torch.where(new_z_where > 1, - disp, disp)
        new_disp = torch.where(new_z_where < -1, - new_disp, new_disp)
        new_z_where = torch.where(new_z_where > 1, 2.0 - new_z_where, new_z_where)
        new_z_where = torch.where(new_z_where < -1, - 2.0 - new_z_where, new_z_where)
        pt = Normal(new_z_where, self.prior_Sigmat)
        log_pt = pt.log_prob(z_where[:,:,1:,:,:])# S * B * T-1 * K * D
        log_P.append(log_pt)
        return new_z_where, new_disp, torch.cat(log_P, 2) # S * B * T * K * D
