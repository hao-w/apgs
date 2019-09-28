import torch
import torch.nn as nn
import torch.nn.functional.conv2d
from torch.distributions.normal import Normal
import probtorch
import math

class Dec_coor(nn.Module):
    """
    z_where S * B * T * 2
    """
    def __init__(self, num_hidden, init_std, noise_std, CUDA, device):
        super(self.__class__, self)
        self.dec_hidden = nn.Sequential(
                            nn.Linear(4, num_hidden),
                            nn.Tanh(),
                            nn.Linear(num_hidden, 2),
                            nn.Tanh())

        self.prior_Sigma = torch.ones(2) * noise_std
        self.prior_init_mean = torch.zeros(2)
        self.prior_init_std = torch.ones(2) * init_std
        if CUDA:
            with torch.cuda.device(device):
                self.prior_Sigma = self.prior_Sigma.cuda()
                self.prior_init_mean = self.prior_init_mean.cuda()
                self.prior_init_std = self.prior_init_std.mean()

    def forward(self, init_v, z_where, sampled=False):
        S, B, T _ = z_where.shape
        log_P = []
        log_p_0 = Normal(self.prior_init_mean, self.prior_init_std).log_prob(z_where[:,:,0,:]).sum(-1)
        log_P.append(log_p_0.unsqueeze(2))
        for t in range(T-1):
            if t == 0:
                v_new = self.enc_hidden(torch.cat((z_where[:,:, t, :], init_v), -1))
            else:
                v_new = self.enc_hidden(torch.cat((z_where[:,:, t, :], v_new), -1))

            pt = Normal(v_new * 0.2 + z_where[:,:, t, :], self.prior_Sigma)
            # if sampled:
            #     samples_t = pt.sample()
            #     log_p_t = pt.log_prob(samples_t)
            #     return samples_t
            # else:
            log_p_t = pt.log_prob(z_where[:,:, t+1, :]).sum(-1)
            log_P.append(log_p_t.unsqueeze(2))
        return torch.cat(log_P, 2) # S * B * T
