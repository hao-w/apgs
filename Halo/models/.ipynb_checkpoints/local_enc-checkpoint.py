import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Enc_z(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(2*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()
        gamma_list = []
        for k in range(K):
            data_ck = torch.cat((obs, obs_mu[:, :, k, :].unsqueeze(-2).repeat(1,1,N,1), obs_rad), -1)
            gamma_list.append(self.pi_log_prob(data_ck))
        q_probs = F.softmax(torch.cat(gamma_list, -1), -1) # S * B * N * K
        z = cat(q_probs).sample()
        _ = q.variable(cat, probs=q_probs, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p
    
    def sample_prior(self, N, sample_size, batch_size):
        p_init_z = cat(self.prior_pi)
        state = p_init_z.sample((sample_size, batch_size, N,))
        return state
    
class Enc_rad(nn.Module):
    def __init__(self, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.rad_log_alpha = nn.Sequential(
            nn.Linear(2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))
        self.rad_log_beta = nn.Sequential(
            nn.Linear(2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))
        
        self.prior_rad_alpha = torch.ones(1)
        self.prior_rad_beta = torch.ones(1)
        if CUDA:
            self.prior_rad_alpha = self.prior_rad_alpha.cuda().to(device)
            self.prior_rad_beta = self.prior_rad_beta.cuda().to(device)
            
    def forward(self, obs, state, obs_mu, D, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()
        labels = state.argmax(-1)
        labels_mu = labels.unsqueeze(-1).repeat(1, 1, 1, D)
        obs_mu_expand = torch.gather(obs_mu, 2, labels_mu)
        data_mu = torch.cat((obs, obs_mu_expand), -1)
        q_rad_alpha = self.rad_log_alpha(data_mu).exp()
        q_rad_beta = self.rad_log_beta(data_mu).exp()
        
        rad = Gamma(q_rad_alpha, q_rad_beta).sample()
        q.gamma(q_rad_alpha,
                q_rad_beta,
                value=rad,
                name='radius')
        p.gamma(self.prior_rad_alpha,
                self.prior_rad_beta,
                value=q['radius'],
                name='radius')
        return q, p
