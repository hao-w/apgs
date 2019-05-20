import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

def data_to_stats(obs, states):
    """
    stat1 : sum of I[z_n=k], S * B * K * 1
    stat2 : sum of I[z_n=k]*x_n, S * B * K * D
    stat3 : sum of I[z_n=k]*x_n^2, S * B * K * D
    """
    D = obs.shape[-1]
    K = states.shape[-1]
    stat1 = states.sum(2).unsqueeze(-1)
    states_expand = states.unsqueeze(-1).repeat(1, 1, 1, 1, D)
    obs_expand = obs.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2)
    stat2 = (states_expand * obs_expand).sum(2)
    stat3 = (states_expand * (obs_expand**2)).sum(2)
    return stat1, stat2, stat3
# ============================
class Enc_eta(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()
        self.Reparameterized = Reparameterized

#         self.neural_stats = nn.Sequential(
#             nn.Linear(K+D, num_hidden),
#             nn.Tanh(),
#             nn.Linear(num_hidden, K*num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(1+2*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(1+2*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.radi_log_alpha = nn.Sequential(
            nn.Linear(1+2*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.radi_log_beta = nn.Sequential(
            nn.Linear(1+2*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_mean_mu = torch.zeros((K, D))
        self.prior_mean_sigma = torch.ones((K, D)) * 4.0
        self.prior_radi_alpha = torch.ones((K, 1)) * 8.0
        self.prior_radi_beta = torch.ones((K, 1)) * 16.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)
            self.prior_radi_alpha = self.prior_radi_alpha.cuda().to(device)
            self.prior_radi_beta = self.prior_radi_alpha.cuda().to(device)

    def forward(self, obs, state, radi_prev, K, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()
        stat1, stat2, stat3 = data_to_stats(obs, state)
        nss1 = torch.cat((stat1[:,:,0,:], stat2[:,:,0,:], stat3[:,:,0,:], radi_prev[:,:,0,:]), -1)
        nss2 = torch.cat((stat1[:,:,1,:], stat2[:,:,1,:], stat3[:,:,1,:], radi_prev[:,:,1,:]), -1)
        nss3 = torch.cat((stat1[:,:,2,:], stat2[:,:,2,:], stat3[:,:,2,:], radi_prev[:,:,2,:]), -1)
        q_radi_alpha = torch.cat((self.radi_log_alpha(nss1).exp().unsqueeze(-2), self.radi_log_alpha(nss2).exp().unsqueeze(-2), self.radi_log_alpha(nss3).exp().unsqueeze(-2)), -2)
        q_radi_beta = torch.cat((self.radi_log_beta(nss1).exp().unsqueeze(-2), self.radi_log_beta(nss2).exp().unsqueeze(-2), self.radi_log_beta(nss3).exp().unsqueeze(-2)), -2)

        if self.Reparameterized:
            q.gamma(q_radi_alpha,
                    q_radi_beta,
                    name='radi')
        else:
            radis = Gamma(q_radi_alpha, q_radi_beta).sample()
            q.gamma(q_radi_alpha,
                    q_radi_beta,
                    value=radis,
                    name='radi')

        p.gamma(self.prior_radi_alpha,
                self.prior_radi_beta,
                value=q['radi'],
                name='radi')
        radi = q['radi'].value
        nss_radi1 = torch.cat((stat1[:,:,0,:], stat2[:,:,0,:], stat3[:,:,0,:], radi[:, :, 0, :]), -1)
        nss_radi2 = torch.cat((stat1[:,:,1,:], stat2[:,:,1,:], stat3[:,:,1,:], radi[:, :, 1, :]), -1)
        nss_radi3 = torch.cat((stat1[:,:,2,:], stat2[:,:,2,:], stat3[:,:,2,:], radi[:, :, 2, :]), -1)
        
        q_mean_mu = torch.cat((self.mean_mu(nss_radi1).unsqueeze(-2), self.mean_mu(nss_radi2).unsqueeze(-2),self.mean_mu(nss_radi3).unsqueeze(-2)), -2)
        q_mean_sigma = torch.cat((self.mean_log_sigma(nss_radi1).exp().unsqueeze(-2),self.mean_log_sigma(nss_radi2).exp().unsqueeze(-2),self.mean_log_sigma(nss_radi3).exp().unsqueeze(-2)), -2)
        if self.Reparameterized:
            q.normal(q_mean_mu,
                     q_mean_sigma,
                     name='means')
        else:
            means = Normal(q_mean_mu, q_mean_sigma).sample()
            q.normal(q_mean_mu,
                     q_mean_sigma,
                     value=means,
                     name='means')

        p.normal(self.prior_mean_mu,
                 self.prior_mean_sigma,
                 value=q['means'],
                 name='means')

        return q, p
