import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
from utils import *

class Enc_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()
        self.Reparameterized = Reparameterized

        self.neural_stats = nn.Sequential(
            nn.Linear(K+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))
        
        self.gammas = nn.Sequential(
            nn.Linear(D+K, K),
            nn.Softmax(-1))
        
        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.prior_mean_mu = torch.zeros((K, D))
        self.prior_mean_sigma = torch.ones((K, D)) * 7.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)

    def forward(self, obs, state, K, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()
        D = obs.shape[-1]
        ss = self.neural_stats(torch.cat((obs, state), -1))
        gammas = self.gammas(torch.cat((obs, state), -1))
        nss = ss_to_stats(ss, gammas) # S * B * K * STAT_DIM
        
        mus = []
        sigmas = []
        for k in range(K):
            mu_k = self.mean_mu(torch.cat((nss[:,:,k,:], self.prior_mean_mu[k].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[k].repeat(sample_size, batch_size, 1)), -1))
            sigma_k = self.mean_log_sigma(torch.cat((nss[:,:,k,:], self.prior_mean_mu[k].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[k].repeat(sample_size, batch_size, 1)), -1)).exp()
            mus.append(mu_k.unsqueeze(-2))
            sigmas.append(sigma_k.unsqueeze(-2))
            
            
        q_mean_mu = torch.cat(mus, -2)
        q_mean_sigma = torch.cat(sigmas, -2)
        
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

