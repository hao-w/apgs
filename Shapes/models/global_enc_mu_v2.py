import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
from utils import *

class Enc_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device):
        super(self.__class__, self).__init__()
        self.nss = nn.Sequential(
            nn.Linear(K+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_mu = nn.Sequential(
            nn.Linear(2*D, 2*D),
            nn.Tanh(),
            nn.Linear(2*D, D))
        # self.mean_mu = nn.Sequential(
        #     nn.Linear(2*D, num_hidden),
        #     nn.Tanh(),
        #     nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(2*D, 2*D),
            nn.Tanh(),
            nn.Linear(2*D, D))

        self.prior_mu_mu = torch.zeros(D)
        self.prior_mu_sigma = torch.ones(D) * 10.0

        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu_mu = self.prior_mu_mu.cuda()
                self.prior_mu_sigma = self.prior_mu_sigma.cuda()

    def forward(self, ob, state, angle, sampled=True, mu_old=None):

        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, N, D = ob.shape
        K = state.shape[-1]
        embed = torch.cat((angle, state), -1)
        nss2 = ss_to_stats(ob - self.nss(embed), state)
        nss = ss_to_stats(ob, state)
        q_mu_mu= self.mean_mu(torch.cat((nss, nss2), -1))
        #q_mu_mu = nss
        q_mu_sigma = self.mean_log_sigma(torch.cat((nss, nss2), -1)).exp()
        if sampled == True:
            mu = Normal(q_mu_mu, q_mu_sigma).sample()
            q.normal(q_mu_mu,
                     q_mu_sigma,
                     value=mu,
                     name='means')

            p.normal(self.prior_mu_mu,
                     self.prior_mu_sigma,
                     value=q['means'],
                     name='means')
        else:
            q.normal(q_mu_mu,
                     q_mu_sigma,
                     value=mu_old,
                     name='means')

            p.normal(self.prior_mu_mu,
                     self.prior_mu_sigma,
                     value=q['means'],
                     name='means')
        return q, p

    def sample_prior(self, S, B, K):
        p_mu = Normal(self.prior_mu_mu, self.prior_mu_sigma)
        ob_mu = p_mu.sample((S, B, K,))
        log_p_mu = p_mu.log_prob(ob_mu).sum(-1)
        return ob_mu, log_p_mu
