import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
import probtorch
import math
from utils import *

class Oneshot_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device):
        super(self.__class__, self).__init__()
        self.neural_stats = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.gammas = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K),
            nn.Softmax(-1))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.prior_mu_mu = torch.zeros(D)
        self.prior_mu_sigma = torch.ones(D) * 10.0
        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu_mu = self.prior_mu_mu.cuda()
                self.prior_mu_sigma = self.prior_mu_sigma.cuda()
    def forward(self, ob, K, sampled=True, mu_old=None):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, N, D = ob.shape
        ss = self.neural_stats(ob)
        gammas = self.gammas(ob)
        nss = ss_to_stats(ss, gammas) # S * B * K * STAT_DIM
        nss_prior = torch.cat((nss, self.prior_mu_mu.repeat(S, B, K, 1), self.prior_mu_sigma.repeat(S, B, K, 1)), -1)
        q_mu_mu= self.mean_mu(nss_prior)
        q_mu_sigma = self.mean_log_sigma(nss_prior).exp()
        if sampled:
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
