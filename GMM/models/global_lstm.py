import numpy as np
import torch
import torch.nn as nn
from collections.abc import Iterable
from normal_gamma import *
import probtorch

class LSTM_eta(nn.Module):

    def __init__(self, K, D, B, S, H, L, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()

        self.Reparameterized = Reparameterized

        self.lstm = nn.LSTM(D, H, L)
        if CUDA:
            self.hidden = (torch.zeros(L, B*S, H).cuda().to(device),
                           torch.zeros(L, B*S, H).cuda().to(device))
        else:
            self.hidden = (torch.zeros(L, B*S, H),
                           torch.zeros(L, B*S, H))

        self.gamma = nn.Sequential(
            nn.Linear(H, K),
            nn.Softmax(-1))

        self.ob = nn.Sequential(
            nn.Linear(H, D))

        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.3
        self.prior_alpha = torch.ones((K, D)) * 4
        self.prior_beta = torch.ones((K, D)) * 4
        if CUDA:
            self.prior_mu = self.prior_mu.cuda().to(device)
            self.prior_nu = self.prior_nu.cuda().to(device)
            self.prior_alpha = self.prior_alpha.cuda().to(device)
            self.prior_beta = self.prior_beta.cuda().to(device)

    def forward(self, obs, K, D, batch_first=True):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, T, D = obs.shape
        in_seqs = obs.reshape(S*B, T, D).transpose(0, 1)
        out_seqs, _ = self.lstm(in_seqs, self.hidden)
        out_seqs = out_seqs.transpose(0, 1).reshape(S, B, T, -1)
        # out_last = out_seqs[:, :, T-1, :]
        # Computing sufficient stats
        gammas = self.gamma(out_seqs)
        xs = self.ob(out_seqs)
        #Computing true post params
        q_alpha, q_beta, q_mu, q_nu = Post_eta(xs, gammas, self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu, K, D)
        if self.Reparameterized:
            q.gamma(q_alpha,
                    q_beta,
                    name='precisions')
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     name='means')
        else:
            precisions = Gamma(q_alpha, q_beta).sample()
            q.gamma(q_alpha,
                    q_beta,
                    value=precisions,
                    name='precisions')
            means = Normal(q_mu, 1. / (q_nu * q['precisions'].value).sqrt()).sample()
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     value=means,
                     name='means')
        ## prior distributions
        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=q['precisions'],
                name='precisions')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * p['precisions'].value).sqrt(),
                 value=q['means'],
                 name='means')
        return q, p, q_nu
