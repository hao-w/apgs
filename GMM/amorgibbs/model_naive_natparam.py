%run ../../path_import.py
import torch
import torch.nn as nn
import probtorch
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from normal_gamma_conjugacy import *

class Enc_eta(nn.Module):
    def __init__(self, K, D, CUDA, device):
        super(self.__class__, self).__init__()

        self.gamma = nn.Sequential(
            nn.Linear(K+D, K),
            nn.Softmax(-1))

        self.ob = nn.Sequential(
            nn.Linear(K+D, D))

        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.3
        self.prior_alpha = torch.ones((K, D)) * 4
        self.prior_beta = torch.ones((K, D)) * 4
        if CUDA:
            self.prior_mu = self.prior_mu.cuda().to(device)
            self.prior_nu = self.prior_nu.cuda().to(device)
            self.prior_alpha = self.prior_alpha.cuda().to(device)
            self.prior_beta = self.prior_beta.cuda().to(device)

    def forward(self, data, K, D):
        q = probtorch.Trace()
        p = probtorch.Trace()

        gammas = self.gamma(data) # S * B * N * K --> S * B * N * K
        xs = self.ob(data)  # S * B * N * D --> S * B * N * D
        q_alpha, q_beta, q_mu, q_nu = Post_eta(xs, gammas,
                                                 self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu, K, D)
        precisions = Gamma(q_alpha, q_beta).sample()
        q.gamma(q_alpha,
                q_beta,
                value=precisions,
                name='precisions')
        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=q['precisions'],
                name='precisions')
        means = Normal(q_mu, 1. / (q_nu * q['precisions'].value).sqrt()).sample()
        q.normal(q_mu,
                 1. / (q_nu * q['precisions'].value).sqrt(),
                 value=means,
                 name='means')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * q['precisions'].value).sqrt(),
                 value=q['means'],
                 name='means')
        return q, p, q_nu

class Enc_z(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(3*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, obs, obs_tau, obs_mu, N, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()

        data_c1 = torch.cat((obs, obs_mu[:, :, 0, :].unsqueeze(-2).repeat(1,1,N,1), obs_tau[:, :, 0, :].unsqueeze(-2).repeat(1,1,N,1)), -1) ## S * B * N * 3D
        data_c2 = torch.cat((obs, obs_mu[:, :, 1, :].unsqueeze(-2).repeat(1,1,N,1), obs_tau[:, :, 1, :].unsqueeze(-2).repeat(1,1,N,1)), -1) ## S * B * N * 3D
        data_c3 = torch.cat((obs, obs_mu[:, :, 2, :].unsqueeze(-2).repeat(1,1,N,1), obs_tau[:, :, 2, :].unsqueeze(-2).repeat(1,1,N,1)), -1) ## S * B * N * 3D

        log_prob_c1 = self.pi_log_prob(data_c1)
        log_prob_c2 = self.pi_log_prob(data_c2)
        log_prob_c3 = self.pi_log_prob(data_c3)

        q_probs = F.softmax(torch.cat((log_prob_c1, log_prob_c2, log_prob_c3), -1), -1)
        z = cat(q_probs).sample()
        _ = q.variable(cat, probs=q_probs, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p

def initialize(K, D, num_hidden_local, CUDA, device, LR):
    enc_eta = Enc_eta(K, D, CUDA, device)
    enc_z = Enc_z(K, D, num_hidden_local, CUDA, device)
    if CUDA:
        enc_eta.cuda().to(device)
        enc_z.cuda().to(device)
    optimizer =  torch.optim.Adam(list(enc_z.parameters())+list(enc_eta.parameters()),lr=LR, betas=(0.9, 0.99))
    return enc_eta, enc_z, optimizer
