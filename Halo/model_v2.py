import sys
sys.path.append("../")
sys.path.append('/home/hao/Research/probtorch/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch

class Enc_mu_rad(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device):
        super(self.__class__, self).__init__()

        self.neural_stats = nn.Sequential(
            nn.Linear(K+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.prior_mean_mu = torch.zeros((K, D))
        self.prior_mean_sigma = torch.ones((K, D)) * 4.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)

    def forward(self, obs, states, K, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()

        neural_stats = self.neural_stats(torch.cat((obs, states), -1))
        _, _, _, stat_size = neural_stats.shape
        cluster_size = states.sum(-2)
        cluster_size[cluster_size == 0.0] = 1.0 # S * B * K
        neural_stats_expand = neural_stats.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2) ## S * B * N * K * STAT_SIZE
        states_expand = states.unsqueeze(-1).repeat(1, 1, 1, 1, stat_size) ## S * B * N * K * STAT_SIZE
        sum_stats = (states_expand * neural_stats_expand).sum(2) ## S * B * K * STAT_SIZE
        mean_stats = sum_stats / cluster_size.unsqueeze(-1)

        stat_mu1 = torch.cat((self.prior_mean_mu[0].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[0].repeat(sample_size, batch_size, 1), mean_stats[:,:,0,:]), -1)
        stat_mu2 = torch.cat((self.prior_mean_mu[1].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[1].repeat(sample_size, batch_size, 1), mean_stats[:,:,1,:]), -1)
        stat_mu3 = torch.cat((self.prior_mean_mu[2].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[2].repeat(sample_size, batch_size, 1), mean_stats[:,:,2,:]), -1)

        q_mean_mu = torch.cat((self.mean_mu(stat_mu1).unsqueeze(-2), self.mean_mu(stat_mu2).unsqueeze(-2), self.mean_mu(stat_mu3).unsqueeze(-2)), -2)
        q_mean_sigma = torch.cat((self.mean_log_sigma(stat_mu1).exp().unsqueeze(-2), self.mean_log_sigma(stat_mu2).exp().unsqueeze(-2), self.mean_log_sigma(stat_mu3).exp().unsqueeze(-2)), -2)

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

class Enc_z(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.log_prob = nn.Sequential(
            nn.Linear(2*D+2, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, obs, obs_mu, obs_rad, N, sample_size, batch_size, noise_sigma, device):
        q = probtorch.Trace()
        p = probtorch.Trace()
        noise_sigmas = torch.ones((sample_size, batch_size, N, 1)).cuda().to(device) * noise_sigma

        prob1 = self.log_prob(torch.cat((obs, obs_mu[:, :, 0, :].unsqueeze(-2).repeat(1,1,N,1), obs_rad[:, :, 0, :].unsqueeze(-2).repeat(1,1,N,1), noise_sigmas), -1))
        prob2 = self.log_prob(torch.cat((obs, obs_mu[:, :, 1, :].unsqueeze(-2).repeat(1,1,N,1), obs_rad[:, :, 1, :].unsqueeze(-2).repeat(1,1,N,1), noise_sigmas), -1))
        prob3 = self.log_prob(torch.cat((obs, obs_mu[:, :, 2, :].unsqueeze(-2).repeat(1,1,N,1), obs_rad[:, :, 2, :].unsqueeze(-2).repeat(1,1,N,1), noise_sigmas), -1))

        probs = torch.cat((prob1, prob2, prob3), -1) # S * B * N * K
        q_pi = F.softmax(probs, -1)
        z = cat(q_pi).sample()

        _ = q.variable(cat, probs=q_pi, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p

def initialize(NUM_HIDDEN_GLOBAL, STAT_SIZE, NUM_HIDDEN_LOCAL, K, D, CUDA, DEVICE, LR):
    enc_mu_rad = Enc_mu_rad(K, D, num_hidden=NUM_HIDDEN_GLOBAL, num_stats=STAT_SIZE, CUDA=CUDA, device=DEVICE)
    enc_z = Enc_z(K, D, num_hidden=NUM_HIDDEN_LOCAL, CUDA=CUDA, device=DEVICE)
    if CUDA:
        enc_mu_rad.cuda().to(DEVICE)
        enc_z.cuda().to(DEVICE)
    optimizer =  torch.optim.Adam(list(enc_z.parameters())+list(enc_mu_rad.parameters()),lr=LR, betas=(0.9, 0.99))
    return enc_mu_rad, enc_z, optimizer
