import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Oneshot_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device):
        super(self.__class__, self).__init__()

        self.neural_stats = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*K*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K*D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*K*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K*D))

        self.prior_mean_mu = torch.zeros(K*D)
        self.prior_mean_sigma = torch.ones(K*D) * 5.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)

    def forward(self, obs, K, D, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()

        neural_stats = self.neural_stats(obs)
        mean_stats = neural_stats.mean(-2)  # S * B * STAT_DIM

        stat_mu = torch.cat((self.prior_mean_mu.repeat(sample_size, batch_size, 1), self.prior_mean_sigma.repeat(sample_size, batch_size, 1), mean_stats), -1)

        q_mean_mu = self.mean_mu(stat_mu).view(sample_size, batch_size, K, D)
        q_mean_sigma = self.mean_log_sigma(stat_mu).exp().view(sample_size, batch_size, K, D)


        q.normal(q_mean_mu,
                 q_mean_sigma,
                 name='means')
        
        p.normal(self.prior_mean_mu.view(K, D),
                 self.prior_mean_sigma.view(K, D),
                 value=q['means'],
                 name='means')
        return q, p
    
    def sample_prior(self, sample_size, batch_size):
        p_mu = Normal(self.prior_mean_mu, self.prior_mean_sigma)
        obs_mu = p_mu.sample((sample_size, batch_size,))
        return obs_mu
    
class Enc_mu(nn.Module):
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
        self.prior_mean_sigma = torch.ones((K, D)) * 5.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)

    def forward(self, obs, state, K, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()

        neural_stats = self.neural_stats(torch.cat((obs, state), -1))
        _, _, _, stat_size = neural_stats.shape
        cluster_size = state.sum(-2)
        cluster_size[cluster_size == 0.0] = 1.0 # S * B * K
        neural_stats_expand = neural_stats.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2) ## S * B * N * K * STAT_SIZE
        states_expand = state.unsqueeze(-1).repeat(1, 1, 1, 1, stat_size) ## S * B * N * K * STAT_SIZE
        sum_stats = (states_expand * neural_stats_expand).sum(2) ## S * B * K * STAT_SIZE
        mean_stats = sum_stats / cluster_size.unsqueeze(-1)

        stat_mu1 = torch.cat((self.prior_mean_mu[0].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[0].repeat(sample_size, batch_size, 1), mean_stats[:,:,0,:]), -1)
        stat_mu2 = torch.cat((self.prior_mean_mu[1].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[1].repeat(sample_size, batch_size, 1), mean_stats[:,:,1,:]), -1)
        stat_mu3 = torch.cat((self.prior_mean_mu[2].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[2].repeat(sample_size, batch_size, 1), mean_stats[:,:,2,:]), -1)

        q_mean_mu = torch.cat((self.mean_mu(stat_mu1).unsqueeze(-2), self.mean_mu(stat_mu2).unsqueeze(-2), self.mean_mu(stat_mu3).unsqueeze(-2)), -2)
        q_mean_sigma = torch.cat((self.mean_log_sigma(stat_mu1).exp().unsqueeze(-2), self.mean_log_sigma(stat_mu2).exp().unsqueeze(-2), self.mean_log_sigma(stat_mu3).exp().unsqueeze(-2)), -2)

        q.normal(q_mean_mu,
                 q_mean_sigma,
                 name='means')
        p.normal(self.prior_mean_mu,
                 self.prior_mean_sigma,
                 value=q['means'],
                 name='means')
        return q, p
    
    def sample_prior(self, sample_size, batch_size):
        p_mu = Normal(self.prior_mean_mu, self.prior_mean_sigma)
        obs_mu = p_mu.sample((sample_size, batch_size,))
        return obs_mu
    
class Enc_z(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.log_prob = nn.Sequential(
            nn.Linear(2*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, obs, obs_mu, obs_rad, N, sample_size, batch_size, noise_sigma, device):
        q = probtorch.Trace()
        p = probtorch.Trace()
        # noise_sigmas = torch.ones((sample_size, batch_size, N, 1)).cuda().to(device) * noise_sigma
        obs_rads = torch.ones((sample_size, batch_size, N, 1)).cuda().to(device) * obs_rad
        prob1 = self.log_prob(torch.cat((obs, obs_mu[:, :, 0, :].unsqueeze(-2).repeat(1,1,N,1), obs_rads), -1))
        prob2 = self.log_prob(torch.cat((obs, obs_mu[:, :, 1, :].unsqueeze(-2).repeat(1,1,N,1), obs_rads), -1))
        prob3 = self.log_prob(torch.cat((obs, obs_mu[:, :, 2, :].unsqueeze(-2).repeat(1,1,N,1), obs_rads), -1))

        probs = torch.cat((prob1, prob2, prob3), -1) # S * B * N * K
        q_pi = F.softmax(probs, -1)
        z = cat(q_pi).sample()

        _ = q.variable(cat, probs=q_pi, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p
    def sample_prior(self, N, sample_size, batch_size):
        p_init_z = cat(self.prior_pi)
        state = p_init_z.sample((sample_size, batch_size, N,))
        return state
    
class Gibbs_z():
    """
    Gibbs sampling for p(z | mu, tau, x) given mu, tau, x
    """
    def __init__(self, K, CUDA, device):

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)
            
    def forward(self, obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size):
        obs_mu_expand = obs_mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
        obs_expand = obs.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
        distance = ((obs_expand - obs_mu_expand)**2).sum(-1).sqrt()
        obs_dist = Normal(obs_rad,  noise_sigma)
        log_distance = (obs_dist.log_prob(distance) - (2*math.pi*distance).log()).transpose(-1, -2) + self.prior_pi.log() # S * B * N * K   

        q_pi = F.softmax(log_distance, -1)
        q = probtorch.Trace()
        p = probtorch.Trace()
        z = cat(q_pi).sample()
        _ = q.variable(cat, probs=q_pi, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p

    def sample_prior(self, N, sample_size, batch_size):
        p_init_z = cat(self.prior_pi)
        state = p_init_z.sample((sample_size, batch_size, N,))
        return state