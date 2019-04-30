import torch.nn as nn
import torch
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch import logsumexp
import math

def shuffler(data):
    DIM1, DIM2, DIM3 = data.shape
    indices = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)])
    indices_expand = indices.unsqueeze(-1).repeat(1, 1, DIM3)
    return torch.gather(data, 1, indices_expand)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1e-3)

from torch.distributions.categorical import Categorical

def resample_mu(obs_mu, weights):
    """
    weights is S * B * K
    """
    S, B, K, D = obs_mu.shape
    ancesters = Categorical(weights.transpose(0,1).transpose(1,2)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
    obs_mu_r = torch.gather(obs_mu, 0, ancesters)
    return obs_mu_r

def resample_states(states, weights):
    """
    weights is S * B * N
    """
    S, B, N, K = states.shape
    ancesters = Categorical(weights.transpose(0,1).transpose(1,2)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, K) ## S * B * N * K
    states_r = torch.gather(states, 0, ancesters)
    return states_r

def True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma, gpu, cluster_flag=False):
    """
    cluster_flag = False : return S * B * N
    cluster_flag = True, return S * B * K
    """
    labels = states.argmax(-1)
    labels_mu = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    # labels_rad = labels.unsqueeze(-1)
    obs_mu_expand = torch.gather(obs_mu, 2, labels_mu)
    obs_rad_expand = torch.gather(obs_rad.squeeze(-1), 2, labels)
    distance = ((obs - obs_mu_expand)**2).sum(-1).sqrt()
    obs_dist = Normal(obs_rad_expand, torch.ones(1).cuda().to(gpu) * noise_sigma)
    log_distance = obs_dist.log_prob(distance) - (2*math.pi*distance).log()
    if cluster_flag:
        log_distance = torch.cat([((labels==k).float() * log_distance).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    return log_distance

def sample_single_batch(num_seqs, Xs, sample_size, batch_size, gpu):
    indices = torch.randperm(num_seqs)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Xs[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs = obs.cuda().to(gpu)
    return obs

def test(enc_mu, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, gpu):
    p_init_z = cat(enc_z.prior_pi)
    states = p_init_z.sample((sample_size, batch_size, N,))
    log_p_z = p_init_z.log_prob(states)## S * B * N
    log_q_z = p_init_z.log_prob(states)
    for m in range(mcmc_size):
        q_mu, p_mu = enc_mu(obs, states, sample_size, batch_size)
        log_q_mu = q_mu['means'].log_prob.sum(-1)
        log_p_mu = p_mu['means'].log_prob.sum(-1) # S * B * K
        obs_mu = q_mu['means'].value
        log_obs_k = Log_likelihood(obs, states, obs_mu, K, D, radius=1.5, noise_sigma = 0.05, gpu=gpu, cluster_flag=True)
        log_weights_global = log_obs_k + log_p_mu - log_q_mu
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## resample mu
        obs_mu = resample_mu(obs_mu, weights_global)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_mu, 1.5, 0.05, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob ## S * B * N
        states = q_z['zs'].value
        log_obs_n = Log_likelihood(obs, states, obs_mu, K, D, radius=1.5, noise_sigma = 0.05, gpu=gpu, cluster_flag=False)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()

    return q_mu, q_z
