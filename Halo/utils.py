import torch
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
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

def resample_eta(obs_mu, radi, weights, idw_flag=True):
    S, B, K, D = obs_mu.shape
    if idw_flag: ## individual importance weight S * B * K
        ancesters_mu = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, D)
        ancesters_radi = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1)
        obs_mu_r = torch.gather(obs_mu, 0, ancesters_mu)
        radi_r = torch.gather(radi, 0, ancesters_radi)
    else: ## joint importance weight S * B
        ancesters_mu = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, D)
        ancesters_radi = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, 1)
        obs_mu_r = torch.gather(obs_mu, 0, ancesters_mu)
        radi_r = torch.gather(radi, 0, ancesters_radi)
    return obs_mu_r, radi_r

def resample_mu(obs_mu, weights, idw_flag=True):
    S, B, K, D = obs_mu.shape
    if idw_flag: ## individual importance weight S * B * K
        ancesters = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, D)
        obs_mu_r = torch.gather(obs_mu, 0, ancesters)
    else: ## joint importance weight S * B
        ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, D)
        obs_mu_r = torch.gather(obs_mu, 0, ancesters)
    return obs_mu_r


def resample_state(state, weights, idw_flag=True):
    S, B, N, K = state.shape
    if idw_flag: ## individual importance weight S * B * K
        ancesters = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, K) ## S * B * N * K
        state_r = torch.gather(state, 0, ancesters)
    else: ## joint importance weight S * B
        ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, N, K) ## S * B * N * K
        state_r = torch.gather(state, 0, ancesters)
    return state_r

def True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=False):
    """
    cluster_flag = False : return S * B * N
    cluster_flag = True, return S * B * K
    """
    labels = state.argmax(-1)
    labels_mu = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    # labels_rad = labels.unsqueeze(-1)
    obs_mu_expand = torch.gather(obs_mu, 2, labels_mu)
    distance = ((obs - obs_mu_expand)**2).sum(-1).sqrt()
    obs_dist = Normal(obs_rad, noise_sigma)
    log_distance = obs_dist.log_prob(distance) - (2*math.pi*distance).log()
    if cluster_flag:
        log_distance = torch.cat([((labels==k).float() * log_distance).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    return log_distance

def True_Log_likelihood_rad(obs, state, obs_mu, radi, noise_sigma, K, D, cluster_flag=False):
    """
    cluster_flag = False : return S * B * N
    cluster_flag = True, return S * B * K
    """
    labels = state.argmax(-1)
    labels_mu = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    # labels_rad = labels.unsqueeze(-1)
    obs_mu_expand = torch.gather(obs_mu, 2, labels_mu)
    distance = ((obs - obs_mu_expand)**2).sum(-1).sqrt()
    obs_dist = Normal(torch.gather(radi.squeeze(-1), 2, labels), noise_sigma)
    log_distance = obs_dist.log_prob(distance) - (2*math.pi*distance).log()
    if cluster_flag:
        log_distance = torch.cat([((labels==k).float() * log_distance).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    return log_distance

def data_to_stats(obs, states, K, D):
    """
    stat1 : sum of I[z_n=k], S * B * K
    stat2 : sum of I[z_n=k]*x_n, S * B * K * D
    stat3 : sum of I[z_n=k]*x_n^2, S * B * K * D
    """
    stat1 = states.sum(2)
    states_expand = states.unsqueeze(-1).repeat(1, 1, 1, 1, D)
    obs_expand = obs.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2)
    stat2 = (states_expand * obs_expand).sum(2)
    stat3 = (states_expand * (obs_expand**2)).sum(2)
    return stat1, stat2, stat3