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

def resample_eta(obs_mu, obs_rad, weights):
    """
    weights is S * B * K
    """
    S, B, K, D = obs_mu.shape
    ancesters_mu = Categorical(weights.transpose(0,1).transpose(1,2)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
    ancesters_rad = Categorical(weights.transpose(0,1).transpose(1,2)).sample((S, )).unsqueeze(-1) ## S * B * K * 1
    obs_mu_r = torch.gather(obs_mu, 0, ancesters_mu)
    obs_rad_r = torch.gather(obs_rad, 0, ancesters_rad)
    return obs_mu_r, obs_rad_r

def resample_mu(obs_mu, weights):
    """
    weights is S * B * K
    """
    S, B, K, D = obs_mu.shape
    ancesters = Categorical(weights.transpose(0,1).transpose(1,2)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
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

def True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=False, fixed_radius=True):
    """
    cluster_flag = False : return S * B * N
    cluster_flag = True, return S * B * K
    """
    labels = state.argmax(-1)
    labels_mu = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    # labels_rad = labels.unsqueeze(-1)
    obs_mu_expand = torch.gather(obs_mu, 2, labels_mu)
    distance = ((obs - obs_mu_expand)**2).sum(-1).sqrt()
    if fixed_radius:
        obs_dist = Normal(obs_rad, noise_sigma)
    else:
        obs_dist = Normal(torch.gather(obs_rad.squeeze(-1), 2, labels), noise_sigma)
    log_distance = obs_dist.log_prob(distance) - (2*math.pi*distance).log()
    if cluster_flag:
        log_distance = torch.cat([((labels==k).float() * log_distance).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    return log_distance
