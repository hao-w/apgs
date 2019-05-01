import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def shuffler(data):
    DIM1, DIM2, DIM3 = data.shape
    indices = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)])
    indices_expand = indices.unsqueeze(-1).repeat(1, 1, DIM3)
    return torch.gather(data, 1, indices_expand)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1e-2)

def resample_eta(obs_mu, obs_sigma, weights):
    """
    weights is S * B
    """
    S, B, K, D = obs_mu.shape
    ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, D) ## S * B * K * D
    obs_mu_r = torch.gather(obs_mu, 0, ancesters)
    obs_sigma_r = torch.gather(obs_sigma, 0, ancesters)
    return obs_mu_r, obs_sigma_r

def resample_states(states, weights):
    """
    weights is S * B
    """
    S, B, N, K = states.shape
    ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, N, K) ## S * B * N * K
    states_r = torch.gather(states, 0, ancesters)
    return states_r

def Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=False):
    """
    cluster_flag = False : return S * B * N
    cluster_flag = True, return S * B * K
    """
    labels = states.argmax(-1)
    labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    obs_mu_expand = torch.gather(obs_mu, 2, labels_flat)
    obs_sigma_expand = torch.gather(obs_sigma, 2, labels_flat)
    log_obs = Normal(obs_mu_expand, obs_sigma_expand).log_prob(obs).sum(-1) # S * B * N
    if cluster_flag:
        log_obs = torch.cat([((labels==k).float() * log_obs).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    return log_obs
