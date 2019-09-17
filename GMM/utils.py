import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from normal_gamma import *
import probtorch

def shuffler(data):
    DIM1, DIM2, DIM3 = data.shape
    indices = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)])
    indices_expand = indices.unsqueeze(-1).repeat(1, 1, DIM3)
    return torch.gather(data, 1, indices_expand)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1e-3)

def Resample(var, weights, idw_flag=True):
    dim1, dim2, dim3, dim4 = var.shape
    if idw_flag:
        if dim2 == 1:
            ancesters = Categorical(weights.permute(1, 2, 0).squeeze(0)).sample((dim1, )).unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, dim4)
        else:
            ancesters = Categorical(weights.permute(1, 2, 0)).sample((dim1, )).unsqueeze(-1).repeat(1, 1, 1, dim4)
    else:
        ancesters = Categorical(weights.transpose(0, 1)).sample((dim1, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4) ## S * B * N * K
    return torch.gather(var, 0, ancesters)
# def resample_eta(obs_mu, obs_sigma, weights, idw_flag=True):
#     S, B, K, D = obs_mu.shape
#     if idw_flag: ## individual importance weight S * B * K
#         ancesters = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, D)
#         obs_mu_r = torch.gather(obs_mu, 0, ancesters)
#         obs_sigma_r = torch.gather(obs_sigma, 0, ancesters)
#     else: ## joint importance weight S * B
#         ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, D)
#         obs_mu_r = torch.gather(obs_mu, 0, ancesters)
#         obs_sigma_r = torch.gather(obs_sigma, 0, ancesters)
#     return obs_mu_r, obs_sigma_r
#
#
# def resample_state(state, weights, idw_flag=True):
#     S, B, N, K = state.shape
#     if idw_flag: ## individual importance weight S * B * K
#         ancesters = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, K) ## S * B * N * K
#         state_r = torch.gather(state, 0, ancesters)
#     else: ## joint importance weight S * B
#         ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, N, K) ## S * B * N * K
#         state_r = torch.gather(state, 0, ancesters)
#     return state_r

def Log_likelihood(ob, state, ob_tau, ob_mu, cluster_flag=False):
    """
    cluster_flag = False : return S * B * N
    cluster_flag = True, return S * B * K
    """
    ob_sigma = 1. / ob_tau.sqrt()
    labels = state.argmax(-1)
    labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, ob.shape[-1])
    ob_mu_expand = torch.gather(ob_mu, 2, labels_flat)
    ob_sigma_expand = torch.gather(ob_sigma, 2, labels_flat)
    log_ob = Normal(ob_mu_expand, ob_sigma_expand).log_prob(ob).sum(-1) # S * B * N
    if cluster_flag:
        log_ob = torch.cat([((labels==k).float() * log_ob).sum(-1).unsqueeze(-1) for k in range(state.shape[-1])], -1) # S * B * K
    return log_ob
