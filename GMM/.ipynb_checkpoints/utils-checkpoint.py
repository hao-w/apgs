import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from normal_gamma_kls import *
from normal_gamma_conjugacy import *
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

def resample_eta(obs_mu, obs_sigma, weights, idw_flag=True):
    S, B, K, D = obs_mu.shape
    if idw_flag: ## individual importance weight S * B * K
        ancesters = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, D)
        obs_mu_r = torch.gather(obs_mu, 0, ancesters)
        obs_sigma_r = torch.gather(obs_sigma, 0, ancesters)
    else: ## joint importance weight S * B
        ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, D)
        obs_mu_r = torch.gather(obs_mu, 0, ancesters)
        obs_sigma_r = torch.gather(obs_sigma, 0, ancesters)
    return obs_mu_r, obs_sigma_r

def resample_state(state, weights, idw_flag=True):
    S, B, N, K = state.shape
    if idw_flag: ## individual importance weight S * B * K
        ancesters = Categorical(weights.permute(1, 2, 0)).sample((S, )).unsqueeze(-1).repeat(1, 1, 1, K) ## S * B * N * K
        state_r = torch.gather(state, 0, ancesters)
    else: ## joint importance weight S * B
        ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, N, K) ## S * B * N * K
        state_r = torch.gather(state, 0, ancesters)
    return state_r

def Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False):
    """
    cluster_flag = False : return S * B * N
    cluster_flag = True, return S * B * K
    """
    obs_sigma = 1. / obs_tau.sqrt()
    labels = state.argmax(-1)
    labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    obs_mu_expand = torch.gather(obs_mu, 2, labels_flat)
    obs_sigma_expand = torch.gather(obs_sigma, 2, labels_flat)
    log_obs = Normal(obs_mu_expand, obs_sigma_expand).log_prob(obs).sum(-1) # S * B * N
    if cluster_flag:
        log_obs = torch.cat([((labels==k).float() * log_obs).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    return log_obs

def kl_train(q_eta, p_eta, q_z, p_z, q_nu, pr_nu, obs, K):
    _, _, N, D = obs.shape
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_mu = q_eta['means'].dist.loc
    q_pi = q_z['zs'].dist.probs

    pr_alpha = p_eta['precisions'].dist.concentration
    pr_beta = p_eta['precisions'].dist.rate
    pr_mu = p_eta['means'].dist.loc
    pr_pi = p_z['zs'].dist.probs

    states = q_z['zs'].value
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value

    post_alpha, post_beta, post_mu, post_nu = Post_eta(obs, states, pr_alpha, pr_beta, pr_mu, pr_nu, K, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_tau, obs_mu, pr_pi, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_pi.log(), post_logits)
    return kl_eta_ex.sum(-1).mean(), kl_eta_in.sum(-1).mean(), kl_z_ex.sum(-1).mean(), kl_z_in.sum(-1).mean()
