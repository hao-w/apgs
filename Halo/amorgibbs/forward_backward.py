import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch

def Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_mu_prev, obs_rad, noise_sigma, DETACH):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu + tau).
    """
    log_p_eta = p_eta['means'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    log_obs = True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=True, fixed_radius=True)
    log_ratio_forward = log_obs + log_p_eta - log_q_eta
    ## backward
    log_p_eta_prev = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1)
    log_q_eta_prev = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1)
    log_obs_prev = True_Log_likelihood(obs, state, obs_mu_prev, obs_rad, noise_sigma, K, D, cluster_flag=True, fixed_radius=True)
    log_ratio_backward = log_obs_prev + log_p_eta_prev - log_q_eta_prev
    log_weights_eta = log_ratio_forward - log_ratio_backward
    weights_eta = F.softmax(log_weights_eta, 0)
    if DETACH:
        weights_eta = weights_eta.detach()
    eubo_eta = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
    elbo_eta = log_weights_eta.sum(-1).mean(0).mean()
    symkl = eubo_eta - elbo_eta
    ess_eta = (1. / (weights_eta**2).sum(0)).mean()
    return symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_mu
