import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch

def Init_step_eta(obs, N, K, D, oneshot_eta, enc_z):
    """
    initialize eta, using oneshot encoder, and then update z using its (gibbs or neural gibbs) encoder
    return the samples and log_weights
    """
    q_eta, p_eta, q_nu = oneshot_eta(obs, K, D)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
    log_weights_state = log_obs_n.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
    return obs_tau, obs_mu, state, log_weights_state

def Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau_prev, obs_mu_prev):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu + tau).
    """
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    log_obs = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=True)
    log_w_forward = log_obs + log_p_eta - log_q_eta
    ## backward
    log_p_eta_prev = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(p_eta['precisions'].dist.concentration, p_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1)
    log_q_eta_prev = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(q_eta['precisions'].dist.concentration, q_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1)
    log_obs_prev = Log_likelihood(obs, state, obs_tau_prev, obs_mu_prev, K, D, cluster_flag=True)
    log_w_backward = log_obs_prev + log_p_eta_prev - log_q_eta_prev
    return obs_tau, obs_mu, log_w_forward, log_w_backward

def Incremental_eta_ep(q_eta, p_eta, obs, state, K, D, obs_tau_prev, obs_mu_prev, DETACH):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu + tau).
    """
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    log_obs = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=True)
    log_ratio_forward = log_obs + log_p_eta - log_q_eta
    ## backward
    log_p_eta_prev = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(p_eta['precisions'].dist.concentration, p_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1)
    log_q_eta_prev = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(q_eta['precisions'].dist.concentration, q_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1)
    log_obs_prev = Log_likelihood(obs, state, obs_tau_prev, obs_mu_prev, K, D, cluster_flag=True)
    log_ratio_backward = log_obs_prev + log_p_eta_prev - log_q_eta_prev
    log_weights_eta = log_ratio_forward - log_ratio_backward
    weights_eta = F.softmax(log_weights_eta, 0)
    if DETACH:
        weights_eta = weights_eta.detach()
    eubo_eta = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
    elbo_eta = log_weights_eta.sum(-1).mean(0).mean()
    symkl = eubo_eta - elbo_eta
    ess_eta = (1. / (weights_eta**2).sum(0)).mean()
    return symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_tau, obs_mu
