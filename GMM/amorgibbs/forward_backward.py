import torch
import torch.nn as nn
from utils import *
from normal_gamma_kls import *
from normal_gamma_conjugacy import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch

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
    log_ratio_forward = log_obs + log_p_eta - log_q_eta
    ## backward
    log_p_eta_prev = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(p_eta['precisions'].dist.concentration, p_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1)
    log_q_eta_prev = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(q_eta['precisions'].dist.concentration, q_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1)
    log_obs_prev = Log_likelihood(obs, state, obs_tau_prev, obs_mu_prev, K, D, cluster_flag=True)
    log_ratio_backward = log_obs_prev + log_p_eta_prev - log_q_eta_prev
    log_weights_eta = log_ratio_forward - log_ratio_backward
    weights_eta = F.softmax(log_weights_eta, 0).detach()
    eubo_eta = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
    elbo_eta = log_weights_eta.sum(-1).mean(0).mean()
    symkl = eubo_eta - elbo_eta
    ess_eta = (1. / (weights_eta**2).sum(0)).mean()
    return symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_mu, obs_tau


def Incremental_forward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D):
    """
    Given the current samples for global variables,
    sample new local variable.
    """

    state = q_z['zs'].value ## S * B * N * K
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    log_obs_n = Log_likelihood(obs, state, obs_mu, 1. / obs_tau.sqrt(), K, D, cluster_flag=False)
    log_weights_state = log_obs_n + log_p_z - log_q_z
    weights_state = F.softmax(log_weights_state, 0).detach()
    eubo_state = (weights_state * log_weights_state).sum(0).sum(-1).mean()
    elbo_state = log_weights_state.sum(-1).mean()
    ess_state = (1. / (weights_state**2).sum(0)).mean()
    return eubo_state, elbo_state, ess_state, weights_state

def Incremental_backward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D, state):
    """
    Given the current samples for global variables,
    sample new local variable.
    """
    log_p_z = cat(p_z['zs'].dist.probs).log_prob(state)
    log_q_z = cat(q_z['zs'].dist.probs).log_prob(state)
    log_obs_n = Log_likelihood(obs, state, obs_mu, 1. / obs_tau.sqrt(), K, D, cluster_flag=False)
    log_weights_state = log_obs_n + log_p_z - log_q_z
    weights_state = F.softmax(log_weights_state, 0).detach()
    eubo_state = (weights_state * log_weights_state).sum(0).sum(-1).mean()
    elbo_state = log_weights_state.sum(-1).mean()
    ess_state = (1. / (weights_state**2).sum(0)).mean()
    return eubo_state, elbo_state, ess_state, weights_state
