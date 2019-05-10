import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch

def Init_step_eta(models, obs, N, K, D, sample_size, batch_size, prior_flag):
    """
    initialize eta, using oneshot encoder, and then update z using its (gibbs or neural gibbs) encoder
    return the samples and log_weights
    """
    if prior_flag:
        (enc_eta, enc_z) = models
        obs_mu, obs_tau = enc_eta.sample_prior(sample_size, batch_size)
    else:
        (oneshot_eta, enc_eta, enc_z) = models
        q_eta, p_eta, q_nu = oneshot_eta(obs, K, D)
        log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
        log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
    q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
    if prior_flag:
        log_weights = log_obs_n + log_p_z - log_q_z
    else:
        log_weights = log_obs_n.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
    return obs_tau, obs_mu, state, log_weights

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
