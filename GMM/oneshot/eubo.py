import torch
import torch.nn as nn
from utils import *
from normal_gamma_kls import *
from normal_gamma_conjugacy import *
import probtorch

def Eubo_os(enc_eta, enc_z, obs, N, K, D, sample_size, batch_size, device, idw_flag=True):
    """
    oneshot encoder using exact same architecture from amor-gibbs
    initialize eta
    """
    ## update tau and mu -- global variables
    q_eta, p_eta, q_nu = enc_eta(obs, K, D)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    obs_sigma = 1. / obs_tau.sqrt()
    ## update z -- cluster assignments
    q_z, p_z = enc_z(obs, obs_sigma, obs_mu, N, sample_size, batch_size)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(obs, state, obs_mu, obs_sigma, K, D, cluster_flag=False)
    log_weights = log_obs_n.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1)
    weights = F.softmax(log_weights, 0).detach()
    ## EUBO, ELBO, ESS
    eubo = (weights * log_weights).sum(0).mean()
    elbo = log_weights.mean()
    ess = (1. / (weights**2).sum(0)).mean()
    return eubo, elbo, ess, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_os_cfz(enc_eta, gibbs_z, obs, N, K, D, sample_size, batch_size, device, idw_flag=True):
    """
    oneshot encoder using exact same architecture from amor-gibbs
    initialize eta
    """
    ## update tau and mu -- global variables
    q_eta, p_eta, q_nu = enc_eta(obs, K, D)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    obs_sigma = 1. / obs_tau.sqrt()
    ## update z -- cluster assignments
    q_z, p_z = gibbs_z.forward(obs, obs_sigma, obs_mu, N, K)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(obs, state, obs_mu, obs_sigma, K, D, cluster_flag=False)
    log_weights = log_obs_n.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1)
    weights = F.softmax(log_weights, 0).detach()
    ## EUBO, ELBO, ESS
    eubo = (weights * log_weights).sum(0).mean()
    elbo = log_weights.mean()
    ess = (1. / (weights**2).sum(0)).mean()
    return eubo, elbo, ess, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu
