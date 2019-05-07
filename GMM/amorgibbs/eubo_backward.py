import torch
import torch.nn as nn
from utils import *
from normal_gamma_kls import *
from normal_gamma_conjugacy import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch

def Incremental_forward_eta(q_eta, p_eta, obs, state, K, D):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu + tau).
    """
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    log_obs_k = Log_likelihood(obs, state, obs_mu, 1. / obs_tau.sqrt(), K, D, cluster_flag=True)
    log_weights_eta = log_obs_k + log_p_eta - log_q_eta
    weights_eta = F.softmax(log_weights_eta, 0).detach()
    eubo_eta = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
    elbo_eta = log_weights_eta.sum(-1).mean()
    ess_eta = (1. / (weights_eta**2).sum(0)).mean()
    return eubo_eta, elbo_eta, ess_eta, weights_eta

def Incremental_backward_eta(q_eta, p_eta, obs, state, K, D, obs_mu, obs_tau):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu + tau).
    """
    log_p_eta = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu).sum(-1) + Gamma(p_eta['precisions'].dist.concentration, p_eta['precisions'].dist.rate).log_prob(obs_tau).sum(-1)
    log_q_eta = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu).sum(-1) + Gamma(q_eta['precisions'].dist.concentration, q_eta['precisions'].dist.rate).log_prob(obs_tau).sum(-1)
    log_obs_k = Log_likelihood(obs, state, obs_mu, 1. / obs_tau.sqrt(), K, D, cluster_flag=True)
    log_weights_eta = log_obs_k + log_p_eta - log_q_eta
    weights_eta = F.softmax(log_weights_eta, 0).detach()
    eubo_eta = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
    elbo_eta = log_weights_eta.sum(-1).mean()
    ess_eta = (1. / (weights_eta**2).sum(0)).mean()
    return eubo_eta, elbo_eta, ess_eta, weights_eta

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

# def Eubo_mcmc_init_z(enc_eta, enc_z, obs, K, mcmc_size, device, RESAMPLE=False):
#     """
#     EUBO for amortized gibbs with backward transition,
#     individually compute importance weights
#     """
#     sample_size, batch_size, N, D  = obs.shape
#     eubos = torch.zeros(2*mcmc_size+1).cuda().to(device)
#     elbos = torch.zeros(2*mcmc_size+1).cuda().to(device)
#     esss = torch.zeros(2*mcmc_size+1).cuda().to(device)
#     ## sample from prior and finish one full update
#     state = enc_z.sample_prior(N, sample_size, batch_size)
#     q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
#     eubo_eta, elbo_eta, ess_eta, weights_eta = Incremental_forward_eta(q_eta, p_eta, obs, state, K, D)
#     eubos[0] = eubo_eta
#     elbos[0] = elbo_eta
#     esss[0] = ess_eta
#     for m in range(mcmc_size):
#         if RESAMPLE:
#             obs_mu, obs_tau = resample_eta(q_eta['means'].value, q_eta['precisions'].value, weights_eta, idw_flag=True)
#         else:
#             obs_mu = q_eta['means'].value
#             obs_tau = q_eta['precisions'].value
#         q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
#         eubo_state_forward, elbo_state_forward, ess_state_forward, weights_state = Incremental_forward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D)
#         eubo_state_backward, elbo_state_backward, ess_state_backward, _ = Incremental_backward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D, state=state)
#         eubos[1+2*m] = eubo_state_forward - eubo_state_backward
#         elbos[1+2*m] = elbo_state_forward - elbo_state_backward
#         esss[1+2*m] = (ess_state_forward + ess_state_backward) / 2
#         if RESAMPLE:
#             state = resample_state(q_z['zs'].value, weights_state, idw_flag=True)
#         else:
#             state = q_z['zs'].value
#         q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
#         eubo_eta_forward, elbo_eta_forward, ess_eta_forward, weights_eta  = Incremental_forward_eta(q_eta, p_eta, obs, state, K, D)
#         eubo_eta_backward, elbo_eta_backward, ess_eta_backward, _ = Incremental_backward_eta(q_eta, p_eta, obs, state, K, D, obs_mu=obs_mu, obs_tau=obs_tau)
#         eubos[2+2*m] = eubo_eta_forward - eubo_eta_backward
#         elbos[m+1] = elbo_eta_forward - elbo_eta_backward
#         esss[2+2*m] = (ess_eta_forward + ess_eta_backward) / 2
#     return eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_sis_init_eta(enc_eta, enc_z, obs, K, mcmc_size, device, RESAMPLE=True):
    """
    EUBO for amortized gibbs with backward transition,
    individually compute importance weights
    """
    sample_size, batch_size, N, D  = obs.shape
    eubos = torch.zeros(mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    ## sample from prior and finish one full update
    obs_mu, obs_tau  = enc_eta.sample_prior(sample_size, batch_size)
    q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
    eubo_state_forward, elbo_state_forward, ess_state_forward, weights_state = Incremental_forward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D)
    eubos[0] = eubo_state_forward
    elbos[0] = elbo_state_forward
    esss[0] = ess_state_forward
    for m in range(mcmc_size):
        if RESAMPLE:
            state = resample_state(q_z['zs'].value, weights_state, idw_flag=True)
        else:
            state = q_z['zs'].value
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        eubo_eta_forward, elbo_eta_forward, ess_eta_forward, weights_eta  = Incremental_forward_eta(q_eta, p_eta, obs, state, K, D)
        if RESAMPLE:
            obs_mu, obs_tau = resample_eta(q_eta['means'].value, q_eta['precisions'].value, weights_eta, idw_flag=True)
        else:
            obs_mu = q_eta['means'].value
            obs_tau = q_eta['precisions'].value
        q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
        eubo_state_forward, elbo_state_forward, ess_state_forward, weights_state = Incremental_forward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D)

        eubos[1+m] = (eubo_state_forward + eubo_eta_forward) / 2
        elbos[1+m] = (elbo_state_forward + elbo_eta_forward) / 2
        esss[1+m] = (ess_state_forward + ess_eta_forward) / 2
    return eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu
# 
# def Eubo_mcmc_init_eta(enc_eta, enc_z, obs, K, mcmc_size, device, RESAMPLE=False):
#     """
#     EUBO for amortized gibbs with backward transition,
#     individually compute importance weights
#     """
#     sample_size, batch_size, N, D  = obs.shape
#     eubos = torch.zeros(2*mcmc_size+1).cuda().to(device)
#     elbos = torch.zeros(2*mcmc_size+1).cuda().to(device)
#     esss = torch.zeros(2*mcmc_size+1).cuda().to(device)
#     ## sample from prior and finish one full update
#     obs_mu, obs_tau  = enc_eta.sample_prior(sample_size, batch_size)
#     q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
#     eubo_state_forward, elbo_state_forward, ess_state_forward, weights_state = Incremental_forward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D)
#     eubos[0] = eubo_state_forward
#     elbos[0] = elbo_state_forward
#     esss[0] = ess_state_forward
#     for m in range(mcmc_size):
#         state = q_z['zs'].value
#         q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
#         eubo_eta_backward, elbo_eta_backward, ess_eta_backward, _ = Incremental_backward_eta(q_eta, p_eta, obs, state, K, D, obs_mu=obs_mu, obs_tau=obs_tau)
#         state_r = resample_state(q_z['zs'].value, weights_state, idw_flag=True)
#         q_eta, p_eta, q_nu = enc_eta(obs, state_r, K, D)
#         eubo_eta_forward, elbo_eta_forward, ess_eta_forward, weights_eta  = Incremental_forward_eta(q_eta, p_eta, obs, state_r, K, D)
#
#         eubos[1+2*m] = eubo_eta_forward - eubo_eta_backward
#         elbos[1+2*m] = elbo_eta_forward - elbo_eta_backward
#         esss[1+2*m] = (ess_eta_forward + ess_eta_backward) / 2
#
#         obs_mu = q_eta['means'].value
#         obs_tau = q_eta['precisions'].value
#         q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
#         eubo_state_backward, elbo_state_backward, ess_state_backward, _ = Incremental_backward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D, state=state)
#
#         obs_mu_r, obs_tau_r = resample_eta(q_eta['means'].value, q_eta['precisions'].value, weights_eta, idw_flag=True)
#         q_z, p_z = enc_z(obs, obs_tau_r, obs_mu_r, N, sample_size, batch_size)
#         eubo_state_forward, elbo_state_forward, ess_state_forward, weights_state = Incremental_forward_state(q_z, p_z, obs, obs_mu_r, obs_tau_r, K, D)
#
#         eubos[2+2*m] = eubo_state_forward - eubo_state_backward
#         elbos[m+1] = elbo_state_forward - elbo_state_backward
#         esss[2+2*m] = (ess_state_forward + ess_state_backward) / 2
#     return eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu
