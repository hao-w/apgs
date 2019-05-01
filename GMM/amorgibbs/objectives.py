%run ../../path_import.py
import torch
import torch.nn as nn
from utils import *
from normal_gamma_kls import *
from normal_gamma_conjugacy import *
import probtorch

def Eubo_init_z(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device):
    """
    initialize z
    adaptive resampling after each step
    jointly compute importance weights
    """
    eubos = torch.zeros(mcmc_size).cuda().to(device)
    elbos = torch.zeros(mcmc_size).cuda().to(device)
    esss = torch.zeros(mcmc_size).cuda().to(device)

    for m in range(mcmc_size):
        if m == 0:
            p_init_z = cat(enc_z.prior_pi)
            states = p_init_z.sample((sample_size, batch_size, N,))
            log_p_z = p_init_z.log_prob(states).sum(-1)## S * B * N
            log_q_z = p_init_z.log_prob(states).sum(-1)
        else:
            ## adaptive resampling
            obs_mu, obs_sigma = resample_eta(obs_mu, obs_sigma, weights)
            ## update z -- cluster assignments
            q_z, p_z = enc_z(obs, obs_sigma, obs_mu, N, sample_size, batch_size)
            log_p_z = p_z.log_joint(sample_dims=0, batch_dim=1)
            log_q_z = q_z.log_joint(sample_dims=0, batch_dim=1)
            states = q_z['zs'].value ## S * B * N * K
        ## update tau and mu -- global variables
        local_vars = torch.cat((obs, states), -1)
        q_eta, p_eta, q_nu = enc_eta(local_vars, K, D)
        log_p_eta = p_eta.log_joint(sample_dims=0, batch_dim=1)
        log_q_eta = q_eta.log_joint(sample_dims=0, batch_dim=1)

        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        obs_sigma = 1. / obs_tau.sqrt()
        log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights = log_obs.sum(-1) + log_p_eta + log_p_z - log_q_eta - log_q_z
        weights = F.softmax(log_weights, 0).detach()
        eubos[m] = (weights * log_weights).sum(0).mean()
        elbos[m] = log_weights.mean()
        esss[m] = (1. / (weights**2).sum(0)).mean()
    return eubos.mean(), elbos.mean(), esss.mean(), q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_init_eta(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device):
    """
    initialize eta
    adaptive resampling after each step
    jointly compute importance weights
    """
    eubos = torch.zeros(mcmc_size).cuda().to(device)
    elbos = torch.zeros(mcmc_size).cuda().to(device)
    esss = torch.zeros(mcmc_size).cuda().to(device)

    for m in range(mcmc_size):
        if m == 0:
            prior_mu_expand = enc_eta.prior_mu.unsqueeze(0).unsqueeze(0).repeat(sample_size, batch_size, 1, 1)
            p_init_tau = Gamma(enc_eta.prior_alpha, enc_eta.prior_alpha)
            obs_tau = p_init_tau.sample((sample_size, batch_size,))
            p_init_mu = Normal(prior_mu_expand, 1. / (enc_eta.prior_nu * obs_tau).sqrt())
            obs_mu = p_init_mu.sample()
            log_p_eta = p_init_tau.log_prob(obs_tau).sum(-1).sum(-1) + p_init_mu.log_prob(obs_mu).sum(-1).sum(-1)## S * B
            log_q_eta = log_p_eta
            obs_sigma = 1. / obs_tau.sqrt()
        else:
            states = resample_states(states, weights)
            ## update tau and mu -- global variables
            local_vars = torch.cat((obs, states), -1)
            q_eta, p_eta, q_nu = enc_eta(local_vars, K, D)
            log_p_eta = p_eta.log_joint(sample_dims=0, batch_dim=1)
            log_q_eta = q_eta.log_joint(sample_dims=0, batch_dim=1)
            ## for individual importance weight, S * B * K
            obs_mu = q_eta['means'].value
            obs_tau = q_eta['precisions'].value
            obs_sigma = 1. / obs_tau.sqrt()
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_sigma, obs_mu, N, sample_size, batch_size)
        log_p_z = p_z.log_joint(sample_dims=0, batch_dim=1)
        log_q_z = q_z.log_joint(sample_dims=0, batch_dim=1)
        states = q_z['zs'].value ## S * B * N * K
        log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights = log_obs.sum(-1) + log_p_eta + log_p_z - log_q_eta - log_q_z
        weights = F.softmax(log_weights, 0).detach()
        eubos[m] = (weights * log_weights).sum(0).mean()
        elbos[m] = log_weights.mean()
        esss[m] = (1. / (weights**2).sum(0)).mean()
    return eubos.mean(), elbos.mean(), esss.mean(), q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu
