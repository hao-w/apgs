import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch

def Eubo_mu(enc_mu, enc_z, obs, obs_rad, N, K, D, mcmc_size, sample_size, batch_size, device, noise_sigma):
    """
    initialize z

    """
    eubos = torch.zeros(mcmc_size).cuda()
    elbos = torch.zeros(mcmc_size).cuda()
    esss = torch.zeros(mcmc_size).cuda()

    p_init_z = cat(enc_z.prior_pi)
    states = p_init_z.sample((sample_size, batch_size, N,))
    log_p_z = p_init_z.log_prob(states)## S * B * N
    log_q_z = p_init_z.log_prob(states)

    for m in range(mcmc_size):
        if m != 0:
            states = resample_states(states, weights_local)
        q_mu, p_mu = enc_mu(obs, states, K, sample_size, batch_size)
        log_q_mu = q_mu['means'].log_prob.sum(-1)
        log_p_mu = p_mu['means'].log_prob.sum(-1) # S * B * K
        obs_mu = q_mu['means'].value
        log_obs_k = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma=noise_sigma, device=device, cluster_flag=True, fixed_radius=True)
        log_weights_global = log_obs_k + log_p_mu - log_q_mu
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## resample mu
        obs_mu = resample_mu(obs_mu, weights_global)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_mu, obs_rad, N, sample_size, batch_size, noise_sigma, device)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob ## S * B * N
        states = q_z['zs'].value
        log_obs_n = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma=noise_sigma, device=device, cluster_flag=False, fixed_radius=True)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()

        eubos[m] =((weights_global * log_weights_global).sum(0).sum(-1).mean() + (weights_local * log_weights_local).sum(0).sum(-1).mean()) / 2
        elbos[m] = (log_weights_global.sum(-1).mean() + log_weights_local.sum(-1).mean()) / 2
        esss[m] = ((1. / (weights_local**2).sum(0)).mean() + (1. / (weights_global**2).sum(0)).mean() ) / 2
    return eubos, elbos, esss, q_mu, q_z

def Eubo_ag_rad(enc_mu_rad, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, noise_sigma, device):
    """
    initialize z
    learn both mean and radius

    """
    eubos = torch.zeros(mcmc_size).cuda().to(device)
    elbos = torch.zeros(mcmc_size).cuda().to(device)
    esss = torch.zeros(mcmc_size).cuda().to(device)

    p_init_z = cat(enc_z.prior_pi)
    states = p_init_z.sample((sample_size, batch_size, N,))
    log_p_z = p_init_z.log_prob(states)## S * B * N
    log_q_z = p_init_z.log_prob(states)

    for m in range(mcmc_size):
        if m != 0:
            states = resample_states(states, weights_local)
        q_eta, p_eta = enc_mu_rad(obs, states, K, sample_size, batch_size)
        log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['radius'].log_prob.sum(-1)
        log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['radius'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        obs_rad = q_eta['radius'].value
        log_obs_k = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma, device, cluster_flag=True)
        log_weights_global = log_obs_k + log_p_eta - log_q_eta
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## resample mu
        obs_mu, obs_rad = resample_eta(obs_mu, obs_rad, weights_global)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_mu, obs_rad, N, sample_size, batch_size, noise_sigma, device)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob ## S * B * N
        states = q_z['zs'].value
        log_obs_n = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma, device, cluster_flag=False)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()

        eubos[m] =((weights_global * log_weights_global).sum(0).sum(-1).mean() + (weights_local * log_weights_local).sum(0).sum(-1).mean()) / 2
        elbos[m] = (log_weights_global.sum(-1).mean() + log_weights_local.sum(-1).mean()) / 2
        esss[m] = ((1. / (weights_local**2).sum(0)).mean() + (1. / (weights_global**2).sum(0)).mean() ) / 2
    return eubos, elbos, esss
