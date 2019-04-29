import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch

def Eubo_gibbs_z(enc_z, obs, obs_mu, obs_rad, N, K, D, sample_size, batch_size, gpu, decay_factor, noise_sigma):
    q_z, p_z = enc_z(obs, obs_mu, obs_rad, sample_size, batch_size, decay_factor)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob ## S * B * N
    states = q_z['zs'].value
    log_obs_n = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma, gpu=gpu, cluster_flag=False)
    log_weights_local = log_obs_n + log_p_z - log_q_z
    weights_local = F.softmax(log_weights_local, 0).detach()
    eubo =(weights_local * log_weights_local).sum(0).sum(-1).mean()
    elbo = log_weights_local.sum(-1).mean()
    ess = (1. / (weights_local**2).sum(0)).mean()
    return eubo, elbo, ess

def Eubo_ag(enc_mu, enc_z, obs, obs_rad, N, K, D, mcmc_size, sample_size, batch_size, gpu, decay_factor, noise_sigma):
    """
    initialize z

    """
    loss = 0.0
    eubos = torch.zeros(mcmc_size).cuda()
    elbos = torch.zeros(mcmc_size).cuda()
    esss = torch.zeros(mcmc_size).cuda()

    p_init_z = cat(enc_z.prior_pi)
    states = p_init_z.sample((sample_size, batch_size, N,))
    log_p_z = p_init_z.log_prob(states)## S * B * N
    log_q_z = p_init_z.log_prob(states)

    for m in range(mcmc_size):
        q_mu, p_mu = enc_mu(obs, states, sample_size, batch_size)
        log_q_mu = q_mu['means'].log_prob.sum(-1)
        log_p_mu = p_mu['means'].log_prob.sum(-1) # S * B * K
        obs_mu = q_mu['means'].value
        log_obs_k = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma=noise_sigma, gpu=gpu, cluster_flag=True)
        log_weights_global = log_obs_k + log_p_mu - log_q_mu
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## resample mu
        obs_mu = resample_mu(obs_mu, weights_global)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_mu, obs_rad, sample_size, batch_size, decay_factor=decay_factor)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob ## S * B * N
        states = q_z['zs'].value
        log_obs_n = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma=noise_sigma, gpu=gpu, cluster_flag=False)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()

        eubos[m] =((weights_global * log_weights_global).sum(0).sum(-1).mean() + (weights_local * log_weights_local).sum(0).sum(-1).mean()) / 2
        elbos[m] = (log_obs_n.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1) + log_p_mu.sum(-1) - log_q_mu.sum(-1)).mean()
        esss[m] = ((1. / (weights_local**2).sum(0)).mean() + (1. / (weights_global**2).sum(0)).mean() ) / 2
    return eubos, elbos, esss
