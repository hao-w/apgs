import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import probtorch
from forward_backward import *

def Eubo_cfz_init_z(enc_mu, gibbs_z, obs, obs_rad, noise_sigma, N, K, D, mcmc_size, sample_size, batch_size, device, RESAMPLE, DETACH):
    """
    initialize z

    """
    symkls = torch.zeros(mcmc_size+1).cuda().to(device)
    eubos = torch.zeros(mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)

    state = gibbs_z.sample_prior(N, sample_size, batch_size)
    q_mu, p_mu = enc_mu(obs, state, K, sample_size, batch_size)
    log_q_mu = q_mu['means'].log_prob.sum(-1)
    log_p_mu = p_mu['means'].log_prob.sum(-1) # S * B * K
    obs_mu = q_mu['means'].value
    log_obs_k = True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=True, fixed_radius=True)
    log_weights_eta = log_obs_k + log_p_mu - log_q_mu
    weights_eta = F.softmax(log_weights_eta, 0).detach()
    ## EUBO, ELBO, ESS
    eubos[0] = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
    elbos[0] = log_weights_eta.sum(-1).mean()
    esss[0] = (1. / (weights_eta**2).sum(0)).mean()
    
    for m in range(mcmc_size):
        if RESAMPLE:
            obs_mu = resample_mu(obs_mu, weights_eta)
        q_z, p_z = gibbs_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value ## S * B * N * K
        log_obs_n = True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=False, fixed_radius=True)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()
#         if RESAMPLE:
#             state = resample_state(state, weights_local)
        ## update mu
        q_mu, p_mu = enc_mu(obs, state, K, sample_size, batch_size)
        symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_mu = Incremental_eta(q_mu, p_mu, obs, state, K, D, obs_mu, obs_rad, noise_sigma, DETACH=DETACH)
        eubos[m+1] = eubo_eta
        elbos[m+1] = elbo_eta
        symkls[m+1] = symkl
        esss[m+1] = ess_eta
    return symkls, eubos, elbos, esss, q_mu, p_mu, q_z, p_z


def Eubo_cfz_init_eta(enc_mu, gibbs_z, obs, obs_rad, noise_sigma, N, K, D, mcmc_size, sample_size, batch_size, device, RESAMPLE, DETACH):
    """
    initialize eta

    """
    symkls = torch.zeros(mcmc_size+1).cuda().to(device)
    eubos = torch.zeros(mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)

    obs_mu = enc_mu.sample_prior(sample_size, batch_size)
    q_z, p_z = gibbs_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=False, fixed_radius=True)
    log_weights = log_obs_n + log_p_z - log_q_z
    weights = F.softmax(log_weights, 0).detach()
    ## EUBO, ELBO, ESS
    eubos[0] = (weights * log_weights).sum(0).sum(-1).mean()
    elbos[0] = log_weights.sum(-1).mean()
    esss[0] = (1. / (weights**2).sum(0)).mean()
    
    for m in range(mcmc_size):
#         if RESAMPLE:
#             state = resample_state(state, weights)
        ## update mu
        q_eta, p_eta = enc_mu(obs, state, K, sample_size, batch_size)
        symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_mu = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_mu, obs_rad, noise_sigma, DETACH=DETACH)
        if RESAMPLE:
            obs_mu = resample_mu(obs_mu, weights_eta)
        ## update z -- cluster assignments
        q_z, p_z = gibbs_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value ## S * B * N * K
        ## EUBO, ELBO, ESS
        eubos[m+1] = eubo_eta
        elbos[m+1] = elbo_eta
        symkls[m+1] = symkl
        esss[m+1] = ess_eta
    return symkls, eubos, elbos, esss, q_eta, p_eta, q_z, p_z


def Eubo_mu_init_z(enc_mu, enc_z, obs, obs_rad, N, K, D, mcmc_size, sample_size, batch_size, device, noise_sigma):
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

def Eubo_rad(enc_mu_rad, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, noise_sigma):
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
        log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['rads'].log_prob.sum(-1)
        log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['rads'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        obs_rad = q_eta['rads'].value
        log_obs_k = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma, device, cluster_flag=True, fixed_radius=False)
        log_weights_global = log_obs_k + log_p_eta - log_q_eta
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## resample mu
        obs_mu, obs_rad = resample_eta(obs_mu, obs_rad, weights_global)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_mu, obs_rad, N, sample_size, batch_size, device)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob ## S * B * N
        states = q_z['zs'].value
        log_obs_n = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma, device, cluster_flag=False, fixed_radius=False)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()

        eubos[m] =((weights_global * log_weights_global).sum(0).sum(-1).mean() + (weights_local * log_weights_local).sum(0).sum(-1).mean()) / 2
        elbos[m] = (log_weights_global.sum(-1).mean() + log_weights_local.sum(-1).mean()) / 2
        esss[m] = ((1. / (weights_local**2).sum(0)).mean() + (1. / (weights_global**2).sum(0)).mean() ) / 2
    return eubos, elbos, esss
