import torch
import torch.nn as nn
from utils import *
from normal_gamma_kls import *
from normal_gamma_conjugacy import *
from forward_backward import *
import probtorch

def Eubo_cfz(enc_eta, gibbs_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, RESAMPLE=False):
    """
    Use Gibbs update for z, in order to use reparameterized-style gradient estimation
    """
    symkls = torch.zeros(mcmc_size+1).cuda().to(device)
    eubos = torch.zeros(mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    ## initialize mu, tau from the prior
    obs_mu, obs_tau = enc_eta.sample_prior(sample_size, batch_size)
    q_z, p_z = gibbs_z.forward(obs, obs_tau, obs_mu, N, K)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
    log_weights = log_obs_n + log_p_z - log_q_z
    weights = F.softmax(log_weights, 0).detach()
    ## EUBO, ELBO, ESS
    eubos[0] = (weights * log_weights).sum(0).sum(-1).mean()
    elbos[0] = log_weights.sum(-1).mean()
    esss[0] = (1. / (weights**2).sum(0)).mean()
    for m in range(mcmc_size):
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_mu, obs_tau  = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau, obs_mu)
        if RESAMPLE:
            ## resample eta -- mu and tau
            obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, weights_eta, idw_flag=True)
        ## update z -- cluster assignments
        q_z, p_z = gibbs_z.forward(obs, obs_tau, obs_mu, N, K)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value ## S * B * N * K
        ## EUBO, ELBO, ESS
        eubos[m+1] = eubo_eta
        elbos[m+1] = elbo_eta
        symkls[m+1] = symkl
        esss[m+1] = ess_eta
    return symkls, eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_oneshot_eta(oneshot_eta, enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, idw_flag=True):
    """
    initialize eta
    adaptive resampling after each step
    individually compute importance weights
    """
    eubos = torch.zeros(mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    ## initialize mu, tau from the prior
    q_eta, p_eta, q_nu = oneshot_eta(obs, K, D)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(obs, state, obs_mu, 1. / obs_tau.sqrt(), K, D, cluster_flag=False)
    log_weights = log_obs_n.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1)
    weights = F.softmax(log_weights, 0).detach()
    ## EUBO, ELBO, ESS
    eubos[0] = (weights * log_weights).sum(0).mean()
    elbos[0] = log_weights.mean()
    esss[0] = (1. / (weights**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            ## resample z -- cluster assignments
            state = resample_state(state, weights, idw_flag=False)
        else:
            state = resample_state(state, weights_local, idw_flag=idw_flag)
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
        log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        log_obs_k = Log_likelihood(obs, state, obs_mu, 1. / obs_tau.sqrt(), K, D, cluster_flag=True)
        log_weights_global = log_obs_k + log_p_eta - log_q_eta
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## resample eta -- mu and tau
        obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, weights_global, idw_flag=idw_flag)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value ## S * B * N * K
        log_obs_n = Log_likelihood(obs, state, obs_mu, 1. / obs_tau.sqrt(), K, D, cluster_flag=False)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()
        ## EUBO, ELBO, ESS
        eubos[m+1] =((weights_global * log_weights_global).sum(0).sum(-1).mean() + (weights_local * log_weights_local).sum(0).sum(-1).mean()) / 2
        elbos[m+1] = (log_weights_global.sum(-1).mean() + log_weights_local.sum(-1).mean()) / 2
        esss[m+1] = ((1. / (weights_local**2).sum(0)).mean() + (1. / (weights_global**2).sum(0)).mean() ) / 2
    return eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_idw_init_z(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, idw_flag=True):
    """
    initialize z
    adaptive resampling after each step
    individually compute importance weights
    """
    eubos = torch.zeros(mcmc_size).cuda().to(device)
    elbos = torch.zeros(mcmc_size).cuda().to(device)
    esss = torch.zeros(mcmc_size).cuda().to(device)
    ## initialize z from the prior
    state = enc_z.sample_prior(N, sample_size, batch_size)
    for m in range(mcmc_size):
        if m != 0:
            state = resample_state(state, weights_local, idw_flag=idw_flag)
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
        log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        obs_sigma = 1. / obs_tau.sqrt()
        log_obs_k = Log_likelihood(obs, state, obs_mu, obs_sigma, K, D, cluster_flag=True)
        log_weights_global = log_obs_k + log_p_eta - log_q_eta
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## resample global
        obs_mu, obs_sigma = resample_eta(obs_mu, obs_sigma, weights_global, idw_flag=idw_flag)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_sigma, obs_mu, N, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value ## S * B * N * K
        log_obs_n = Log_likelihood(obs, state, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()
        ## EUBO, ELBO, ESS
        eubos[m] =((weights_global * log_weights_global).sum(0).sum(-1).mean() + (weights_local * log_weights_local).sum(0).sum(-1).mean()) / 2
        elbos[m] = (log_weights_global.sum(-1).mean() + log_weights_local.sum(-1).mean()) / 2
        esss[m] = ((1. / (weights_local**2).sum(0)).mean() + (1. / (weights_global**2).sum(0)).mean() ) / 2
    return eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_idw_init_eta(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, idw_flag=True):
    """
    initialize eta
    adaptive resampling after each step
    individually compute importance weights
    """
    eubos = torch.zeros(mcmc_size).cuda().to(device)
    elbos = torch.zeros(mcmc_size).cuda().to(device)
    esss = torch.zeros(mcmc_size).cuda().to(device)
    ## initialize mu, tau from the prior
    obs_mu, obs_sigma = enc_eta.sample_prior(sample_size, batch_size)
    for m in range(mcmc_size):
        if m != 0:
            obs_mu, obs_sigma = resample_eta(obs_mu, obs_sigma, weights_global, idw_flag=idw_flag)
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_sigma, obs_mu, N, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value ## S * B * N * K
        log_obs_n = Log_likelihood(obs, state, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights_local = log_obs_n + log_p_z - log_q_z
        weights_local = F.softmax(log_weights_local, 0).detach()
        ## resample z -- cluster assignments
        state = resample_state(state, weights_local, idw_flag=idw_flag)
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
        log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        obs_sigma = 1. / obs_tau.sqrt()
        log_obs_k = Log_likelihood(obs, state, obs_mu, obs_sigma, K, D, cluster_flag=True)
        log_weights_global = log_obs_k + log_p_eta - log_q_eta
        weights_global = F.softmax(log_weights_global, 0).detach()
        ## EUBO, ELBO, ESS
        eubos[m] =((weights_global * log_weights_global).sum(0).sum(-1).mean() + (weights_local * log_weights_local).sum(0).sum(-1).mean()) / 2
        elbos[m] = (log_weights_global.sum(-1).mean() + log_weights_local.sum(-1).mean()) / 2
        esss[m] = ((1. / (weights_local**2).sum(0)).mean() + (1. / (weights_global**2).sum(0)).mean() ) / 2
    return eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_init_z(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, idw_flag=False):
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
            states = enc_z.sample_prior(N, sample_size, batch_size)
        else:
            ## adaptive resampling
            obs_mu, obs_sigma = resample_eta(obs_mu, obs_sigma, weights, idw_flag=idw_flag)
            ## update z -- cluster assignments
            q_z, p_z = enc_z(obs, obs_sigma, obs_mu, N, sample_size, batch_size)
            log_p_z = p_z.log_joint(sample_dims=0, batch_dim=1)
            log_q_z = q_z.log_joint(sample_dims=0, batch_dim=1)
            states = q_z['zs'].value ## S * B * N * K
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, states, K, D)
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

def Eubo_init_eta(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, idw_flag=False):
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
            obs_mu, obs_sigma = enc_eta.sample_prior(sample_size, batch_size)
        else:
            # states = resample_states(states, weights, idw_flag=idw_flag)
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
