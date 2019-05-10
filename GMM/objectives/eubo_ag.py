import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward import *
import probtorch

def Eubo_cfz_pr_z(enc_eta, gibbs_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, RESAMPLE=False, DETACH=True):
    """
    Use Gibbs update for z, in order to use reparameterized-style gradient estimation
    initialize z
    """
    symkls = torch.zeros(mcmc_size+1).cuda().to(device)
    eubos = torch.zeros(mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    ## initialize mu, tau from the prior
    state = gibbs_z.sample_prior(N, sample_size, batch_size)
    q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value

    log_obs_k = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=True)
    log_weights_eta = log_obs_k + log_p_eta - log_q_eta
    weights_eta = F.softmax(log_weights_eta, 0).detach()
    ## EUBO, ELBO, ESS
    eubos[0] = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
    elbos[0] = log_weights_eta.sum(-1).mean()
    esss[0] = (1. / (weights_eta**2).sum(0)).mean()
    for m in range(mcmc_size):
        if RESAMPLE:
            ## resample eta -- mu and tau
            obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, weights_eta, idw_flag=True)
        ## update z -- cluster assignments
        q_z, p_z = gibbs_z.forward(obs, obs_tau, obs_mu, N, K)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value ## S * B * N * K
        log_obs_n = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
        log_weights_state = log_obs_n + log_p_z - log_q_z
        weights_state = F.softmax(log_weights_state, 0)
#         if RESAMPLE:
#             state = resample_state(state, weights_state)
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_tau, obs_mu  = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau, obs_mu)
        ## EUBO, ELBO, ESS
        eubos[m+1] = eubo_eta
        elbos[m+1] = elbo_eta
        symkls[m+1] = symkl
        esss[m+1] = ess_eta
    symkls.sum().backward()
    optimizer.step()
    return symkls, eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu

def Eubo_cfz_pr_eta(enc_eta, gibbs_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device, RESAMPLE=False, DETACH=True):
    """
    Use Gibbs update for z, in order to use reparameterized-style gradient estimation
    initialize eta
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
    log_weights_state = log_obs_n + log_p_z - log_q_z
    weights_state = F.softmax(log_weights_state, 0).detach()
    ## EUBO, ELBO, ESS
    eubos[0] = (weights_state * log_weights_state).sum(0).sum(-1).mean()
    elbos[0] = log_weights_state.sum(-1).mean()
    esss[0] = (1. / (weights_state**2).sum(0)).mean()
    for m in range(mcmc_size):
#         if RESAMPLE:
#             state = resample_state(state, weights_state)
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        symkl, eubo_eta, elbo_eta, ess_eta, weights_eta, obs_tau, obs_mu  = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau, obs_mu, DETACH=DETACH)
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

def Eubo_cfz_os_eta(obs, Model_Params, device, models):
    """
    Use Gibbs update for z, in order to use reparameterized-style gradient estimation
    initialize eta
    """
    (oneshot_eta, enc_eta, enc_z) = models
    (N, K, D, mcmc_size, sample_size, batch_size) = Model_Params
    symkls = torch.zeros(mcmc_size+1).cuda().to(device)
    eubos = torch.zeros(mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    ## initialize mu, tau from the prior
    obs_tau, obs_mu, state, log_weights_state = Init_step_eta(obs, oneshot_eta, enc_z, N, K, D, sample_size, batch_size)
    weights_state = F.softmax(log_weights_state, 0)
    ## EUBO, ELBO, ESS
    eubos[0] = (weights_state * log_weights_state).sum(0).sum(-1).mean()
    elbos[0] = log_weights_state.sum(-1).mean()
    symkls[0] = (weights_state * log_weights_state).sum(0).sum(-1).mean() - log_weights_state.sum(-1).mean()
    esss[0] = (1. / (weights_state**2).sum(0)).mean()
    for m in range(mcmc_size):
        ## update tau and mu -- global variables
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        obs_tau, obs_mu, log_w_forward, log_w_backward  = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau, obs_mu)
        log_weights_eta = log_w_forward - log_w_backward
        weights_eta = F.softmax(log_weights_eta, 0).detach()
        eubos[m+1] = (weights_eta * log_weights_eta).sum(0).sum(-1).mean()
        elbos[m+1] = log_weights_eta.sum(-1).mean(0).mean()
        symkls[m+1] = eubos[m+1] - elbos[m+1]
        esss[m+1] = (1. / (weights_eta**2).sum(0)).mean()
        ## resample eta
        obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, weights_eta, idw_flag=True)
        ## update z using Gibbs sampler -- cluster assignments
        q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
        state = q_z['zs'].value ## S * B * N * K
    reused = (q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu)
    loss = symkls.sum()
    metric_step = {"symKL" : symkls.sum().item(), "eubo" : eubos.sum().item(), "elbo" : elbos.sum().item(), "ess" : esss.mean().item()}
    return loss, metric_step, reused
