import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward import *
import probtorch
"""
IMPORTANT NOTICE : This script tries the up-to-now imporatance weights strategy without any resampling

EUBO : loss function is  E_{p(z'|x)}[KL(p(z | z', x) || q_\f(z | z', x))]
                        + E_{p(z|x)}[KL(p(z' | z, x) || q_\f(z' | z, x))]
init_eta : initialize eta as the first step

"""
def EUBO_init_eta(models, obs, SubTrain_Params):
    """
    NO Resampling
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, N, K, D, mcmc_size, prior_flag) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls = torch.zeros(mcmc_size+1).cuda().to(device)
    obs_tau, obs_mu, state, log_w_z_f = Init_step_eta(models, obs, N, K, D, sample_size, batch_size, prior_flag)
    w_z_f = F.softmax(log_w_z_f, 0).detach()
    if prior_flag:
        (enc_eta, enc_z) = models
        losss[0] = (w_z_f * log_w_z_f).sum(0).sum(-1).mean() ## weights S * B * N
        symkls[0] = (w_z_f * log_w_z_f).sum(0).sum(-1).mean() - log_w_z_f.sum(-1).mean()
    else:
        (oneshot_eta, enc_eta, enc_z) = models
        losss[0] = (w_z_f * log_w_z_f).sum(0).mean()  ## weights S * B
        symkls[0] = (w_z_f * log_w_z_f).sum(0).mean() - log_w_z_f.mean()
    esss[0] = (1. / (w_z_f**2).sum(0)).mean()
    for m in range(mcmc_size):
        state = resample_state(state, w_z_f, idw_flag=True) ## resample state
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        obs_tau, obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau, obs_mu)
        w_eta_f = F.softmax(log_w_eta_f, 0).detach()
        losss[m+1] += (w_eta_f * log_w_eta_f).sum(0).sum(-1).mean() + (w_eta_f * log_w_eta_b).sum(0).sum(-1).mean()
        obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, w_eta_f, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
        state, log_w_z_f, log_w_z_b = Incremental_z(q_z, p_z, obs, obs_tau, obs_mu, K, D, state)
        w_z_f = F.softmax(log_w_z_f, 0).detach()
        losss[m+1] += (w_z_f * log_w_z_f).sum(0).sum(-1).mean() + (w_z_f * log_w_z_b).sum(0).sum(-1).mean()
        ## compute the symmetric KL as a metric
        log_w_eta_sym = log_w_eta_f - log_w_eta_b
        w_eta_sym = F.softmax(log_w_eta_sym, 0)
        symkls[m+1] += (w_eta_sym * log_w_eta_sym).sum(0).sum(-1).mean() - log_w_eta_sym.sum(-1).mean()
        log_w_z_sym = log_w_z_f - log_w_z_b
        w_z_sym = F.softmax(log_w_z_sym, 0)
        symkls[m+1] += (w_z_sym * log_w_z_sym).sum(0).sum(-1).mean() - log_w_z_sym.sum(-1).mean()

        esss[m+1] = ((1. / (w_eta_f**2).sum(0)).mean() + (1. / (w_z_f**2).sum(0)).mean() ) / 2
    reused = (q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu)
    metric_step = {"symKL" : symkls, "loss" : losss,  "ess" : esss)}
    return losss.sum(), metric_step, reused

def EUBO_init_eta_cfz(models, obs, SubTrain_Params):
    """
    NO Resampling
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, N, K, D, mcmc_size, prior_flag) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls = torch.zeros(mcmc_size+1).cuda().to(device)
    if prior_flag:
        (enc_eta, enc_z) = models
        obs_mu, obs_tau = enc_eta.sample_prior(sample_size, batch_size)
        q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
        state = q_z['zs'].value ## S * B * N * K
        esss[0] = 1.0 ## assume ess is 1
    else:
        (oneshot_eta, enc_eta, enc_z) = models
        q_eta, p_eta, q_nu = oneshot_eta(obs, K, D)
        log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
        log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
        state = q_z['zs'].value ## S * B * N * K
        log_obs_n = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
        log_w_z_f = log_obs_n.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
        w_z_f = F.softmax(log_w_z_f, 0).detach()
        losss[0] = (w_z_f * log_w_z_f).sum(0).mean()  ## weights S * B
        symkls[0] = (w_z_f * log_w_z_f).sum(0).mean() - log_w_z_f.mean()
        esss[0] = (1. / (w_z_f**2).sum(0)).mean()
    for m in range(mcmc_size):
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        obs_tau, obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau, obs_mu)
        w_eta_f = F.softmax(log_w_eta_f, 0).detach()
        losss[m+1] += (w_eta_f * log_w_eta_f).sum(0).sum(-1).mean() + (w_eta_f * log_w_eta_b).sum(0).sum(-1).mean()
        obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, w_eta_f, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
        state = q_z['zs'].value
        ## compute the symmetric KL as a metric
        log_w_eta_sym = log_w_eta_f - log_w_eta_b
        w_eta_sym = F.softmax(log_w_eta_sym, 0)
        symkls[m+1] += (w_eta_sym * log_w_eta_sym).sum(0).sum(-1).mean() - log_w_eta_sym.sum(-1).mean()
        esss[m+1] = (1. / (w_eta_f**2).sum(0)).mean()
    reused = (q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu)
    metric_step = {"symKL" : symkls, "loss" : losss,  "ess" : esss)}
    return losss.sum(), metric_step, reused
