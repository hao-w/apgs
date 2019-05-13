import torch
import torch.nn as nn
from utils import *
from forward_backward import *
import probtorch
"""
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
    (device, sample_size, batch_size, obs_rad, noise_sigma, N, K, D, mcmc_size, prior_flag, only_forward) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_eta = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_z = torch.zeros(mcmc_size+1).cuda().to(device)

    obs_mu, state, log_w_f_z = Init_step_eta(models, obs, obs_rad, noise_sigma, N, K, D, sample_size, batch_size, prior_flag)
    w_f_z = F.softmax(log_w_f_z, 0).detach()
    if prior_flag:
        (enc_eta, enc_z) = models
        losss[0] = (w_f_z * log_w_f_z).sum(0).sum(-1).mean() ## weights S * B * N
        symkls_DB_z[0] = (w_f_z * log_w_f_z).sum(0).sum(-1).mean() - log_w_f_z.sum(-1).mean()
    else:
        (oneshot_eta, enc_eta, enc_z) = models
        losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## weights S * B
        symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
        symkls_DB_z[0] = symkls_DB_eta[0] ##
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        else:
            state = resample_state(state, w_f_z, idw_flag=True)
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, obs_rad, noise_sigma, K, D, obs_mu)
        symkl_detailed_balance_eta, eubo_p_q_eta, w_sym_eta, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b, only_forward=only_forward)
        obs_mu = resample_mu(obs_mu, w_f_eta) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state, log_w_z_f, log_w_z_b = Incremental_z(q_z, p_z, obs, obs_mu, obs_rad, noise_sigma, K, D, state)
        symkl_detailed_balance_z, eubo_p_q_z, w_sym_z, w_f_z = detailed_balances(log_w_z_f, log_w_z_b, only_forward=only_forward)
        losss[m+1] = eubo_p_q_eta + eubo_p_q_z
        ## symmetric KLs as metrics
        symkls_DB_eta[m+1] = symkl_detailed_balance_eta
        symkls_DB_z[m+1] = symkl_detailed_balance_z
        esss[m+1] = ((1. / (w_sym_eta**2).sum(0)).mean() + (1. / (w_sym_z**2).sum(0)).mean() ) / 2
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"symKL_DB_eta" : symkls_DB_eta, "symKL_DB_z" : symkls_DB_z, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused

def EUBO_init_eta_cfz(models, obs, SubTrain_Params):
    """
    NO Resampling
    Learn neural gibbs samplers for eta, closed-form gibbs sampler for z
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, N, K, D, mcmc_size, prior_flag) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_eta = torch.zeros(mcmc_size+1).cuda().to(device)

    if prior_flag:
        (enc_eta, enc_z) = models
        obs_mu = enc_eta.sample_prior(sample_size, batch_size)
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state = q_z['zs'].value ## S * B * N * K
        esss[0] = 1.0 ## assume ess is 1
    else:
        (oneshot_eta, enc_eta, enc_z) = models
        q_eta, p_eta, q_nu = oneshot_eta(obs, K, D)
        log_p_eta = p_eta['means'].log_prob.sum(-1)
        log_q_eta = q_eta['means'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state = q_z['zs'].value ## S * B * N * K
        log_obs_n = Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=False, fixed_radius=True)
        log_w_f_z = log_obs_n.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
        w_f_z = F.softmax(log_w_f_z, 0).detach()
        losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## weights S * B
        symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
        symkls_DB_z[0] = symkls_DB_eta[0]
        esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, obs_rad, noise_sigma, K, D, obs_mu)
        symkl_detailed_balance_eta, eubo_p_q_eta, w_sym_eta, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b)
        obs_mu = resample_mu(obs_mu, w_f_eta) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state = q_z['zs'].value
        losss[m+1] = eubo_p_q_eta
        ##  symmetric KL as a metric
        symkls_DB_eta[m+1] = symkl_detailed_balance_eta
        esss[m+1] = (1. / (w_sym_eta**2).sum(0)).mean()
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"symKL_DB_eta" : symkls_DB_eta, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused
