import torch
import torch.nn as nn
from utils import *
from forward_backward_rad import *
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
    (device, sample_size, batch_size, noise_sigma, N, K, D, mcmc_size) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_eta = torch.zeros(mcmc_size).cuda().to(device)
    symkls_DB_z = torch.zeros(mcmc_size).cuda().to(device)
    gaps = torch.zeros(mcmc_size+1).cuda().to(device)

    obs_mu, radi, state, log_w_f_z = Init_step_eta(models, obs, noise_sigma, N, K, D, sample_size, batch_size)
    w_f_z = F.softmax(log_w_f_z, 0).detach()

    (oneshot_eta, enc_eta, enc_z) = models
    losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## weights S * B
    gaps[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        else:
            state = resample_state(state, w_f_z, idw_flag=True)
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, radi, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, noise_sigma, K, D, obs_mu, radi)
        symkl_detailed_balance_eta, eubo_p_q_eta, gap_gibbs_q_eta, w_sym_eta, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b)
        obs_mu, radi = resample_eta(obs_mu, radi, w_f_eta, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, radi, noise_sigma, N, K, sample_size, batch_size)
        state, log_w_z_f, log_w_z_b = Incremental_z(q_z, p_z, obs, obs_mu, radi, noise_sigma, K, D, state)
        symkl_detailed_balance_z, eubo_p_q_z, gap_gibbs_q_z, w_sym_z, w_f_z = detailed_balances(log_w_z_f, log_w_z_b)
        losss[m+1] = eubo_p_q_eta + eubo_p_q_z
        gaps[m+1] = gap_gibbs_q_eta +  gap_gibbs_q_z
        ## symmetric KLs as metrics
        symkls_DB_eta[m] = symkl_detailed_balance_eta
        symkls_DB_z[m] = symkl_detailed_balance_z
        esss[m+1] = ((1. / (w_sym_eta**2).sum(0)).mean() + (1. / (w_sym_z**2).sum(0)).mean() ) / 2
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"symKL_DB_eta" : symkls_DB_eta, "symKL_DB_z" : symkls_DB_z, "gap" : gaps, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused

def EUBO_init_z_sanity(models, obs, SubTrain_Params):
    """
    NO Resampling
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, noise_sigma, N, K, D, mcmc_size) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_eta = torch.zeros(mcmc_size).cuda().to(device)
    gaps = torch.zeros(mcmc_size+1).cuda().to(device)

    obs_mu, radi, state, log_w_f_eta = Init_step_z(models, obs, noise_sigma, N, K, D, sample_size, batch_size)
    w_f_eta = F.softmax(log_w_f_eta, 0).detach()

    (enc_eta, enc_z) = models
    losss[0] = (w_f_eta * log_w_f_eta).sum(0).mean()  ## weights S * B
    gaps[0] = (w_f_eta * log_w_f_eta).sum(0).mean() - log_w_f_eta.mean()
    esss[0] = (1. / (w_f_eta**2).sum(0)).mean()
    for m in range(mcmc_size):
        obs_mu, radi = resample_eta(obs_mu, radi, w_f_eta, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, radi, noise_sigma, N, K, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value
        log_obs = True_Log_likelihood_rad(obs, state, obs_mu, radi, noise_sigma, K, D, cluster_flag=False)
        log_w_f_z = log_obs + log_p_z - log_q_z
        w_f_z = F.softmax(log_w_f_z, 0)
        state = resample_state(state, w_f_z, idw_flag=True)
        
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, radi, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, noise_sigma, K, D, obs_mu, radi)
        symkl_detailed_balance_eta, eubo_p_q_eta, gap_gibbs_q_eta, w_sym_eta, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b)        
        losss[m+1] = eubo_p_q_eta
        gaps[m+1] = gap_gibbs_q_eta
        ## symmetric KLs as metrics
        symkls_DB_eta[m] = symkl_detailed_balance_eta
        esss[m+1] = (1. / (w_sym_eta**2).sum(0)).mean()
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"symKL_DB_eta" : symkls_DB_eta, "gap" : gaps, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused
