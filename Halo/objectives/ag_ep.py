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
def EUBO_fixed_radi(models, obs, SubTrain_Params):
    """
    NO Resampling
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, obs_rad, noise_sigma, N, K, D, mcmc_size) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
#     symkls_DB_eta = torch.zeros(mcmc_size+1).cuda().to(device)
#     symkls_DB_z = torch.zeros(mcmc_size+1).cuda().to(device)
    gaps_eta = torch.zeros(mcmc_size+1).cuda().to(device)
    gaps_z = torch.zeros(mcmc_size+1).cuda().to(device)


    obs_mu, state, log_w_f_z = Init_step_eta(models, obs, obs_rad, noise_sigma, N, K, D, sample_size, batch_size)
    w_f_z = F.softmax(log_w_f_z, 0).detach()

    (oneshot_eta, enc_eta, enc_z) = models
    losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## weights S * B
#     symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
#     symkls_DB_z[0] = symkls_DB_eta[0] ##
    gaps_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
    gaps_z[0] =  gaps_eta[0]
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        else:
            state = resample_state(state, w_sym_z, idw_flag=True)
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, obs_rad, noise_sigma, K, D, obs_mu)
        db_eta, eubo_p_q_eta, gap_gibbs_q_eta, w_sym_eta = detailed_balances(log_w_eta_f, log_w_eta_b)
        obs_mu = resample_mu(obs_mu, w_sym_eta, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state, log_w_z_f, log_w_z_b = Incremental_z(q_z, p_z, obs, obs_mu, obs_rad, noise_sigma, K, D, state)
        db_z, eubo_p_q_z, gap_gibbs_q_z, w_sym_z = detailed_balances(log_w_z_f, log_w_z_b)
        losss[m+1] = eubo_p_q_eta + eubo_p_q_z
        gaps_eta[m+1] = gap_gibbs_q_eta
        gaps_z[m+1] = gap_gibbs_q_z
        ## symmetric KLs as metrics
#         symkls_DB_eta[m+1] = symkl_detailed_balance_eta
#         symkls_DB_z[m+1] = symkl_detailed_balance_z
        esss[m+1] = ((1. / (w_sym_eta**2).sum(0)).mean() + (1. / (w_sym_z**2).sum(0)).mean() ) / 2
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"gap_eta" : gaps_eta, "gap_z" : gaps_z, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused

def EUBO_fixed_radi_joint_both(models, obs, SubTrain_Params):
    """
    joint sampling for both mu and local variables
    """
    (device, sample_size, batch_size, obs_rad, noise_sigma, N, K, D, mcmc_size) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    gaps = torch.zeros(mcmc_size+1).cuda().to(device)
    
    obs_mu, state, log_w_f_z = Init_step_eta(models, obs, obs_rad, noise_sigma, N, K, D, sample_size, batch_size)
    w_f_z = F.softmax(log_w_f_z, 0).detach()

    (oneshot_eta, enc_eta, enc_z) = models
    losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## weights S * B
#     symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
#     symkls_DB_z[0] = symkls_DB_eta[0] ##
    gaps[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        else:
            state = resample_state(state, w_sym, idw_flag=False)
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu_prev = obs_mu
        obs_mu = q_eta['means'].value        
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state_prev = state
        state = q_z['zs'].value
        log_p = p_eta['means'].log_prob.sum(-1).sum(-1) + p_z['zs'].log_prob.sum(-1)
        log_q = q_eta['means'].log_prob.sum(-1).sum(-1) + q_z['zs'].log_prob.sum(-1)
        log_obs = True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=False)
        log_w_f = log_obs.sum(-1) + log_p - log_q
        ## backward
        log_p_prev = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1).sum(-1) + cat(probs=p_z['zs'].dist.probs).log_prob(state_prev).sum(-1)
        log_q_prev = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1).sum(-1) + cat(probs=q_z['zs'].dist.probs).log_prob(state_prev).sum(-1)
        log_obs_prev = True_Log_likelihood(obs, state_prev, obs_mu_prev, obs_rad, noise_sigma, K, D, cluster_flag=False)
        log_w_b = log_obs_prev.sum(-1) + log_p_prev - log_q_prev
        db, eubo_p_q, gap_gibbs_q, w_sym = detailed_balances(log_w_f, log_w_b)
        losss[m+1] = eubo_p_q
        gaps[m+1] = gap_gibbs_q
        ## symmetric KLs as metrics
#         symkls_DB_eta[m+1] = symkl_detailed_balance_eta
#         symkls_DB_z[m+1] = symkl_detailed_balance_z
        esss[m+1] = (1. / (w_sym**2).sum(0)).mean()
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"gap" : gaps, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused