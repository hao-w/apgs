import torch
import torch.nn as nn
from utils import *
from forward_backward_dec import *
import probtorch
"""
EUBO : loss function is  E_{p(z'|x)}[KL(p(z | z', x) || q_\f(z | z', x))]
                        + E_{p(z|x)}[KL(p(z' | z, x) || q_\f(z' | z, x))]
init_eta : initialize eta as the first step

"""
def EUBO_fixed_radi_enc(models, obs, SubTrain_Params):
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

    (oneshot_eta, enc_eta, enc_z, dec_x) = models

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
        obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(dec_x, q_eta, p_eta, obs, state, obs_rad, noise_sigma, K, D, obs_mu)
        db_eta, eubo_p_q_eta, elbo_p_q_eta, w_sym_eta = detailed_balances(log_w_eta_f, log_w_eta_b)
        obs_mu = resample_mu(obs_mu, w_sym_eta, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state, log_w_z_f, log_w_z_b = Incremental_z(dec_x, q_z, p_z, obs, obs_mu, obs_rad, noise_sigma, K, D, state)
        db_z, eubo_p_q_z, elbo_p_q_z, w_sym_z = detailed_balances(log_w_z_f, log_w_z_b)
        losss[m+1] = eubo_p_q_eta + eubo_p_q_z
        gaps_eta[m+1] = eubo_p_q_eta - elbo_p_q_eta
        gaps_z[m+1] = eubo_p_q_z - elbo_p_q_z
        ## symmetric KLs as metrics
#         symkls_DB_eta[m+1] = symkl_detailed_balance_eta
#         symkls_DB_z[m+1] = symkl_detailed_balance_z
        esss[m+1] = ((1. / (w_sym_eta**2).sum(0)).mean() + (1. / (w_sym_z**2).sum(0)).mean() ) / 2
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"gap_eta" : gaps_eta, "gap_z" : gaps_z, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused

def EUBO_fixed_radi_dec(models, obs, SubTrain_Params):
    """
    NO Resampling
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, obs_rad, noise_sigma, N, K, D, mcmc_size) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
#     symkls_DB_eta = torch.zeros(mcmc_size+1).cuda().to(device)
#     symkls_DB_z = torch.zeros(mcmc_size+1).cuda().to(device)


    obs_mu, state, log_w_f_z = Init_step_eta(models, obs, obs_rad, noise_sigma, N, K, D, sample_size, batch_size)
    w_f_z = F.softmax(log_w_f_z, 0).detach()
    (oneshot_eta, enc_eta, enc_z, dec_x) = models
    losss[0] = - log_w_f_z.mean()
#     symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
#     symkls_DB_z[0] = symkls_DB_eta[0] ##
    for m in range(mcmc_size):
        # if m == 0:
        #     state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        # else:
        #     state = resample_state(state, w_sym_z, idw_flag=True)
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        log_p_eta = p_eta['means'].log_prob.sum(-1)
        log_q_eta = q_eta['means'].log_prob.sum(-1)
        obs_mu = q_eta['means'].value

        labels = state.argmax(-1)
        p_x = dec_x(obs, state, obs_mu, obs_rad)
        log_obs_n = p_x['x_recon'].log_prob.sum(-1)
        log_obs_k = torch.cat([((labels==k).float() * log_obs_n).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
        log_w_f_eta = log_obs_k + log_p_eta - log_q_eta

        # obs_mu = resample_mu(obs_mu, w_sym_eta, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        log_p_z = p_z['zs'].log_prob
        log_q_z = q_z['zs'].log_prob
        state = q_z['zs'].value
        p_x = dec_x(obs, state, obs_mu, obs_rad)
        log_obs_n = p_x['x_recon'].log_prob.sum(-1)
        log_w_f_z = log_obs_n + log_p_z - log_q_z
        losss[m+1] = - log_w_f_eta.sum(-1).mean() - log_w_f_z.sum(-1).mean()
        ## symmetric KLs as metrics
#         symkls_DB_eta[m+1] = symkl_detailed_balance_eta
#         symkls_DB_z[m+1] = symkl_detailed_balance_z
    return losss.sum()
