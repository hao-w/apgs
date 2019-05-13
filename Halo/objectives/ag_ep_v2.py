import torch
import torch.nn as nn
from utils import *
from forward_backward_v2 import *
import probtorch
"""
EUBO : loss function is  E_{p(z'|x)}[KL(p(z | z', x) || q_\f(z | z', x))]
                        + E_{p(z|x)}[KL(p(z' | z, x) || q_\f(z' | z, x))]
init_eta : initialize eta as the first step

"""
def EUBO_init_eta_v2(models, obs, SubTrain_Params, p_flag):
    """
    NO Resampling
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, obs_rad, noise_sigma, N, K, D, mcmc_size, only_forward) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_eta = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_z = torch.zeros(mcmc_size+1).cuda().to(device)
    gaps = torch.zeros(mcmc_size+1).cuda().to(device)

    obs_mu, state, log_w_f_z = Init_step_eta(models, obs, obs_rad, noise_sigma, N, K, D, sample_size, batch_size)
    w_f_z = F.softmax(log_w_f_z, 0).detach()

    (oneshot_eta, enc_eta, enc_z, dec_x) = models
    if p_flag == True:
        losss[0] = - log_w_f_z.mean()
    else:
        losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## EUBO for updating inference network
    gaps[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        else:
            state = resample_state(state, w_f_z, idw_flag=True)
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(dec_x, q_eta, p_eta, obs, state, obs_rad, noise_sigma, K, D, obs_mu)
        symkl_detailed_balance_eta, eubo_p_q_eta, elbo_p_q_eta, w_sym_eta, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b, only_forward=only_forward)
        obs_mu = resample_mu(obs_mu, w_f_eta) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state, log_w_z_f, log_w_z_b = Incremental_z(dec_x, q_z, p_z, obs, obs_mu, obs_rad, noise_sigma, K, D, state)
        symkl_detailed_balance_z, eubo_p_q_z, elbo_p_q_z, w_sym_z, w_f_z = detailed_balances(log_w_z_f, log_w_z_b, only_forward=only_forward)
        if p_flag:
            losss[m+1] = - elbo_p_q_z - elbo_p_q_eta
        else:
            losss[m+1] = eubo_p_q_eta + eubo_p_q_z
        gaps[m+1] = (eubo_p_q_eta - elbo_p_q_eta + eubo_p_q_z - elbo_p_q_z) / 2
        ## symmetric KLs as metrics
        symkls_DB_eta[m] = symkl_detailed_balance_eta
        symkls_DB_z[m] = symkl_detailed_balance_z
        esss[m+1] = ((1. / (w_sym_eta**2).sum(0)).mean() + (1. / (w_sym_z**2).sum(0)).mean() ) / 2
    reused = (q_eta, p_eta, q_z, p_z)
    metric_step = {"symKL_DB_eta" : symkls_DB_eta, "symKL_DB_z" : symkls_DB_z, "gap" : gaps, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused
