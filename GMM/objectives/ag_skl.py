import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward import *
import probtorch
"""
SKL : loss function is the symmetric KL divergence that JW derived
init_z : initialize z as the first step
init_eta : initialize eta as the first step
"""
def SKL_init_eta(models, obs, SubTrain_Params):
    """
    Use Gibbs update for z, in order to use reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, N, K, D, mcmc_size, prior_flag) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    symkls_DB_eta = torch.zeros(mcmc_size+1).cuda().to(device)
    ## initialize mu, tau from the prior
    obs_tau, obs_mu, state, log_w_f_z = Init_step_eta(models, obs, N, K, D, sample_size, batch_size, prior_flag)
    w_f_z = F.softmax(log_w_f_z, 0).detach()
    if prior_flag:
        (enc_eta, enc_z) = models
        ## weights S * B * N
        losss[0] = (w_f_z * log_w_f_z).sum(0).sum(-1).mean() ## weights S * B * N
        symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).sum(-1).mean() - log_w_f_z.sum(-1).mean()
    else:
        (oneshot_eta, enc_eta, enc_z) = models
        losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## weights S * B
        symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        q_eta, p_eta = enc_eta(obs, state, K, D)
        obs_tau, obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, K, D, obs_tau, obs_mu)
        symkl_detailed_balance_eta, eubo_p_q_eta, w_sym_eta, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b)
        obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, w_f_eta, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
        state = q_z['zs'].value
        losss[m+1] = symkl_detailed_balance_eta
        ## symmetric KLs as metrics
        symkls_DB_eta[m+1] = symkl_detailed_balance_eta
        esss[m+1] = (1. / (w_sym_eta**2).sum(0)).mean()
    reused = (q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu)
    metric_step = {"symKL_DB_eta" : symkls_DB_eta, "loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused
