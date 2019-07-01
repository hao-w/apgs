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
def AG_dec(models, obs, SubTrain_Params):
    """
    NO Resampling
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    initialize eta
    """
    (device, sample_size, batch_size, obs_rad, noise_sigma, N, K, D, mcmc_size) = SubTrain_Params
    losss_phi = torch.zeros(mcmc_size+1).cuda().to(device)
    losss_theta = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
#     symkls_DB_eta = torch.zeros(mcmc_size+1).cuda().to(device)
#     symkls_DB_z = torch.zeros(mcmc_size+1).cuda().to(device)
    # gaps_eta = torch.zeros(mcmc_size+1).cuda().to(device)
    # gaps_z = torch.zeros(mcmc_size+1).cuda().to(device)
    obs_mu, state, w_f_z, loss_phi, loss_theta = Init_step_eta(models, obs, obs_rad, noise_sigma, N, K, D, sample_size, batch_size)
    (oneshot_eta, enc_eta, enc_z, dec_x) = models
    losss_phi[0] = loss_phi  ## weights S * B
    losss_theta[0] = loss_theta  ## weights S * B
#     symkls_DB_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
#     symkls_DB_z[0] = symkls_DB_eta[0] ##
    # gaps_eta[0] = (w_f_z * log_w_f_z).sum(0).mean() - log_w_f_z.mean()
    # gaps_z[0] =  gaps_eta[0]
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        else:
            state = resample_state(state, w_sym_z, idw_flag=True)
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, w_sym_eta, loss_phi_eta, loss_theta_eta  = Incremental_eta(dec_x, q_eta, p_eta, obs, state, obs_rad, noise_sigma, K, D, obs_mu)
        obs_mu = resample_mu(obs_mu, w_sym_eta, idw_flag=True) ## resample eta
        q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
        state, w_sym_z, loss_phi_z, loss_theta_z = Incremental_z(dec_x, q_z, p_z, obs, obs_mu, obs_rad, noise_sigma, K, D, state)
        losss_phi[m+1] = loss_phi_eta + loss_phi_z
        losss_theta[m+1] = loss_theta_eta + loss_theta_z
        # gaps_eta[m+1] = eubo_p_q_eta - elbo_p_q_eta
        # gaps_z[m+1] = eubo_p_q_z - elbo_p_q_z
        ## symmetric KLs as metrics
#         symkls_DB_eta[m+1] = symkl_detailed_balance_eta
#         symkls_DB_z[m+1] = symkl_detailed_balance_z
        esss[m+1] = ((1. / (w_sym_eta**2).sum(0)).mean() + (1. / (w_sym_z**2).sum(0)).mean() ) / 2
    # reused = (q_eta, p_eta, q_z, p_z)
    # metric_step = {"loss" : losss,  "ess" : esss}
    return losss_phi.sum(), losss_theta.sum(), esss.mean()
