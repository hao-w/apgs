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
    Learn neural gibbs samplers for both eta and z,
    non-reparameterized-style gradient estimation
    """
    (device, sample_size, batch_size, noise_sigma, N, K, D, mcmc_size) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    
    DBs_eta = torch.zeros(mcmc_size).cuda().to(device)
    DBs_z = torch.zeros(mcmc_size).cuda().to(device)
    DBs_rad = torch.zeros(mcmc_size).cuda().to(device)
        
    gaps_eta = torch.zeros(mcmc_size).cuda().to(device)
    gaps_z = torch.zeros(mcmc_size).cuda().to(device)
    gaps_rad = torch.zeros(mcmc_size+1).cuda().to(device)

    obs_mu, state, rad, log_w_f = Init_step_eta(models, obs, noise_sigma, N, K, D, sample_size, batch_size)
    w_f = F.softmax(log_w_f, 0).detach()

    (oneshot_eta, enc_eta, enc_z, enc_rad) = models
    losss[0] = (w_f * log_w_f).sum(0).mean()  ## weights S * B
    gaps_rad[0] = (w_f * log_w_f).sum(0).mean() - log_w_f.mean()
    esss[0] = (1. / (w_f**2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            obs_mu = resample_mu(obs_mu, w_f, idw_flag=False) ## resample state
            rad = resample_rad(rad, w_f, idw_flag=False)
        else:
            rad = resample_rad(rad, w_f_rad, idw_flag=True)
            
        q_z, p_z = enc_z.forward(obs, obs_mu, rad, noise_sigma, N, K, sample_size, batch_size)
        state, log_w_z_f, log_w_z_b = Incremental_z(q_z, p_z, obs, obs_mu, rad, noise_sigma, K, D, state)
        DB_z, eubo_p_q_z, gap_z, w_sym_z, w_f_z = detailed_balances(log_w_z_f, log_w_z_b)
        
        state = resample_state(state, w_f_z, idw_flag=True)
        
        q_eta, p_eta = enc_eta(obs, state, K, sample_size, batch_size)
        obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta(q_eta, p_eta, obs, state, rad, noise_sigma, K, D, obs_mu)
        DB_eta, eubo_p_q_eta, gap_eta, w_sym_eta, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b)
        
        obs_mu = resample_mu(obs_mu, w_f_eta, idw_flag=True) ## resample eta
        
        q_rad, p_rad = enc_rad.forward(obs, state, obs_mu, D, sample_size, batch_size)
        rad, log_w_rad_f, log_w_rad_b = Incremental_rad(q_rad, p_rad, obs, state, obs_mu, noise_sigma, K, D, rad)
        DB_rad, eubo_p_q_rad, gap_rad, w_sym_rad, w_f_rad = detailed_balances(log_w_rad_f, log_w_rad_b)

        losss[m+1] = eubo_p_q_eta + eubo_p_q_z + eubo_p_q_rad
        gaps_eta[m] = gap_eta
        gaps_z[m] = gap_z
        gaps_rad[m+1] = gap_rad
        ## symmetric KLs as metrics
        DBs_eta[m] = DB_eta
        DBs_z[m] = DB_z
        DBs_rad[m] = DB_rad
        
        esss[m+1] = ((1. / (w_sym_eta**2).sum(0)).mean() + (1. / (w_sym_z**2).sum(0)).mean() + (1. / (w_sym_rad**2).sum(0)).mean() ) / 3
    reused = (q_eta, p_eta, q_z, p_z, q_rad, p_rad)
    metric_step = {"DB_eta" : DBs_eta, "DB_z" : DBs_z, "DB_rad" : DBs_rad, "gap_eta" : gaps_eta, "gap_z" : gaps_z, "gap_rad" : gaps_rad,"loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused
