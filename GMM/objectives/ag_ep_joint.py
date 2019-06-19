import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward import *
import probtorch
"""
Joint Sampling, i.e. the Gibbs block is \{ \mu, \tau, \z \}

"""

def EUBO_init_eta_joint_both(models, obs, SubTrain_Params):
    """
    jointly sampling all variables in a full update and compute the joint IW
    """
    (device, sample_size, batch_size, N, K, D, mcmc_size) = SubTrain_Params
    losss = torch.zeros(mcmc_size+1).cuda().to(device)
    esss = torch.zeros(mcmc_size+1).cuda().to(device)
    (oneshot_eta, enc_eta, enc_z) = models
    model_os = (oneshot_eta, enc_z)
    obs_tau, obs_mu, state, log_w_f_z, q_eta, p_eta, q_z, p_z = Init_step_eta(model_os, obs, N, K, D, sample_size, batch_size)
    w_f_z = F.softmax(log_w_f_z, 0).detach()

    losss[0] = (w_f_z * log_w_f_z).sum(0).mean()  ## weights S * B
    esss[0] = (1. / (w_f_z**2).sum(0)).mean()
    for m in range(mcmc_size):
        state = resample_state(state, w_f_z, idw_flag=False) ## resample state
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        obs_tau_prev = obs_tau
        obs_mu_prev = obs_mu
        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
        state_prev = state
        state = q_z['zs'].value
        log_p = p_eta['means'].log_prob.sum(-1).sum(-1) + p_eta['precisions'].log_prob.sum(-1).sum(-1) + p_z['zs'].log_prob.sum(-1)
        log_q = q_eta['means'].log_prob.sum(-1).sum(-1) + q_eta['precisions'].log_prob.sum(-1).sum(-1) + q_z['zs'].log_prob.sum(-1)
        log_obs = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
        log_w_f = log_obs.sum(-1) + log_p - log_q ## S * B
        log_p_prev = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1).sum(-1) + Gamma(p_eta['precisions'].dist.concentration, p_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1).sum(-1) + cat(probs=p_z['zs'].dist.probs).log_prob(state_prev).sum(-1)
        log_q_prev = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1).sum(-1) + Gamma(q_eta['precisions'].dist.concentration, q_eta['precisions'].dist.rate).log_prob(obs_tau_prev).sum(-1).sum(-1) + cat(probs=q_z['zs'].dist.probs).log_prob(state_prev).sum(-1)
        log_obs_prev = Log_likelihood(obs, state_prev, obs_tau_prev, obs_mu_prev, K, D, cluster_flag=False)
        log_w_b = log_obs_prev.sum(-1) + log_p_prev - log_q_prev ## S * B
        w_sym = F.softmax(log_w_f - log_w_b, 0).detach()
        eubo_p_q = (w_sym * log_w_f).sum(0).sum(-1).mean()
        losss[m+1] = eubo_p_q
        ## symmetric KLs as metrics
        esss[m+1] = (1. / (w_sym**2).sum(0)).mean()
    reused = (state)
    metric_step = {"loss" : losss,  "ess" : esss}
    return losss.sum(), metric_step, reused
