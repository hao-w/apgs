import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch

def Eubo_gibbs_mu(enc_mu, obs, states, K, D, sample_size, batch_size):
    q, p = enc_mu(obs, states, sample_size, batch_size)
    log_q_mu = q['means'].log_prob.sum(-1)
    log_p_mu = p['means'].log_prob.sum(-1)
    obs_mu = q['means'].value
    log_obs = True_Log_likelihood(obs, states, obs_mu, K, D, radius=1.5, noise_sigma = 0.05, gpu=gpu2, cluster_flag=True)
    log_weights = log_obs + log_p_mu - log_q_mu
    weights = F.softmax(log_weights, 0).detach()
    eubo = (weights * log_weights).sum(0).sum(-1).mean()
    elbo = log_weights.sum(-1).mean()
    ess = (1. / (weights**2).sum(0)).mean(-1).mean()
    return eubo, elbo, ess

def Eubo_gibbs_z(enc_z, obs, obs_mu, obs_rad, N, K, D, sample_size, batch_size, gpu, decay_factor, noise_sigma):
    q_z, p_z = enc_z(obs, obs_mu, obs_rad, sample_size, batch_size, decay_factor)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob ## S * B * N
    states = q_z['zs'].value
    log_obs_n = True_Log_likelihood(obs, states, obs_mu, obs_rad, K, D, noise_sigma, gpu=gpu, cluster_flag=False)
    log_weights_local = log_obs_n + log_p_z - log_q_z
    weights_local = F.softmax(log_weights_local, 0).detach()
    eubo =(weights_local * log_weights_local).sum(0).sum(-1).mean()
    elbo = log_weights_local.sum(-1).mean()
    ess = (1. / (weights_local**2).sum(0)).mean()
    return eubo, elbo, ess
