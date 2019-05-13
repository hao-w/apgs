import torch
import torch.nn as nn
from utils import *
from forward_backward import *

import probtorch

def EP(models, obs, SubTrain_Params):
    """
    oneshot encoder using exact same architecture from amor-gibbs
    initialize eta
    """
    (device, sample_size, batch_size, obs_rad, noise_sigma, N, K, D) = SubTrain_Params
    (oneshot_eta, enc_z) = models
    q_eta, p_eta = oneshot_eta(obs, K, D, sample_size, batch_size)
    log_p_eta = p_eta['means'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    q_z, p_z = enc_z.forward(obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = True_Log_likelihood(obs, state, obs_mu, obs_rad, noise_sigma, K, D, cluster_flag=False, fixed_radius=True)
    log_weights = log_obs_n.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
    weights = F.softmax(log_weights, 0).detach()
    ## EUBO, ELBO, ESS
    eubo = (weights * log_weights).sum(0)
    elbo = log_weights.mean(0)
    ess = (1. / (weights**2).sum(0))
    loss = eubo.mean()
    metric_step = {"eubo" : eubo, "elbo" : elbo, "ess" : ess}
    reused = (q_eta, p_eta, q_z, p_z)
    return loss, metric_step, reused