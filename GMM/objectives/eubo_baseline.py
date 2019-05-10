import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward import *

import probtorch

def Eubo_baseline(obs, Model_Params, device, models):
    """
    oneshot encoder using exact same architecture from amor-gibbs
    initialize eta
    """
    (oneshot_eta, enc_z) = models
    (N, K, D, sample_size, batch_size) = Model_Params
    obs_tau, obs_mu, state, log_weights_state = Init_step_eta(obs, oneshot_eta, enc_z, N, K, D, sample_size, batch_size)
    weights_state = F.softmax(log_weights_state, 0).detach()
    ## EUBO, ELBO, ESS
    eubo = (weights_state * log_weights_state).sum(0).mean()
    elbo = log_weights_state.mean()
    ess = (1. / (weights_state**2).sum(0)).mean()
    loss = eubo
    metric_step = {"eubo" : eubo.item(), "elbo" : elbo.item(), "ess" : ess.item()}
    reused = None
    return loss, metric_step, reused

def Eubo_os_cfz(enc_eta, gibbs_z, obs, N, K, D, sample_size, batch_size, device, idw_flag=True):
    """
    oneshot encoder using exact same architecture from amor-gibbs
    initialize eta
    """
    ## update tau and mu -- global variables
    q_eta, p_eta, q_nu = enc_eta(obs, K, D)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    ## update z -- cluster assignments
    q_z, p_z = gibbs_z.forward(obs, obs_tau, obs_mu, N, K)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
    log_weights = log_obs_n.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1)
    weights = F.softmax(log_weights, 0).detach()
    ## EUBO, ELBO, ESS
    eubo = (weights * log_weights).sum(0).mean()
    elbo = log_weights.mean()
    ess = (1. / (weights**2).sum(0)).mean()
    return eubo, elbo, ess, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu
