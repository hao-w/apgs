import torch
import torch.nn as nn
from utils import *
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch

def Eubo_eta_nc(enc_mu, enc_tau, data, K, D, sample_size, batch_size):
    obs = data[:, :, :, :2]
    states = data[:, :, :, 2:]
    stat1, stat2, stat3 = data_to_stats(obs, states, K, D)
    q_tau, p_tau = enc_tau(stat1, stat2, stat3)
    obs_tau = q_tau['precisions'].value.view(sample_size, batch_size, K, D)
    q_mu, p_mu = enc_mu(stat1, stat2, stat3, obs_tau)
    ## for individual importance weight, S * B * K
    log_q_mu = q_mu['means'].log_prob.sum(-1)
    log_q_tau = q_tau['precisions'].log_prob.sum(-1)
    log_p_mu = p_mu['means'].log_prob.sum(-1)
    log_p_tau = p_tau['precisions'].log_prob.sum(-1)

    obs_mu = q_mu['means'].value.view(sample_size, batch_size, K, D)
    obs_sigma = 1. / obs_tau.sqrt()
    log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=True)
    log_weights = log_obs + log_p_mu + log_p_tau - log_q_mu - log_q_tau
    weights = F.softmax(log_weights, 0).detach()
    eubo = (weights * log_weights).sum(0).sum(-1).mean()
    elbo = log_weights.sum(-1).mean()
    ess = (1. / (weights**2).sum(0)).mean(-1).mean()
    return eubo, elbo, ess, weights

def Eubo_eta_ng(enc_eta, obs, states, K, D, sample_size, batch_size):
    stat1, stat2, stat3 = data_to_stats(obs, states, K, D)
    q, p, q_nu = enc_eta(stat1, stat2, stat3)
    ## for individual importance weight, S * B * K
    log_q_mu = q['means'].log_prob.sum(-1)
    log_q_tau = q['precisions'].log_prob.sum(-1)
    log_p_mu = p['means'].log_prob.sum(-1)
    log_p_tau = p['precisions'].log_prob.sum(-1)

    obs_mu = q['means'].value.view(sample_size, batch_size, K, D)
    obs_sigma = 1. / (q['precisions'].value.view(sample_size, batch_size, K, D)).sqrt()

    log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=True)
    log_weights = log_obs + log_p_mu + log_p_tau - log_q_mu - log_q_tau
    weights = F.softmax(log_weights, 0).detach()
    eubo = (weights * log_weights).sum(0).sum(-1).mean()
    elbo = log_weights.sum(-1).mean()
    ess = (1. / (weights**2).sum(0)).mean(-1).mean()
    ## KL with conjugate prior
    post_alpha, post_beta, post_mu, post_nu = Post_mu_tau(stat1, stat2, stat3, enc_eta.prior_alpha, enc_eta.prior_beta, enc_eta.prior_mu, enc_eta.prior_nu, D)
    q_mu = q['means'].dist.loc
    q_alpha = q['precisions'].dist.concentration
    q_beta = q['precisions'].dist.rate
    kl_ex, kl_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    akl_ex = kl_ex.sum(-1).mean()
    akl_in = kl_in.sum(-1).mean()
    return eubo, elbo, ess, akl_ex, akl_in

def Eubo_eta_ng_stat(enc_eta, data, K, D, sample_size, batch_size):
    obs = data[:, :, :, :2]
    states = data[:, :, :, 2:]
    stat1_t, stat2_t, stat3_t = data_to_stats(obs, states, K, D)
    q, p, q_nu = enc_eta(data)
    ## for individual importance weight, S * B * K
    log_q_mu = q['means'].log_prob.sum(-1)
    log_q_tau = q['precisions'].log_prob.sum(-1)
    log_p_mu = p['means'].log_prob.sum(-1)
    log_p_tau = p['precisions'].log_prob.sum(-1)

    obs_mu = q['means'].value.view(sample_size, batch_size, K, D)
    obs_sigma = 1. / (q['precisions'].value.view(sample_size, batch_size, K, D)).sqrt()

    log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=True)
    log_weights = log_obs + log_p_mu + log_p_tau - log_q_mu - log_q_tau
    weights = F.softmax(log_weights, 0).detach()
    eubo = (weights * log_weights).sum(0).sum(-1).mean()
    elbo = log_weights.sum(-1).mean()
    ess = (1. / (weights**2).sum(0)).mean(-1).mean()
    ## KL with conjugate prior
    post_alpha, post_beta, post_mu, post_nu = Post_mu_tau(stat1_t, stat2_t, stat3_t, enc_eta.prior_alpha, enc_eta.prior_beta, enc_eta.prior_mu, enc_eta.prior_nu, D)
    q_mu = q['means'].dist.loc
    q_alpha = q['precisions'].dist.concentration
    q_beta = q['precisions'].dist.rate
    kl_ex, kl_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    akl_ex = kl_ex.sum(-1).mean()
    akl_in = kl_in.sum(-1).mean()
    return eubo, elbo, ess, akl_ex, akl_in
