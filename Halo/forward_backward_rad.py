import torch
import torch.nn.functional as F
from utils import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch

def Init_step_eta(models, obs, noise_sigma, K):
    """
    initialize eta, using oneshot encoder, and then update z using its (gibbs or neural gibbs) encoder
    return the samples and log_weights
    """

    (oneshot_eta, enc_eta, enc_z) = models
    q_eta, p_eta = oneshot_eta(obs, K)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['radi'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['radi'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    radi = q_eta['radi'].value

    q_z, p_z = enc_z.forward(obs, obs_mu, radi, noise_sigma)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = True_Log_likelihood_rad(obs, state, obs_mu, radi, noise_sigma, cluster_flag=False)
    log_weights = log_obs_n.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
    return obs_mu, radi, state, log_weights

def Incremental_eta(q_eta, p_eta, obs, state, noise_sigma, obs_mu_prev, radi_prev):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu ).
    """
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['radi'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['radi'].log_prob.sum(-1)
    obs_mu = q_eta['means'].value
    radi = q_eta['radi'].value
    log_obs = True_Log_likelihood_rad(obs, state, obs_mu, radi, noise_sigma, cluster_flag=True)
    log_w_forward = log_obs + log_p_eta - log_q_eta
    ## backward
    log_p_eta_prev = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(p_eta['radi'].dist.concentration, p_eta['radi'].dist.rate).log_prob(radi_prev).sum(-1)
    log_q_eta_prev = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(obs_mu_prev).sum(-1) + Gamma(q_eta['radi'].dist.concentration, q_eta['radi'].dist.rate).log_prob(radi_prev).sum(-1)
    log_obs_prev = True_Log_likelihood_rad(obs, state, obs_mu_prev, radi_prev, noise_sigma, cluster_flag=True)
    log_w_backward = log_obs_prev + log_p_eta_prev - log_q_eta_prev
    return obs_mu, radi, log_w_forward, log_w_backward


def Incremental_z(q_z, p_z, obs, obs_mu, radi, noise_sigma, state_prev):
    """
    Given the current samples for global variable (eta = mu),
    sample new local variable (state).
    """
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value
    log_obs = True_Log_likelihood_rad(obs, state, obs_mu, radi, noise_sigma, cluster_flag=False)
    log_w_forward = log_obs + log_p_z - log_q_z
    ## backward
    log_p_z_prev = cat(probs=p_z['zs'].dist.probs).log_prob(state_prev)
    log_q_z_prev = cat(probs=q_z['zs'].dist.probs).log_prob(state_prev)
    log_obs_prev = True_Log_likelihood_rad(obs, state_prev, obs_mu, radi, noise_sigma, cluster_flag=False)
    log_w_backward = log_obs_prev + log_p_z_prev - log_q_z_prev
    return state, log_w_forward, log_w_backward


def detailed_balances(log_w_f, log_w_b):
    """
    log_w_f : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_w_b : log \frac {p(x, z)} {q_\f (z | z', x)}
    """
    ## symmetric KLs, i.e., Expectation w.r.t. q_\f
    log_w_sym = log_w_f - log_w_b
    w_sym = F.softmax(log_w_sym, 0).detach()
    kl_f_b = - log_w_sym.sum(-1).mean()
    kl_b_f = (w_sym * log_w_sym).sum(0).sum(-1).mean()
    symkl_db = kl_f_b + kl_b_f
    ## "asymmetric detailed balance"
    w_f = F.softmax(log_w_f, 0).detach()
    eubo_p_qf = (w_sym * log_w_f).sum(0).sum(-1).mean()
    elbo_p_qf = log_w_f.sum(-1).mean()
    # symkl_p_qf = eubo_p_qf - elbo_p_qf
    #
    eubo_p_qb = (w_f * log_w_b).sum(0).sum(-1).mean()
    elbo_p_qb = (w_sym * log_w_b).sum(-1).mean()
    # symkl_p_qb = eubo_p_qb - elbo_p_qb
    eubo_p_q = eubo_p_qf
    gap_gibbs_q = (eubo_p_qf - elbo_p_qf)

    return symkl_db, eubo_p_q, gap_gibbs_q, w_sym, w_f
