import torch
import torch.nn.functional as F
from utils import *
from normal_gamma import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch

def Init_eta(os_eta, f_z, ob):
    """
    initialize eta, using oneshot encoder, and then update z using its (gibbs or neural gibbs) encoder
    return the samples and log_weights
    """
    q_eta, p_eta, q_nu = os_eta(ob)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    ob_mu = q_eta['means'].value
    ob_tau = q_eta['precisions'].value
    q_z, p_z = f_z.forward(ob, ob_tau, ob_mu)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(ob, state, ob_tau, ob_mu, cluster_flag=False)
    log_w = log_obs_n.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
    w = F.softmax(log_w, 0).detach()
    loss = (w * log_w).sum(0).mean()
    ess = (1. / (w**2).sum(0)).mean().unsqueeze(0)
    return loss, ess, w, ob_tau, ob_mu, state

def Update_eta(f_eta, ob, state, ob_tau_old, ob_mu_old):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu + tau).
    """
    q_eta, p_eta, q_nu = f_eta(ob, state)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    ob_mu = q_eta['means'].value
    ob_tau = q_eta['precisions'].value
    log_obs = Log_likelihood(ob, state, ob_tau, ob_mu, cluster_flag=True)
    log_f_w = log_obs + log_p_eta - log_q_eta
    ## backward
    log_p_eta_old = Normal(p_eta['means'].dist.loc, p_eta['means'].dist.scale).log_prob(ob_mu_old).sum(-1) + Gamma(p_eta['precisions'].dist.concentration, p_eta['precisions'].dist.rate).log_prob(ob_tau_old).sum(-1)
    log_q_eta_old = Normal(q_eta['means'].dist.loc, q_eta['means'].dist.scale).log_prob(ob_mu_old).sum(-1) + Gamma(q_eta['precisions'].dist.concentration, q_eta['precisions'].dist.rate).log_prob(ob_tau_old).sum(-1)
    log_obs_old = Log_likelihood(ob, state, ob_tau_old, ob_mu_old, cluster_flag=True)
    log_b_w = log_obs_old + log_p_eta_old - log_q_eta_old
    loss, _, w = Compose_IW(log_f_w, log_b_w)
    ess = (1. / (w**2).sum(0)).mean()
    return loss, ess, w, ob_tau, ob_mu

def Update_z(f_z, ob, ob_tau, ob_mu, state_old):
    """
    Given the current samples for global variable (eta = mu + tau),
    sample new local variable (state).
    """
    q_z, p_z = f_z.forward(ob, ob_tau, ob_mu)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    state = q_z['zs'].value
    log_obs = Log_likelihood(ob, state, ob_tau, ob_mu, cluster_flag=False)
    log_f_w = log_obs + log_p_z - log_q_z
    ## backward
    log_p_z_old = cat(probs=p_z['zs'].dist.probs).log_prob(state_old)
    log_q_z_old = cat(probs=q_z['zs'].dist.probs).log_prob(state_old)
    log_obs_old = Log_likelihood(ob, state_old, ob_tau, ob_mu, cluster_flag=False)
    log_b_w = log_obs_old + log_p_z_old - log_q_z_old
    loss, _, w = Compose_IW(log_f_w, log_b_w)
    ess = (1. / (w**2).sum(0)).mean()
    return loss, ess, w, state

# def Incremental_z_joint(q_z, p_z, obs, obs_tau, obs_mu, K, D, state_prev):
#     """
#     Given the current samples for global variable (eta = mu + tau),
#     sample new local variable (state).
#     """
#     log_p_z = p_z['zs'].log_prob
#     log_q_z = q_z['zs'].log_prob
#     state = q_z['zs'].value
#     log_obs = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
#     log_w_forward = log_obs + log_p_z - log_q_z
#     ## backward
#     log_p_z_prev = cat(probs=p_z['zs'].dist.probs).log_prob(state_prev)
#     log_q_z_prev = cat(probs=q_z['zs'].dist.probs).log_prob(state_prev)
#     log_obs_prev = Log_likelihood(obs, state_prev, obs_tau, obs_mu, K, D, cluster_flag=False)
#     log_w_backward = log_obs_prev.sum(-1) + log_p_z_prev.sum(-1) - log_q_z_prev.sum(-1)
#     return state, log_w_forward, log_w_backward

def Compose_IW(log_f_w, log_b_w):
    """
    log_f_w : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_b_w : log \frac {p(x, z)} {q_\f (z | z', x)}
    """
    ## symmetric KLs, i.e., Expectation w.r.t. q_\f
    log_w = log_f_w - log_b_w
    w = F.softmax(log_w, 0).detach()
    # kl_f_b = - log_w_sym.sum(-1).mean()
    # kl_b_f = (w_sym * log_w_sym).sum(0).sum(-1).mean()
    # symkl_db = kl_f_b + kl_b_f
    ## "asymmetric detailed balance"
    # w_f = F.softmax(log_w_f, 0).detach()
    loss = (w * log_f_w).sum(0).sum(-1).mean()

    return loss, log_w, w
