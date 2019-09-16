import torch
import torch.nn.functional as F
from utils import *
from normal_gamma import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch

def Init_eta(os_eta, f_z, ob, training=True):
    """
    One-shot predicts eta and z, like a normal VAE
    """
    q_eta, p_eta, q_nu = os_eta(ob)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    ob_mu = q_eta['means'].value
    ob_tau = q_eta['precisions'].value
    q_z, p_z = f_z.forward(ob, ob_tau, ob_mu)
    log_p_z = p_z['zs'].log_prob
    log_q_z = q_z['zs'].log_prob
    z = q_z['zs'].value ## S * B * N * K
    log_obs_n = Log_likelihood(ob, z, ob_tau, ob_mu, cluster_flag=False)
    log_w = log_obs_n.sum(-1) + log_p_z.sum(-1) - log_q_z.sum(-1) + log_p_eta.sum(-1) - log_q_eta.sum(-1)
    w = F.softmax(log_w, 0).detach()
    if training :
        loss = (w * log_w).sum(0).mean()
        ess = (1. / (w**2).sum(0)).mean().unsqueeze(0)
        return loss, ess, w, ob_tau, ob_mu, z
    else:
        E_z = q_z['zs'].dist.probs.mean(0)[0].cpu().data.numpy()
        E_mu = q_eta['means'].dist.loc.mean(0)[0].cpu().data.numpy()
        E_tau = (q_eta['precisions'].dist.concentration / q_eta['precisions'].dist.rate).mean(0)[0].cpu().data.numpy()
        return E_tau, E_mu, E_z, w, ob_tau, ob_mu, z

def Update_eta(f_eta, b_eta, ob, state, ob_tau_old, ob_mu_old, training=True):
    """
    Given the current samples for local variable (state),
    update global variable (eta = mu + tau).
    """
    q_f_eta, p_f_eta, q_f_nu = f_eta(ob, state) ## forward kernel
    log_p_f_eta = p_f_eta['means'].log_prob.sum(-1) + p_f_eta['precisions'].log_prob.sum(-1)
    log_q_f_eta = q_f_eta['means'].log_prob.sum(-1) + q_f_eta['precisions'].log_prob.sum(-1)
    ob_mu = q_f_eta['means'].value
    ob_tau = q_f_eta['precisions'].value
    log_f_obs = Log_likelihood(ob, state, ob_tau, ob_mu, cluster_flag=True)
    log_f_w = log_f_obs + log_p_f_eta - log_q_f_eta
    q_b_eta, p_b_eta, q_b_nu = b_eta(ob, state, sampled=False, tau_old=ob_tau_old, mu_old=ob_mu_old) ## backward kernel
    log_p_b_eta = p_b_eta['means'].log_prob.sum(-1) + p_b_eta['precisions'].log_prob.sum(-1)
    log_q_b_eta = q_b_eta['means'].log_prob.sum(-1) + q_b_eta['precisions'].log_prob.sum(-1)
    log_b_obs = Log_likelihood(ob, state, ob_tau_old, ob_mu_old, cluster_flag=True)
    log_b_w = log_b_obs + log_p_b_eta - log_q_b_eta
    if training:
        loss, _, w = Compose_IW(log_f_w, log_q_f_eta, log_b_w, log_q_b_eta)
        ess = (1. / (w**2).sum(0)).mean()
        return loss, ess, w, ob_tau, ob_mu
    else:
        _, _, w = Compose_IW(log_f_w, log_q_f_eta, log_b_w, log_q_b_eta)
        E_mu = q_f_eta['means'].dist.loc.mean(0)[0].cpu().data.numpy()
        E_tau = (q_f_eta['precisions'].dist.concentration / q_f_eta['precisions'].dist.rate).mean(0)[0].cpu().data.numpy()
        return E_tau, E_mu, w, ob_tau, ob_mu

def Update_z(f_z, b_z, ob, ob_tau, ob_mu, z_old, training=True):
    """
    Given the current samples of global variable (eta = mu + tau),
    update local variable (state).
    """
    q_f_z, p_f_z = f_z.forward(ob, ob_tau, ob_mu)
    log_p_f_z = p_f_z['zs'].log_prob
    log_q_f_z = q_f_z['zs'].log_prob
    z = q_f_z['zs'].value
    log_f_obs = Log_likelihood(ob, z, ob_tau, ob_mu, cluster_flag=False)
    log_f_w = log_f_obs + log_p_f_z - log_q_f_z
    q_b_z, p_b_z = b_z.forward(ob, ob_tau, ob_mu, sampled=False, z_old=z_old) ## backward
    log_p_b_z = p_b_z['zs'].log_prob
    log_q_b_z = q_b_z['zs'].log_prob
    log_b_obs = Log_likelihood(ob, z_old, ob_tau, ob_mu, cluster_flag=False)
    log_b_w = log_b_obs + log_p_b_z - log_q_b_z
    if training:
        loss, _, w = Compose_IW(log_f_w, log_q_f_z, log_b_w, log_q_b_z)
        ess = (1. / (w**2).sum(0)).mean()
        return loss, ess, w, z
    else:
        _, _, w = Compose_IW(log_f_w, log_q_f_z, log_b_w, log_q_b_z)
        E_z = q_f_z['zs'].dist.probs.mean(0)[0].cpu().data.numpy()
        return E_z, w, z

def Compose_IW(log_f_w, log_q_f, log_b_w, log_q_b, training=True):
    """
    log_f_w : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_b_w : log \frac {p(x, z)} {q_\f (z | z', x)}

    self-normalized importance weights w := softmax(log_f_w = log_b_w).detach()
    loss := w * (-log_q_f) + (- log_q_b)
    """
    log_w = log_f_w - log_b_w
    w = F.softmax(log_w, 0).detach()
    loss = (w * ( - log_q_f)).sum(0).sum(-1).mean() - log_q_b.sum(-1).mean()
    return loss, log_w, w
