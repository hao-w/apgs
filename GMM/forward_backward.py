import torch
import torch.nn.functional as F
from utils import *
from normal_gamma import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch
from torch import logsumexp

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
    ess = (1. / (w**2).sum(0)).mean().cpu()
    if training :
        loss = (w * log_w).sum(0).mean()
        return loss, ess, w, ob_tau, ob_mu, z
    else:
        elbo = log_w.mean().cpu()
        log_joint = (log_obs_n.sum(-1) + log_p_z.sum(-1) + log_p_eta.sum(-1)).mean().cpu()
        E_z = q_z['zs'].dist.probs.mean(0)[0].cpu().data.numpy()
        E_mu = q_eta['means'].dist.loc.mean(0)[0].cpu().data.numpy()
        E_tau = (q_eta['precisions'].dist.concentration / q_eta['precisions'].dist.rate).mean(0)[0].cpu().data.numpy()
        return E_tau, E_mu, E_z, log_joint, elbo, ess, w, ob_tau, ob_mu, z, (- log_q_z.sum(-1) - log_q_eta.sum(-1)).detach()

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
    log_p_b_eta = (p_b_eta['means'].log_prob.sum(-1) + p_b_eta['precisions'].log_prob.sum(-1)).detach()
    log_q_b_eta = (q_b_eta['means'].log_prob.sum(-1) + q_b_eta['precisions'].log_prob.sum(-1)).detach()
    log_b_obs = Log_likelihood(ob, state, ob_tau_old, ob_mu_old, cluster_flag=True).detach()
    log_b_w = log_b_obs + log_p_b_eta - log_q_b_eta
    if training:
        loss, w = Compose_IW(log_f_w, log_q_f_eta, log_b_w, log_q_b_eta)
        ess = (1. / (w**2).sum(0)).mean().cpu()
        return loss, ess, w, ob_tau, ob_mu
    else:
        w = F.softmax(log_f_w - log_b_w, 0).detach()
        ess = (1. / (w**2).sum(0)).mean().cpu()
        E_mu = q_f_eta['means'].dist.loc.mean(0)[0].cpu().data.numpy()
        E_tau = (q_f_eta['precisions'].dist.concentration / q_f_eta['precisions'].dist.rate).mean(0)[0].cpu().data.numpy()
        return E_tau, E_mu, log_p_f_eta, ess, w, ob_tau, ob_mu,  (log_q_b_eta.sum(-1) - log_q_f_eta.sum(-1)).detach()

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
    log_p_b_z = p_b_z['zs'].log_prob.detach()
    log_q_b_z = q_b_z['zs'].log_prob.detach()
    log_b_obs = Log_likelihood(ob, z_old, ob_tau, ob_mu, cluster_flag=False).detach()
    log_b_w = log_b_obs + log_p_b_z - log_q_b_z
    if training:
        loss, w = Compose_IW(log_f_w, log_q_f_z, log_b_w, log_q_b_z)
        ess = (1. / (w**2).sum(0)).mean().cpu()
        return loss, ess, w, z
    else:
        w = F.softmax(log_f_w - log_b_w, 0).detach()
        ess = (1. / (w**2).sum(0)).mean().cpu()
        E_z = q_f_z['zs'].dist.probs.mean(0)[0].cpu().data.numpy()
        return E_z, log_f_obs, log_p_f_z, ess, w, z, (log_q_b_z.sum(-1) - log_q_f_z.sum(-1)).detach()
#
# def BP(M, os_eta, f_z, f_eta, b_z, b_eta, ob, ob_tau, ob_mu, z, log_S):
#     ob_tau_old = ob_tau
#     ob_mu_old = ob_mu
#     z_old = z
#     S = ob_mu.shape[0]
#     sum_log_Ex = []
#     for m in range(M, 0, -1):
#
#         q_b_eta, p_b_eta, q_b_nu = b_eta(ob, z_old)
#         ob_mu = q_b_eta['means'].value
#         ob_tau = q_b_eta['precisions'].value
#         log_q_b_eta = q_b_eta['means'].log_prob.sum(-1).detach() + q_b_eta['precisions'].log_prob.sum(-1).detach() # S * B * K
#         q_f_eta, p_f_eta, q_f_nu = f_eta(ob, z_old, sampled=False, tau_old=ob_tau_old, mu_old=ob_mu_old)
#         log_q_f_eta = q_f_eta['means'].log_prob.sum(-1).detach() + q_f_eta['precisions'].log_prob.sum(-1).detach() # S * B * K
#         log_ratio_eta = (log_q_f_eta - log_q_b_eta).sum(-1) ## S * 1
#
#         q_b_z, _ = b_z.forward(ob, ob_tau, ob_mu) ## backward
#         z = q_b_z['zs'].value
#         log_q_b_z = q_b_z['zs'].log_prob.detach() # S * B * N
#         q_f_z, p_f_z = f_z.forward(ob, ob_tau, ob_mu, sampled=False, z_old=z_old)
#         log_q_f_z = q_f_z['zs'].log_prob.detach() # S * B * N
#         log_ratio_z = (log_q_f_z - log_q_b_z).sum(-1) ## S * 1
#         ## Monte Carlo estimate that term
#         log_Ex_m = logsumexp(log_ratio_eta + log_ratio_z, dim=0).mean()
#         sum_log_Ex.append(log_Ex_m.unsqueeze(0))
#         ## assign the samples as old samples
#         ob_tau_old = ob_tau
#         ob_mu_old = ob_mu
#         z_old = z
#     q_os_eta, _, _ = os_eta(ob, sampled=False, tau_old=ob_tau_old, mu_old=ob_mu_old)
#     log_q_os_eta = q_os_eta['means'].log_prob.sum(-1).detach() + q_f_eta['precisions'].log_prob.sum(-1).detach() # S * B * K
#     q_os_z, _ = f_z.forward(ob, ob_tau_old, ob_mu_old, sampled=False, z_old=z_old)
#     log_q_os_z = q_os_z['zs'].log_prob.detach() # S * B * N
#     log_Ex_os = logsumexp(log_q_os_eta.sum(-1) + log_q_os_z.sum(-1), dim=0).mean() - log_S
#     sum_log_Ex.append(log_Ex_os.unsqueeze(0))
#     sum_log_Ex = torch.cat(sum_log_Ex, 0)
#     # print("back steps %d" % M)
#     # print(sum_log_Ex.shape)
#     return sum_log_Ex.sum().cpu()

def BP(M, models, ob, ob_tau, ob_mu, z, log_S):
    (os_eta, f_z, f_eta, b_z, b_eta) = models
    ob_tau_old = ob_tau
    ob_mu_old = ob_mu
    z_old = z
    S = ob_mu.shape[0]
    sum_log_Ex = []
    for m in range(M, 0, -1):
        q_b_z, p_b_z = b_z.forward(ob, ob_tau_old, ob_mu_old) ## backward
        z = q_b_z['zs'].value
        log_q_b_z = q_b_z['zs'].log_prob.detach().sum(-1) # S * B

        q_b_eta, _, _ = b_eta(ob, z)
        ob_mu = q_b_eta['means'].value
        ob_tau = q_b_eta['precisions'].value
        log_q_b_eta = q_b_eta['means'].log_prob.sum(-1).detach().sum(-1) + q_b_eta['precisions'].log_prob.sum(-1).detach().sum(-1) # S * B

        q_f_eta, _, _ = f_eta(ob, z, sampled=False, tau_old=ob_tau_old, mu_old=ob_mu_old)
        log_q_f_eta = q_f_eta['means'].log_prob.sum(-1).detach().sum(-1) + q_f_eta['precisions'].log_prob.sum(-1).detach().sum(-1) # S * B

        q_f_z, _ = f_z.forward(ob, ob_tau_old, ob_mu_old, sampled=False, z_old=z_old)
        log_q_f_z = q_f_z['zs'].log_prob.detach().sum(-1) # S * B
        sum_log_Ex.append((log_q_f_z + log_q_f_eta - log_q_b_z - log_q_b_eta).unsqueeze(0))
        ## assign the samples as old samples
        ob_tau_old = ob_tau
        ob_mu_old = ob_mu
        z_old = z
    q_os_eta, _, _ = os_eta(ob, sampled=False, tau_old=ob_tau_old, mu_old=ob_mu_old)
    log_q_os_eta = q_os_eta['means'].log_prob.sum(-1).detach().sum(-1) + q_f_eta['precisions'].log_prob.sum(-1).detach().sum(-1) # S * B
    q_os_z, _ = f_z.forward(ob, ob_tau_old, ob_mu_old, sampled=False, z_old=z_old)
    log_q_os_z = q_os_z['zs'].log_prob.detach().sum(-1) # S * B
    sum_log_Ex.append((log_q_os_eta + log_q_os_z).unsqueeze(0))
    log_Ex = logsumexp(torch.cat(sum_log_Ex, 0).sum(0), dim=0).mean() - log_S

    # print("back steps %d" % M)
    # print(sum_log_Ex.shape)
    return log_Ex.cpu()

def BP_IWAE(M, models, ob, ob_tau, ob_mu, z, log_S):
    """
    IWAE ELBO from Hierachical IW paper
    """
    (os_eta, f_z, f_eta, b_z, b_eta) = models
    ob_tau_old = ob_tau
    ob_mu_old = ob_mu
    z_old = z
    S = ob_mu.shape[0]
    sum_log_Ex = []
    for m in range(M, 0, -1):
        q_b_z, p_b_z = b_z.forward(ob, ob_tau_old, ob_mu_old) ## backward
        z = q_b_z['zs'].value
        log_q_b_z = q_b_z['zs'].log_prob.detach().sum(-1) # S * B

        q_b_eta, _, _ = b_eta(ob, z)
        ob_mu = q_b_eta['means'].value
        ob_tau = q_b_eta['precisions'].value
        log_q_b_eta = q_b_eta['means'].log_prob.sum(-1).detach().sum(-1) + q_b_eta['precisions'].log_prob.sum(-1).detach().sum(-1) # S * B

        q_f_eta, _, _ = f_eta(ob, z, sampled=False, tau_old=ob_tau_old, mu_old=ob_mu_old)
        log_q_f_eta = q_f_eta['means'].log_prob.sum(-1).detach().sum(-1) + q_f_eta['precisions'].log_prob.sum(-1).detach().sum(-1) # S * B

        q_f_z, _ = f_z.forward(ob, ob_tau_old, ob_mu_old, sampled=False, z_old=z_old)
        log_q_f_z = q_f_z['zs'].log_prob.detach().sum(-1) # S * B
        sum_log_Ex.append((log_q_f_z + log_q_f_eta - log_q_b_z - log_q_b_eta).unsqueeze(0))
        ## assign the samples as old samples
        ob_tau_old = ob_tau
        ob_mu_old = ob_mu
        z_old = z
    q_os_eta, _, _ = os_eta(ob, sampled=False, tau_old=ob_tau_old, mu_old=ob_mu_old)
    log_q_os_eta = q_os_eta['means'].log_prob.sum(-1).detach().sum(-1) + q_f_eta['precisions'].log_prob.sum(-1).detach().sum(-1) # S * B
    q_os_z, _ = f_z.forward(ob, ob_tau_old, ob_mu_old, sampled=False, z_old=z_old)
    log_q_os_z = q_os_z['zs'].log_prob.detach().sum(-1) # S * B
    sum_log_Ex.append((log_q_os_eta + log_q_os_z).unsqueeze(0))
    log_Ex = logsumexp(torch.cat(sum_log_Ex, 0).sum(0), dim=0).mean() - log_S

    # print("back steps %d" % M)
    # print(sum_log_Ex.shape)
    return log_Ex.cpu()

def Compose_IW(log_f_w, log_q_f, log_b_w, log_q_b):
    """
    log_f_w : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_b_w : log \frac {p(x, z)} {q_\f (z | z', x)}

    self-normalized importance weights w := softmax(log_f_w = log_b_w).detach()
    loss := w * (-log_q_f) + (- log_q_b)
    """
    w = F.softmax(log_f_w - log_b_w, 0).detach()
    loss = (w * ( - log_q_f)).sum(0).sum(-1).mean()
    # loss = (w * ( - log_q_f)).sum(0).sum(-1).mean() - log_q_b.sum(-1).mean()
    # elbo = log_w.sum(-1).mean().detach().cpu()
    return loss, w
