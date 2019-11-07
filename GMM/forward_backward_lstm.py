import torch
import torch.nn.functional as F
from utils import *
from normal_gamma import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
import probtorch
from torch import logsumexp

def Init_eta(os_eta, f_z, ob, K=3, training=True):
    """
    One-shot predicts eta and z, like a normal VAE
    """
    q_eta, p_eta, q_nu = os_eta(ob, K)
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
        log_j = log_obs_n.sum(-1).detach().cpu() + log_p_z.sum(-1).cpu() + log_p_eta.sum(-1).cpu()
        E_z = q_z['zs'].dist.probs.mean(0)[0].cpu().data.numpy()
        E_mu = q_eta['means'].dist.loc.mean(0)[0].cpu().data.numpy()
        E_tau = (q_eta['precisions'].dist.concentration / q_eta['precisions'].dist.rate).mean(0)[0].cpu().data.numpy()
        return E_tau, E_mu, E_z, log_j, ess, w, ob_tau, ob_mu, z