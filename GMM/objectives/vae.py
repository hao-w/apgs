import torch
import torch.nn as nn
from utils import *
import probtorch

def EUBO(models, ob, K):
    """
    Learn both forward kernels f_z, f_eta and backward kernels b_z, b_eta,
    involving gradient of EUBO and ELBO w.r.t forward parameters and backward
    parameters respectively, in order to NOT use REINFORCE or Reparameterization.
    """
    (vae_eta, vae_z) = models
    metrics = {'loss' : [], 'ess' : []}
    loss_os, ess_os, w_z, ob_tau, ob_mu, z = Init_eta(vae_eta, vae_z, ob, K)
    metrics['loss'].append(loss_os.unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0))

    reused = (z)
    return metrics, reused

def Init_eta(vae_eta, vae_z, ob, K, training=True):
    """
    One-shot predicts eta and z, like a normal VAE
    """
    q_eta, p_eta, q_nu = vae_eta(ob, K)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    ob_mu = q_eta['means'].value
    ob_tau = q_eta['precisions'].value
    q_z, p_z = vae_z.forward(ob, ob_tau, ob_mu)
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
        return E_tau, E_mu, E_z, log_joint, elbo, ess, w, ob_tau, ob_mu, z
