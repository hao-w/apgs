import torch
import torch.nn as nn
from utils import Resample
from forward_backward import *
import probtorch

def VAE(models, optimizer, ob, mcmc_steps, K):
    """
    VAE 'one-shot' encoder as baseline
    """
    S, B, N, D = ob.shape
    (os_mu, f_state, f_angle, dec_x) = models
    metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : []}
    phi_loss_os, theta_loss_os, ess_os, w_os, mu, state, angle = Init_mu(os_mu, f_state, f_angle, dec_x, ob, K)
    metrics['phi_loss'].append(phi_loss_os.unsqueeze(0))
    metrics['theta_loss'].append(theta_loss_os.unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0).detach())
    return metrics

def VAE_test(models, ob, mcmc_steps, K, log_S):
    """
    APG objective used at test time
    NOTE: need to implement an interface
    which returns different variables needed during training and testing
    """
    (os_mu, f_state, f_angle, dec_x) = models
    metrics = {'samples' : [], 'recon' : [], 'log_joint' : [], 'elbos' : [], 'ess' : []}
    E_recon, E_mu, E_z, log_joint_os, elbo_os, ess_os, w_os, mu, state, angle = Init_mu(os_mu, f_state, f_angle, dec_x, ob, K, training=False)
    metrics['samples'].append((E_mu, E_z))
    metrics['recon'].append(E_recon)
    metrics['elbos'].append(elbo_os.unsqueeze(0).unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0))
    metrics['log_joint'].append(log_joint_os.unsqueeze(0))
    return metrics
