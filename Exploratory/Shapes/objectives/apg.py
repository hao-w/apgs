import torch
import torch.nn as nn
from utils import Resample
from forward_backward import *
import probtorch

def APG(models, optimizer, ob, mcmc_steps, K):
    """
    one-shot predicts mu,
    and iterate over mu and (state, angle)
    """
    S, B, N, D = ob.shape
    (os_mu, f_state, f_angle, f_mu, dec_x) = models
    metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : []}
    phi_loss_os, theta_loss_os, ess_os, w_os, mu, state, angle = Init_mu(os_mu, f_state, f_angle, dec_x, ob, K)

    metrics['phi_loss'].append(phi_loss_os.unsqueeze(0))
    metrics['theta_loss'].append(theta_loss_os.unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0).detach())
    for m in range(mcmc_steps):
        if m == 0:
            state = Resample(state, w_os, idw_flag=False) ## resample state
            angle = Resample(angle, w_os, idw_flag=False)
        else:
            state = Resample(state, w_z, idw_flag=True)
            angle = Resample(angle, w_z, idw_flag=True)
        phi_loss_mu, theta_loss_mu, ess_mu, w_mu, mu = Update_mu(f_mu, f_mu, dec_x, ob, state, angle, mu, K)
        mu = Resample(mu, w_mu, idw_flag=True)
        phi_loss_z, theta_loss_z, ess_z, w_z, state, angle = Update_state_angle(f_state, f_angle, f_state, f_angle, dec_x, ob, state, angle, mu, K)
        metrics['phi_loss'].append((phi_loss_mu + phi_loss_z).unsqueeze(0))
        metrics['theta_loss'].append((theta_loss_mu + theta_loss_z).unsqueeze(0))
        metrics['ess'].append((ess_mu + ess_z).unsqueeze(0).detach() / 2)
    return metrics

def APG_test(models, ob, mcmc_steps, K):
    """
    APG objective used at test time
    NOTE: need to implement an interface
    which returns different variables needed during training and testing
    """
    (os_mu, f_state, f_angle, f_mu, dec_x) = models
    metrics = {'samples' : [], 'recon' : [], 'log_joint' : [], 'ess' : []}
    E_recon, E_mu, E_z, E_angle, log_joint_os, elbo_os, ess_os, w_os, mu, state, angle = Init_mu(os_mu, f_state, f_angle, dec_x, ob, K, training=False)
    metrics['samples'].append((E_mu, E_z, E_angle))
    metrics['recon'].append(E_recon)
    metrics['ess'].append(ess_os.unsqueeze(0))
    metrics['log_joint'].append(log_joint_os.unsqueeze(0))
    for m in range(mcmc_steps):
        if m == 0:
            state = Resample(state, w_os, idw_flag=False) ## resample state
            angle = Resample(angle, w_os, idw_flag=False)
        else:
            state = Resample(state, w_z, idw_flag=True)
            angle = Resample(angle, w_z, idw_flag=True)
        E_mu, log_prior_mu, ess_mu, w_mu, mu = Update_mu(f_mu, f_mu, dec_x, ob, state, angle, mu, K, training=False)
        mu = Resample(mu, w_mu, idw_flag=True)
        E_recon, E_z, E_angle, ll, log_prior_z, ess_z, w_z, state, angle = Update_state_angle(f_state, f_angle, f_state, f_angle, dec_x, ob, state, angle, mu, K, training=False)
        log_joint = (ll.sum(-1) + log_prior_z.sum(-1) + log_prior_mu.sum(-1)).mean().cpu() # S * B
        metrics['samples'].append((E_mu, E_z, E_angle))
        metrics['recon'].append(E_recon)
        metrics['ess'].append((ess_mu.unsqueeze(0) + ess_z.unsqueeze(0)) / 2)
        metrics['log_joint'].append(log_joint.unsqueeze(0))
    return metrics
