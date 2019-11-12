import torch
import torch.nn as nn
from utils import Resample, S_Resample
from forward_backward import *
import probtorch

def APG(models, optimizer, ob, mcmc_steps, K):
    """
    one-shot predicts mu,
    and iterate over mu and (state, angle)
    """
    S, B, N, D = ob.shape
    (os_mu, f_state, f_angle, f_mu, dec_x) = models
    metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess_z' : [], 'ess_mu' : []}
    phi_loss_os, theta_loss_os, ess_os, w_os, mu, state, angle = Init_mu(os_mu, f_state, f_angle, dec_x, ob, K)

    metrics['phi_loss'].append(phi_loss_os.unsqueeze(0))
    metrics['theta_loss'].append(theta_loss_os.unsqueeze(0))
    # metrics['ess'].append(ess_os.unsqueeze(0).detach())
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
        metrics['ess_mu'].append(ess_mu.unsqueeze(-1).detach())
        metrics['ess_z'].append(ess_z.unsqueeze(-1).detach())
    metrics['ess_mu'] = torch.cat(metrics['ess_mu'], -1).mean(0)
    metrics['ess_z'] = torch.cat(metrics['ess_z'], -1).mean(0)
    return metrics

def APG_test(models, ob, mcmc_steps, K, DEVICE):
    """
    APG objective used at test time
    NOTE: need to implement an interface
    which returns different variables needed during training and testing
    """
    (os_mu, f_state, f_angle, f_mu, dec_x) = models
    metrics = {'samples' : [], 'recon' : [], 'log_joint' : [],'ess_z' : []}
    E_recon, E_mu, E_z, log_joint_os, ess_os, w_os, mu, state, angle = Init_mu(os_mu, f_state, f_angle, dec_x, ob, K, training=False)
    metrics['samples'].append((E_mu, E_z))
    metrics['recon'].append(E_recon)
    # metrics['ess'].append(ess_os.unsqueeze(-1))
    metrics['log_joint'].append(log_joint_os.mean(0).unsqueeze(-1))
    for m in range(mcmc_steps):
        if m == 0:
            # mu = S_Resample(mu, w_os, DEVICE, idw_flag=False)
            state = Resample(state, w_os, idw_flag=False) ## resample state
            angle = Resample(angle, w_os, idw_flag=False)
        else:
            mu = Resample(mu, w, idw_flag=False)
            state = Resample(state, w, idw_flag=False)
            angle = Resample(angle, w, idw_flag=False)
        E_mu, prior, ess_mu, log_w_mu, mu = Update_mu(f_mu, f_mu, dec_x, ob, state, angle, mu, K, training=False)
        # mu = Resample(mu, w_mu, idw_flag=True)
        # state = Resample(state, w_mu.sum(-1), idw_flag=False) ## resample state
        # angle = Resample(angle, w_mu.sum(-1), idw_flag=False)
        E_recon, E_z, log_joint_p, ess_z, log_w_z, state, angle = Update_state_angle(f_state, f_angle, f_state, f_angle, dec_x, ob, state, angle, mu, K, training=False)
        metrics['samples'].append((E_mu, E_z))
        metrics['recon'].append(E_recon)
        # (ess_max, _) = torch.max(ess_mu, -1)
        # print(ess_max)
        w = F.softmax(log_w_mu.sum(-1) + log_w_z.sum(-1), 0).detach()
        ess_z = (1. /(w**2).sum(0)).cpu()
        # metrics['ess_mu'].append(ess_mu.unsqueeze(-1))
        metrics['ess_z'].append(ess_z.unsqueeze(-1))
        metrics['log_joint'].append((prior+log_joint_p).mean(0).unsqueeze(-1))
    # metrics['ess_mu'] = torch.cat(metrics['ess_mu'], -1)
    metrics['ess_z'] = torch.cat(metrics['ess_z'], -1)
    metrics['log_joint'] = torch.cat(metrics['log_joint'], -1)
    return metrics
