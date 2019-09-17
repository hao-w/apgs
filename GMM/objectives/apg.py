import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward import *
import probtorch

def APG(models, ob, mcmc_steps):
    """
    Learn both forward kernels f_z, f_eta and backward kernels b_z, b_eta,
    involving gradient of EUBO and ELBO w.r.t forward parameters and backward
    parameters respectively, in order to NOT use REINFORCE or Reparameterization.
    """
    (os_eta, f_z, f_eta, b_z, b_eta) = models
    metrics = {'loss' : [], 'ess' : []}

    loss_os, ess_os, w_z, ob_tau, ob_mu, z = Init_eta(os_eta, f_z, ob)
    metrics['loss'].append(loss_os.unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0))
    for m in range(mcmc_steps):
        if m == 0:
            z = Resample(z, w_z, idw_flag=False) ## resample state
        else:
            z = Resample(z, w_z, idw_flag=True)

        loss_eta, ess_eta, w_eta, ob_tau, ob_mu = Update_eta(f_eta, b_eta, ob, z, ob_tau, ob_mu)
        ob_tau = Resample(ob_tau, w_eta, idw_flag=True) ## resample precisions
        ob_mu = Resample(ob_mu, w_eta, idw_flag=True) ## resample centers
        loss_z, ess_z, w_z, z = Update_z(f_z, b_z, ob, ob_tau, ob_mu, z) ## update cluster assignments
        metrics['loss'].append((loss_eta + loss_z).unsqueeze(0))
        metrics['ess'].append((ess_eta + ess_z).unsqueeze(0) / 2)
    reused = (z)
    return metrics, reused


def APG_test(models, ob, mcmc_steps, log_S):
    """
    APG objective used at test time
    NOTE: need to implement an interface
    which returns different variables needed during training and testing
    """
    (os_eta, f_z, f_eta, b_z, b_eta) = models
    metrics = {'samples' : [], 'elbos' : [], 'ess' : []}
    E_tau, E_mu, E_z, elbo_os, ess_os, w_z, ob_tau, ob_mu, z = Init_eta(os_eta, f_z, ob, training=False)
    metrics['samples'].append((E_tau, E_mu, E_z))
    metrics['elbos'].append(elbo_os.unsqueeze(0).unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0))
    for m in range(mcmc_steps):
        if m == 0:
            z = Resample(z, w_z, idw_flag=False) ## resample state
        else:
            z = Resample(z, w_z, idw_flag=True)
        E_tau, E_mu, log_prior_eta, ess_eta, w_eta, ob_tau, ob_mu = Update_eta(f_eta, b_eta, ob, z, ob_tau, ob_mu, training=False)
        ob_tau = Resample(ob_tau, w_eta, idw_flag=True) ## resample precisions
        ob_mu = Resample(ob_mu, w_eta, idw_flag=True) ## resample centers
        E_z, ll, log_prior_z, ess_z, w_z, z = Update_z(f_z, b_z, ob, ob_tau, ob_mu, z, training=False) ## update cluster assignments
        sum_log_Ex = BP(m+1, os_eta, f_z, f_eta, b_z, b_eta, ob, ob_tau, ob_mu, z, log_S)
        log_p_zM_x = ll.sum(-1) + log_prior_z.sum(-1) + log_prior_eta.sum(-1) # S * B
        # print(log_p_zM_x.shape)
        elbo = log_p_zM_x.mean().cpu() - sum_log_Ex
        metrics['samples'].append((E_tau, E_mu, E_z))
        metrics['elbos'].append(elbo.unsqueeze(0))
        metrics['ess'].append((ess_eta.unsqueeze(0) + ess_z.unsqueeze(0)) / 2)
    return metrics
