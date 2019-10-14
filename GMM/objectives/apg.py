import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward import *
import probtorch
from torch import logsumexp
from kls import *

def APG(models, ob, mcmc_steps):
    """
    Learn both forward kernels f_z, f_eta and assume backward kernels to be the same,
    involving gradient of EUBO and ELBO w.r.t forward parameters,
    in order to NOT use REINFORCE or Reparameterization.
    """
    (os_eta, f_z, f_eta) = models
    metrics = {'loss' : [], 'ess' : []}

    loss_os, ess_os, w_z, ob_tau, ob_mu, z = Init_eta(os_eta, f_z, ob)
    metrics['loss'].append(loss_os.unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0))
    for m in range(mcmc_steps):
        if m == 0:
            z = Resample(z, w_z, idw_flag=False) ## resample state
        else:
            z = Resample(z, w_z, idw_flag=True)

        loss_eta, ess_eta, w_eta, ob_tau, ob_mu = Update_eta(f_eta, f_eta, ob, z, ob_tau, ob_mu)
        ob_tau = Resample(ob_tau, w_eta, idw_flag=True) ## resample precisions
        ob_mu = Resample(ob_mu, w_eta, idw_flag=True) ## resample centers
        loss_z, ess_z, w_z, z = Update_z(f_z, f_z, ob, ob_tau, ob_mu, z) ## update cluster assignments
        metrics['loss'].append((loss_eta + loss_z).unsqueeze(0))
        metrics['ess'].append((ess_eta + ess_z).unsqueeze(0) / 2)
    reused = (z)
    return metrics, reused

def APG_test(models, ob, mcmc_steps, EPS=None):
    """
    APG objective used at test time
    NOTE: need to implement an interface
    which returns different variables needed during training and testing
    """
    (os_eta, f_z, f_eta) = models
    metrics = {'kl_ex_eta' : [], 'kl_ex_z' : [], 'kl_in_eta' : [], 'kl_in_z' : []}
    E_tau, E_mu, E_z, ess_os, w_os, ob_tau, ob_mu, z = Init_eta(os_eta, f_z, ob, training=False)
    # metrics['samples'].append((E_tau, E_mu, E_z))
    # metrics['ess'].append(ess_os.unsqueeze(-1))
    # metrics['log_joint'].append(log_joint.mean(0).unsqueeze(-1))
    # kl_step = kl_train(models, ob, z, EPS)
    # metrics['kl_in'].append(((kl_step['kl_eta_in'] + kl_step['kl_z_in']) / 2).unsqueeze(-1))
    # metrics['kl_ex'].append(((kl_step['kl_eta_ex'] + kl_step['kl_z_ex']) / 2).unsqueeze(-1))
    for m in range(mcmc_steps):
        if m == 0:
            z = Resample(z, w_os, idw_flag=False) ## resample state
        else:
            z = Resample(z, w_z, idw_flag=True)
        E_tau, E_mu, ess_eta, w_eta, ob_tau, ob_mu = Update_eta(f_eta, f_eta, ob, z, ob_tau, ob_mu, training=False)
        ob_tau = Resample(ob_tau, w_eta, idw_flag=True) ## resample precisions
        ob_mu = Resample(ob_mu, w_eta, idw_flag=True) ## resample centers
        E_z, ess_z, w_z, z = Update_z(f_z, f_z, ob, ob_tau, ob_mu, z, training=False) ## update cluster assignments

        # metrics['log_joint'].append((p1_joint + prior_eta).mean(0).unsqueeze(-1))
        # metrics['ess_eta'].append(ess_eta.unsqueeze(-1))
        # w_joint = F.softmax(log_w_eta.sum(-1) + log_w_z.sum(-1), 0).detach()
        # ess_joint = (1. / (w_joint**2).sum(0)).cpu()
        # metrics['ess_z'].append(ess_z.unsqueeze(-1))
        if m == 4 or m == 9 or m == 14:
            kl_step = kl_train(models, ob, z, EPS)
            metrics['kl_in_eta'].append(kl_step['kl_eta_in'].unsqueeze(-1))
            metrics['kl_ex_eta'].append(kl_step['kl_eta_ex'].unsqueeze(-1))
            metrics['kl_in_z'].append(kl_step['kl_z_in'].unsqueeze(-1))
            metrics['kl_ex_z'].append(kl_step['kl_z_ex'].unsqueeze(-1))
    #     metrics['ess'].append((ess_eta + ess_z).unsqueeze(0) / 2)
    # metrics['log_joint'] = torch.cat(metrics['log_joint'], -1)
    # metrics['ess'] = torch.cat(metrics['ess'], -1)
    # metrics['ess_eta'] = torch.cat(metrics['ess_eta'], -1)
    metrics['kl_in_eta'] = torch.cat(metrics['kl_in_eta'], -1)
    metrics['kl_ex_eta'] = torch.cat(metrics['kl_ex_eta'], -1)
    metrics['kl_in_z'] = torch.cat(metrics['kl_in_z'], -1)
    metrics['kl_ex_z'] = torch.cat(metrics['kl_ex_z'], -1)
    return metrics
