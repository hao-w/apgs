import torch
import torch.nn as nn
from utils import *
import probtorch
from forward_backward_lstm import *


def VAE(models, ob, mcmc_steps):
    """
    Learn both forward kernels f_z, f_eta and assume backward kernels to be the same,
    involving gradient of EUBO and ELBO w.r.t forward parameters,
    in order to NOT use REINFORCE or Reparameterization.
    """
    (os_eta, f_z) = models
    metrics = {'loss' : [], 'ess' : []}

    loss_os, ess_os, w_z, ob_tau, ob_mu, z = Init_eta(os_eta, f_z, ob, K=3)
    metrics['loss'].append(loss_os.unsqueeze(0))
    metrics['ess'].append(ess_os.unsqueeze(0))
    reused = (z)
    return metrics, reused

def VAE_test(models, ob, mcmc_steps):
    """
    APG objective used at test time
    NOTE: need to implement an interface
    which returns different variables needed during training and testing
    """
    (os_eta, f_z) = models
    metrics = {'samples' : [], 'ess' : [], 'll' : []}
    E_tau, E_mu, E_z, ll, ess_os, w_z, ob_tau, ob_mu, z = Init_eta(os_eta, f_z, ob, training=False)
    metrics['samples'].append((E_tau, E_mu, E_z))
    metrics['ess'] = ess_os.unsqueeze(-1)
    metrics['ll'] = ll.mean(0).unsqueeze(-1)

    return metrics
