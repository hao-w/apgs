import torch
import torch.nn as nn
import probtorch
from utils import Resample
from forward_backward import *

# def APG(models, ob, mcmc_steps):
#     """
#     one-shot predicts z_what as global variables,
#     and iterate over z_where_t and z_what
#     """
#     metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : []}
#     for m in range(mcmc_steps):
#         if m == 0:
#             z_where = Resample()
#         else:
#             z_where = Resample()
#         Update_z_what()
#         z_what = Resample()
#         Update_z_where()
#         metrics['phi_loss'].append((phi_loss_what + phi_loss_where).unsqueeze(0))
#         metrics['theta_loss'].append((theta_loss_what + theta_loss_where).unsqueeze(0))
#         metrics['ess'].append((ess_what + ess_where).unsqueeze(0).detach() / 2)
#
#     return metrics


def UT1(models, frames, mcmc_steps):
    """
    Given true initial velocity, true templates,
    test if we can learn the rest of the things in a VAE
    frames : B * T * H * W
    templates : K * B * H * W
    """
    metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : []}
    (enc_coor, enc_digit, dec_coor, dec_digit) = models
    metrics['phi_loss'].append((phi_loss_what + phi_loss_where).unsqueeze(0))
    metrics['theta_loss'].append((theta_loss_what + theta_loss_where).unsqueeze(0))
    metrics['ess'].append((ess_what + ess_where).unsqueeze(0).detach() / 2)

    return metrics
