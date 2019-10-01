import torch
import torch.nn as nn
import probtorch
from utils import Resample_what, Resample_where
from forward_backward import *

def APG(models, frames, tj_b, tj_std, mcmc_steps, mnist_mean, crop):
    """
    Start with the mnist_mean template,
    and iterate over z_where_t and z_what
    """
    metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : []}
    (enc_coor, enc_digit, dec_digit) = models
    phi_loss, theta_loss, ess, w_what, z_where, z_what = Init_step(enc_coor, enc_digit, dec_digit, frames, tj_b, tj_std, crop, mnist_mean)
    metrics['phi_loss'].append(phi_loss.unsqueeze(0))
    metrics['theta_loss'].append(theta_loss.unsqueeze(0))
    metrics['ess'].append(ess.unsqueeze(0).detach())
    for m in range(mcmc_steps):
        z_what = Resample_what(z_what, w_what)
        phi_loss_where, theta_loss_where, ess_where, w_where, z_where = Update_where(enc_coor, dec_digit, frames, tj_b, tj_std, crop, z_what, z_where_old=z_where)
        z_where = Resample_where(z_where, w_where)
        phi_loss_what, theta_loss_what, ess_what, w_what, z_what = Update_what(enc_digit, dec_digit, frames, crop, z_where, z_what_old=z_what)
        metrics['phi_loss'].append((phi_loss_what + phi_loss_where).unsqueeze(0))
        metrics['theta_loss'].append((theta_loss_what + theta_loss_where).unsqueeze(0))
        metrics['ess'].append((ess_what + ess_where).unsqueeze(0).detach() / 2)

    return metrics
