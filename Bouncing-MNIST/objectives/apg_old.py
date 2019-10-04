import torch
import torch.nn as nn
import probtorch
from forward_backward import Init_where_what, Resample_what, APG_where, APG_what

def APG(models, AT, K, D, frames, mcmc_steps, mnist_mean):
    """
    Start with the mnist_mean template,
    and iterate over z_where_t and z_what
    """
    metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : []}
    (enc_coor, dec_coor, enc_digit, dec_digit) = models
    phi_loss, theta_loss, ess, w_what, z_where, z_what = Init_where_what(enc_coor, dec_coor, enc_digit, dec_digit, AT, frames, mnist_mean)
    metrics['phi_loss'].append(phi_loss.unsqueeze(0))
    metrics['theta_loss'].append(theta_loss.unsqueeze(0))
    metrics['ess'].append(ess.unsqueeze(0).detach())
    for m in range(mcmc_steps):
        z_what = Resample_what(z_what, w_what)
        phi_loss_where, theta_loss_where, ess_where, w_where, z_where = APG_where(enc_coor, dec_coor, dec_digit, AT, frames, z_what, z_where_old=z_where)
        phi_loss_what, theta_loss_what, ess_what, w_what, z_what = APG_what(enc_digit, dec_digit, AT, frames, z_where, z_what_old=z_what)
        metrics['phi_loss'].append((phi_loss_what + phi_loss_where).unsqueeze(0))
        metrics['theta_loss'].append((theta_loss_what + theta_loss_where).unsqueeze(0))
        metrics['ess'].append((ess_what + ess_where).unsqueeze(0).detach() / 2)
    return metrics


# def APG_test(models, K, D, frames, mcmc_steps, mnist_mean, crop):
#     """
#     Start with the mnist_mean template,
#     and iterate over z_where_t and z_what
#     """
#     metrics = {'samples' : [], 'recon' : [], 'll' : [], 'ess' : []}
#     (os_coor, enc_coor, dec_coor, enc_digit, dec_digit) = models
#     E_where, E_what, recon, ess, w_what, z_where, z_what, ll = Init_step(os_coor, dec_coor, enc_digit, dec_digit, K, D, frames, crop, mnist_mean, training=False)
#     metrics['samples'].append((E_where.cpu(), E_what.cpu()))
#     metrics['recon'].append(recon.cpu())
#     metrics['ess'].append(ess.cpu().unsqueeze(0).detach())
#     metrics['ll'].append(ll.cpu().unsqueeze(0))
#     for m in range(mcmc_steps):
#         z_what = Resample_what(z_what, w_what)
#         E_where, _, ess_where, w_where, z_where, _ = Update_where( enc_coor, dec_coor, dec_digit, frames, crop, z_what, z_where_old=z_where, training=False)
#         z_where = Resample_where(z_where, w_where)
#         E_what, recon, ess_what, w_what, z_what, ll = Update_what(enc_digit, dec_digit, frames, crop, z_where, z_what_old=z_what, training=False)
#         metrics['samples'].append((E_where.cpu(), E_what.cpu()))
#         metrics['recon'].append(recon.cpu())
#         metrics['ess'].append((ess_where.cpu().unsqueeze(0) + ess_what.cpu().unsqueeze(0)) / 2)
#         metrics['ll'].append(ll.cpu().unsqueeze(0))
#     return metrics
