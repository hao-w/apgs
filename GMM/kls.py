import torch
import time
from utils import *
from normal_gamma import *
from forward_backward import *

def kl_train(models, ob, reused, EPS):
    (state) = reused
    (os_eta, f_z, f_eta) = models
    q_eta, p_eta, q_nu = f_eta(ob, state)
    ob_mu = q_eta['means'].value
    ob_tau = q_eta['precisions'].value
    q_z, p_z = f_z.forward(ob, ob_tau, ob_mu)
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_mu = q_eta['means'].dist.loc
    q_pi = q_z['zs'].dist.probs
    pr_alpha = p_eta['precisions'].dist.concentration
    pr_beta = p_eta['precisions'].dist.rate
    pr_mu = p_eta['means'].dist.loc
    pr_nu = f_eta.prior_nu
    pr_pi = p_z['zs'].dist.probs

    post_alpha, post_beta, post_mu, post_nu = Post_eta(ob, state, pr_alpha, pr_beta, pr_mu, pr_nu)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(ob, ob_tau, ob_mu, pr_pi)
    kl_z_ex, kl_z_in = kls_cats(q_pi.log(), post_logits, EPS)
    kl_step = {"kl_eta_ex" : kl_eta_ex.mean(-1).mean().item(),"kl_eta_in" : kl_eta_in.mean(-1).mean().item(),"kl_z_ex" : kl_z_ex.mean(-1).mean().item(),"kl_z_in" : kl_z_in.mean(-1).mean().item()}
    return kl_step
