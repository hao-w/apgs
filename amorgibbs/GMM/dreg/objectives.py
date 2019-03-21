import torch
import torch.nn as nn
from utils import *

def rws_dreg(enc_init, enc_local, x, z_t, N, K, D, SAMPLE_DIM, BATCH_DIM, num_samples, batch_size):
    q_eta, p_eta = enc_init(x) 
    q_z, p_z = enc_local(q_eta, x)

    log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)

    zs = q_z['zs'].value
    means = q_eta['means'].value.view(num_samples, batch_size, K, D)
    precisions = q_eta['precisions'].value.view(num_samples, batch_size, K, D)
    ll = loglikelihood(x, zs, means, precisions, D)
    log_weights = log_p_eta + log_p_z - log_q_eta - log_q_z + ll

    weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
    eubo = torch.mul(weights, log_weights).sum(0).mean()
    elbo = log_weights.mean()
    ess = (1. / (weights ** 2).sum(0)).mean()
    loss = torch.mul(weights ** 2 - weights, log_weights).sum(0).mean()

    ## KL
    kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kls_step(x, z_t, enc_init, enc_local, q_eta, q_z, N, K, D, num_samples, batch_size)
    ##
    KL_eta_ex = torch.mul(weights, kl_eta_ex).sum(0).mean()
    KL_eta_in = torch.mul(weights, kl_eta_in).sum(0).mean()
    KL_z_ex = torch.mul(weights, kl_z_ex).sum(0).mean()
    KL_z_in = torch.mul(weights, kl_z_in).sum(0).mean()  
    
    return loss, eubo, elbo, ess, KL_eta_ex, KL_eta_in, KL_z_ex, KL_z_in

def rws_nonrep(enc_init, enc_local, x, z_t, N, K, D, SAMPLE_DIM, BATCH_DIM, num_samples, batch_size):
    q_eta, p_eta = enc_init(x) 
    q_z, p_z = enc_local(q_eta, x)

    log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)

    zs = q_z['zs'].value
    means = q_eta['means'].value.view(num_samples, batch_size, K, D)
    precisions = q_eta['precisions'].value.view(num_samples, batch_size, K, D)
    ll = loglikelihood(x, zs, means, precisions, D)
    log_weights = log_p_eta + log_p_z - log_q_eta - log_q_z + ll
    ##
    weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
    eubo = torch.mul(weights, log_weights).sum(0).mean()
    elbo = log_weights.mean()
    ess = (1. / (weights ** 2).sum(0)).mean()    
    ## KL
    kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kls_step(x, z_t, enc_init, enc_local, q_eta, q_z, N, K, D, num_samples, batch_size)
    ##
    KL_eta_ex = torch.mul(weights, kl_eta_ex).sum(0).mean()
    KL_eta_in = torch.mul(weights, kl_eta_in).sum(0).mean()
    KL_z_ex = torch.mul(weights, kl_z_ex).sum(0).mean()
    KL_z_in = torch.mul(weights, kl_z_in).sum(0).mean()  
    
    return eubo, elbo, ess, KL_eta_ex, KL_eta_in, KL_z_ex, KL_z_in