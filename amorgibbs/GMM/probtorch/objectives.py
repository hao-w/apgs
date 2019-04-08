import torch
import torch.nn as nn
from utils import *
from kls import *

def gibbs_rws(enc_init, data, N, K, D, SAMPLE_DIM, BATCH_DIM, num_samples, batch_size):
    q_eta, p_eta = enc_init(data)

    log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
    means = q_eta['means'].value.view(num_samples, batch_size, K, D)
    precisions = q_eta['precisions'].value.view(num_samples, batch_size, K, D)
    ll = loglikelihood(data[:, :, :, :2], data[:, :, :, 2:], means, precisions, D)
    log_weights = log_p_eta - log_q_eta + ll

    weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
    eubo = torch.mul(weights, log_weights).sum(0).mean()
    elbo = log_weights.mean()
    ess = (1. / (weights ** 2).sum(0)).mean()
    return eubo, elbo, ess

def oneshot_rws(enc_init, enc_local, x, N, K, D, SAMPLE_DIM, BATCH_DIM, num_samples, batch_size):
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

    # ## KL
    # kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kls_step(x, zs, enc_init, enc_local, q_eta, q_z, N, K, D, num_samples, batch_size)
    # ##
    # KL_eta_ex = torch.mul(weights, kl_eta_ex).sum(0).mean()
    # KL_eta_in = torch.mul(weights, kl_eta_in).sum(0).mean()
    # KL_z_ex = torch.mul(weights, kl_z_ex).sum(0).mean()
    # KL_z_in = torch.mul(weights, kl_z_in).sum(0).mean()

    return loss, eubo, elbo, ess

def ag_rws_overall(enc_init, enc_global, enc_local, x, N, K, D, SAMPLE_DIM, BATCH_DIM, steps, num_samples, batch_size):
    log_incremental_weights = torch.zeros((steps, num_samples, batch_size)).cuda()
    log_uptonow_weights = torch.zeros((steps, num_samples, batch_size)).cuda()
    for m in range(steps):
        if m == 0:
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
            log_incremental_weights[m] = log_p_eta + log_p_z - log_q_eta - log_q_z + ll
            log_uptonow_weights[m] = log_incremental_weights[m]
        else:
            q_eta, p_eta = enc_global(q_z, x)
            q_z, p_z = enc_local(q_eta, x)
            log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            zs = q_z['zs'].value
            means = q_eta['means'].value.view(num_samples, batch_size, K, D)
            precisions = q_eta['precisions'].value.view(num_samples, batch_size, K, D)
            ll = loglikelihood(x, zs, means, precisions, D)
            log_incremental_weights[m] = log_p_eta + log_p_z - log_q_eta - log_q_z + ll
            log_uptonow_weights[m] = log_uptonow_weights[m-1] + log_incremental_weights[m]
    weights = torch.exp(log_uptonow_weights - logsumexp(log_uptonow_weights, dim=1).unsqueeze(1).repeat(1, num_samples, 1)).detach()
    eubo = torch.mul(weights, log_incremental_weights).sum(1).mean(0).mean()
    elbo = log_incremental_weights.mean(0).mean(0).mean()
    ess = (1. / (weights[-1] ** 2).sum(0)).mean()
    ## KL
    kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kls_step(x, enc_init, enc_global, enc_local, q_eta, q_z, N, K, D, num_samples, batch_size)
    ##
    # KL_eta_ex = kl_eta_ex.mean(0).mean()
    # KL_eta_in = kl_eta_in.mean(0).mean()
    # KL_z_ex = kl_z_ex.mean(0).mean()
    # KL_z_in = kl_z_in.mean(0).mean()
    ##
    KL_eta_ex = torch.mul(weights[-1], kl_eta_ex).sum(0).mean()
    KL_eta_in = torch.mul(weights[-1], kl_eta_in).sum(0).mean()
    KL_z_ex = torch.mul(weights[-1], kl_z_ex).sum(0).mean()
    KL_z_in = torch.mul(weights[-1], kl_z_in).sum(0).mean()

    return eubo, elbo, ess, KL_eta_ex, KL_eta_in, KL_z_ex, KL_z_in

def ag_rws_increment(enc_init, enc_global, enc_local, x, N, K, D, SAMPLE_DIM, BATCH_DIM, steps, num_samples, batch_size):
    log_incremental_weights = torch.zeros((steps, num_samples, batch_size)).cuda()
    log_uptonow_weights = torch.zeros((steps, num_samples, batch_size)).cuda()
    for m in range(steps):
        if m == 0:
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
            log_incremental_weights[m] = log_p_eta + log_p_z - log_q_eta - log_q_z + ll
            log_uptonow_weights[m] = log_incremental_weights[m]
        else:
            q_eta, p_eta = enc_global(q_z, x)
            q_z, p_z = enc_local(q_eta, x)
            log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            zs = q_z['zs'].value
            means = q_eta['means'].value.view(num_samples, batch_size, K, D)
            precisions = q_eta['precisions'].value.view(num_samples, batch_size, K, D)
            ll = loglikelihood(x, zs, means, precisions, D)
            log_incremental_weights[m] = log_p_eta + log_p_z - log_q_eta - log_q_z + ll
            log_uptonow_weights[m] = log_uptonow_weights[m-1] + log_incremental_weights[m]
    weights = torch.exp(log_incremental_weights - logsumexp(log_incremental_weights, dim=1).unsqueeze(1).repeat(1, num_samples, 1)).detach()
    eubo = torch.mul(weights, log_incremental_weights).sum(1).mean(0).mean()
    elbo = log_incremental_weights.mean(0).mean(0).mean()
    ess = (1. / (weights[-1] ** 2).sum(0)).mean()
    ## KL
    kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kls_step(x, enc_init, enc_global, enc_local, q_eta, q_z, N, K, D, num_samples, batch_size)
    ##
    # KL_eta_ex = kl_eta_ex.mean(0).mean()
    # KL_eta_in = kl_eta_in.mean(0).mean()
    # KL_z_ex = kl_z_ex.mean(0).mean()
    # KL_z_in = kl_z_in.mean(0).mean()
    ##
    KL_eta_ex = torch.mul(weights[-1], kl_eta_ex).sum(0).mean()
    KL_eta_in = torch.mul(weights[-1], kl_eta_in).sum(0).mean()
    KL_z_ex = torch.mul(weights[-1], kl_z_ex).sum(0).mean()
    KL_z_in = torch.mul(weights[-1], kl_z_in).sum(0).mean()

    return eubo, elbo, ess, KL_eta_ex, KL_eta_in, KL_z_ex, KL_z_in


def ag_rws_idw(enc_init, enc_global, enc_local, x, N, K, D, SAMPLE_DIM, BATCH_DIM, steps, num_samples, batch_size):
    log_incremental_weights = torch.zeros((steps, num_samples, batch_size)).cuda()
    log_uptonow_weights = torch.zeros((steps, num_samples, batch_size)).cuda()
    for m in range(steps):
        if m == 0:
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
            log_incremental_weights[m] = log_p_eta + log_p_z - log_q_eta - log_q_z + ll
            log_uptonow_weights[m] = log_incremental_weights[m]
        else:
            q_eta, p_eta = enc_global(q_z, x)
            q_z, p_z = enc_local(q_eta, x)
            log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            zs = q_z['zs'].value
            means = q_eta['means'].value.view(num_samples, batch_size, K, D)
            precisions = q_eta['precisions'].value.view(num_samples, batch_size, K, D)
            ll = loglikelihood(x, zs, means, precisions, D)
            log_incremental_weights[m] = log_p_eta + log_p_z - log_q_eta - log_q_z + ll
            log_uptonow_weights[m] = log_uptonow_weights[m-1] + log_incremental_weights[m]
    weights = torch.exp(log_uptonow_weights - logsumexp(log_uptonow_weights, dim=1).unsqueeze(1).repeat(1, num_samples, 1)).detach()
    eubo = torch.mul(weights, log_incremental_weights).sum(1).mean(0).mean()
    elbo = log_incremental_weights.mean(0).mean(0).mean()
    ess = (1. / (weights[-1] ** 2).sum(0)).mean()
    ## KL
    kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kls_step(x, enc_init, enc_global, enc_local, q_eta, q_z, N, K, D, num_samples, batch_size)
    ##
    # KL_eta_ex = kl_eta_ex.mean(0).mean()
    # KL_eta_in = kl_eta_in.mean(0).mean()
    # KL_z_ex = kl_z_ex.mean(0).mean()
    # KL_z_in = kl_z_in.mean(0).mean()
    ##
    KL_eta_ex = torch.mul(weights[-1], kl_eta_ex).sum(0).mean()
    KL_eta_in = torch.mul(weights[-1], kl_eta_in).sum(0).mean()
    KL_z_ex = torch.mul(weights[-1], kl_z_ex).sum(0).mean()
    KL_z_in = torch.mul(weights[-1], kl_z_in).sum(0).mean()

    return eubo, elbo, ess, KL_eta_ex, KL_eta_in, KL_z_ex, KL_z_in
