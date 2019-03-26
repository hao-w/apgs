import torch.nn as nn
import torch
# from kls import *
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch import logsumexp


def shuffler(batch_Xs, N, K, D, batch_size):
    indices = torch.cat([torch.randperm(N).unsqueeze(0) for b in range(batch_size)])
    indices_Xs = indices.unsqueeze(-1).repeat(1, 1, D)
    return torch.gather(batch_Xs, 1, indices_Xs)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1e-1)

def log_joints_gmm(X, Z, mus, precisions, N, D, K, prior_mean, prior_nu, prior_alpha, prior_beta, prior_pi, num_samples, batch_size):
    log_probs = torch.zeros((num_samples, batch_size)).float()
    ## mus and sigmas
    log_probs = log_probs + Gamma(prior_alpha, prior_beta).log_prob(precisions).sum(-1).sum(-1)
    prior_sigma = 1. / torch.sqrt(prior_nu * precisions)
    log_probs = log_probs + Normal(prior_mean, prior_sigma).log_prob(mus).sum(-1).sum(-1)
    ## Z
    log_probs = log_probs + cat(prior_pi).log_prob(Z).sum(-1)
    ## log likelihoods
    log_probs = log_probs + loglikelihood(X, Z, mus, precisions, D)
    return log_probs

def loglikelihood(X, Z, mus, precisions, D):
    # log-likelihoods
    """
    X should be expanded and repeated along the sample dim
    """
    sigmas = 1. / torch.sqrt(precisions) ## S * B * K * D
    labels = Z.argmax(-1)
    labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    x_mus = torch.gather(mus, 2, labels_flat)
    x_sigmas = torch.gather(sigmas, 2, labels_flat)
    log_p_x = Normal(x_mus, x_sigmas).log_prob(X).sum(-1).sum(-1) # S * B
    return log_p_x

def loglikelihood_relaxed(X, Z, mus, precisions, D):
    # log-likelihoods
    """
    X should be expanded and repeated along the sample dim
    """
    sigmas = 1. / torch.sqrt(precisions) ## S * B * K * D
    labels = Z.argmax(-1)
    labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, D)
    x_mus = torch.gather(mus, 2, labels_flat)
    x_sigmas = torch.gather(sigmas, 2, labels_flat)
    log_p_x = Normal(x_mus, x_sigmas).log_prob(X).sum(-1).sum(-1) # S * B
    return log_p_x

def SNR(enc_init, enc_local, optimizer, x, K, D, SAMPLE_DIM, BATCH_DIM, num_samples, batch_size, num_samples_snr):
    grads = []
    for s in range(num_samples_snr):
        optimizer.zero_grad()
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
        loss = torch.mul(weights ** 2 - weights, log_weights).sum(0).mean()
        loss.backward()
        grad = []
        for param in list(enc_init.parameters()):
            grad.append(param.grad.view(-1, 1))
        for param in list(enc_local.parameters()):
            grad.append(param.grad.view(-1, 1))
        grad = torch.cat(grad, dim=0) ## P * 1
        grads.append(grad) ## P * SNR_SAMPLE_DIM
    grads = torch.cat(grads, dim=-1)
    E_g2 = (grads ** 2).mean(-1)
    E_g = grads.mean(-1)
    E2_g = E_g * E_g
    Var_g = E_g2 - E2_g
    SNR_g = E_g2 / Var_g

    return SNR_g.mean(), Var_g.mean()

def SNR_NRe(enc_init, enc_local, optimizer, x, K, D, SAMPLE_DIM, BATCH_DIM, num_samples, batch_size, num_samples_snr):
    grads = []
    for s in range(num_samples_snr):
        optimizer.zero_grad()
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
        loss = torch.mul(weights, log_weights).sum(0).mean()
        loss.backward()
        grad = []
        for param in list(enc_init.parameters()):
            grad.append(param.grad.view(-1, 1))
        for param in list(enc_local.parameters()):
            grad.append(param.grad.view(-1, 1))
        grad = torch.cat(grad, dim=0) ## P * 1
        grads.append(grad) ## P * SNR_SAMPLE_DIM
    grads = torch.cat(grads, dim=-1)
    E_g2 = (grads ** 2).mean(-1)
    E_g = grads.mean(-1)
    E2_g = E_g * E_g
    Var_g = E_g2 - E2_g
    SNR_g = E_g2 / Var_g

    return SNR_g.mean(), Var_g.mean()
