import matplotlib.pyplot as plt
import time
import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from smc import *
from util import *
from data import *

def flatz(Z, T, K, batch_size):
    return torch.cat((Z[:, :T-1, :].unsqueeze(2), Z[:, 1:, :].unsqueeze(2)), 2).view(batch_size * (T-1), 2*K)

def baseline_vimco(enc, prior_true, prior_mcmc, Zs_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_samples, num_particles_smc, batch_size):
    """
    baseline vimco
    """
    log_increment_weights = torch.zeros((batch_size, num_samples))
    log_q_mcmc = torch.zeros((batch_size, num_samples))
    conj_posts = conj_posterior(prior_true.repeat(batch_size, 1, 1), Zs_true, T, K, batch_size)
    Y_pairs = flatz(Ys, T, D, batch_size)
    for l in range(num_samples):

        ## As B * K * K
        variational, As = enc(Y_pairs, prior_mcmc, batch_size)
        Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
        ## Z B * T * K
        Z = smc_resamplings(Zs, log_weights, batch_size)
        log_ps_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
        log_p_joints = log_joints(prior_mcmc, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
        ## the first incremental weight is just log normalizer since As is sampled from prior
        log_increment_weights[:, l] = log_p_joints - log_qs(variational, As) - log_ps_smc + log_normalizers
        log_q_mcmc[:, l] = log_qs(variational, As) + log_ps_smc - log_normalizers

    variational_true, As_notusing = enc(Y_pairs, prior_mcmc, batch_size)
    kls = kl_dirichlets(variational_true, conj_posts).sum(-1)

    ess = (1. / (log_increment_weights ** 2 ).sum(1)).mean()
    elbos = log_increment_weights.mean(-1)

    log_sum_weights = logsumexp(log_increment_weights, -1)
    log_K = torch.log(torch.FloatTensor([num_samples]))
    term1_A = (log_sum_weights - log_K).unsqueeze(-1).repeat(1, num_samples).detach()
    term1_B1 = (log_increment_weights.sum(-1).unsqueeze(-1).repeat(1, num_samples) - log_increment_weights) / (num_samples - 1.0)
    expand = log_increment_weights.unsqueeze(1).repeat(1, num_samples, 1)
    extra_col = - torch.diagonal(expand, offset=0, dim1=1, dim2=2)
    term1_B2 = logsumexp(torch.cat((expand, extra_col.unsqueeze(-1)), -1), dim=-1)

    term1_B = (logsumexp(torch.cat((term1_B1.unsqueeze(-1), term1_B2.unsqueeze(-1)), -1), dim=-1) - log_K).detach()
    term1 = torch.mul(term1_A - term1_B, log_q_mcmc).sum(-1)
    term2 = log_sum_weights - log_K
    gradient = term1 + term2
    return gradient.mean(), elbos.mean(), ess, kls.mean()

def ag_mcmc_vimco(enc, prior_mcmc, Zs_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_samples, num_particles_smc, mcmc_steps, batch_size):
    """
    mcmc sampling scheme 2
    vimco gradient estimator
    """
    log_increment_weights = torch.zeros((batch_size, num_samples))
    log_q_mcmc = torch.zeros((batch_size, num_samples))
    Zs_candidates = torch.zeros((num_samples, mcmc_steps, batch_size, T, K))
    As_candidates = torch.zeros((num_samples, mcmc_steps, batch_size, K, K))
    conj_posts = conj_posterior(prior_mcmc.repeat(batch_size, 1, 1), Zs_true, T, K, batch_size)
    Z_pairs_true = flatz(Zs_true, T, K, batch_size)

    for m in range(mcmc_steps):
        if m == 0:
            for l in range(num_samples):
                ## As B * K * K
                As = initial_trans(prior_mcmc, K, batch_size)
                As_candidates[l, 0] = As
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                ## Z B * T * K
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[l, 0] = Z
                log_ps_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
                ## the first incremental weight is just log normalizer since As is sampled from prior
                log_increment_weights[:, l] = - log_qs(prior_mcmc.unsqueeze(0).repeat(batch_size, 1, 1), As)
                log_q_mcmc[:, l] = log_qs(prior_mcmc.unsqueeze(0).repeat(batch_size, 1, 1), As) + log_ps_smc - log_normalizers
                # exclusive_kls[:, l] = exclusive_kls[:, l] + log_qs(conj_posts, As) - log_qs(prior_mcmc, As)
        else:
            for l in range(num_samples):
                ## z_pairs (B * T-1)-by-(2K)
                Z = Zs_candidates[l, m-1]
                Z_pair = flatz(Z, T, K, batch_size)
                variational, As = enc(Z_pair, prior_mcmc, batch_size)
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[l, m] = Z
                Z_pair = flatz(Z, T, K, batch_size)
                variational_new, As_notusing = enc(Z_pair, prior_mcmc, batch_size)
                As_prev = As_candidates[l, m-1]
                log_increment_weights[:, l] =  log_increment_weights[:, l] - log_qs(variational, As) + log_qs(variational_new, As_prev)
                As_candidates[l, m] = As
                log_q_mcmc[:, l] = log_q_mcmc[:, l] + log_qs(variational, As) + log_ps_smc - log_normalizers
                if m == (mcmc_steps-1):
                    log_increment_weights[:, l] =  log_increment_weights[:, l] + log_qs(prior_mcmc.unsqueeze(0).repeat(batch_size, 1, 1), As) + log_normalizers
    variational_true, As_notusing = enc(Z_pairs_true, prior_mcmc, batch_size)
    kls = kl_dirichlets(variational_true, conj_posts).sum(-1)

    elbos = log_increment_weights.mean(-1)

    log_sum_weights = logsumexp(log_increment_weights, -1)
    log_K = torch.log(torch.FloatTensor([num_samples]))
    term1_A = (log_sum_weights - log_K).unsqueeze(-1).repeat(1, num_samples).detach()
    term1_B1 = (log_increment_weights.sum(-1).unsqueeze(-1).repeat(1, num_samples) - log_increment_weights) / (num_samples - 1.0)
    expand = log_increment_weights.unsqueeze(1).repeat(1, num_samples, 1)
    extra_col = - torch.diagonal(expand, offset=0, dim1=1, dim2=2)
    term1_B2 = logsumexp(torch.cat((expand, extra_col.unsqueeze(-1)), -1), dim=-1)

    term1_B = (logsumexp(torch.cat((term1_B1.unsqueeze(-1), term1_B2.unsqueeze(-1)), -1), dim=-1) - log_K).detach()
    term1 = torch.mul(term1_A - term1_B, log_q_mcmc).sum(-1)
    term2 = log_sum_weights - log_K
    gradient = term1 + term2
    return gradient.mean(), elbos.mean(), kls.mean()
