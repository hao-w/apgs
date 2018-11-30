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

def ag_decompose2(enc, prior_true, As_true, Zs_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps, batch_size):
    log_overall_weights = torch.zeros((batch_size, num_particles_rws))

    for l in range(num_particles_rws):
        ## As B * K * K
        As = initial_trans(prior_true, K, batch_size)
        Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
        ## Z B * T * K
        Z = smc_resamplings(Zs, log_weights, batch_size)
        ## z_pairs (B * T-1)-by-(2K)
        Z_pairs = flatz(Z, T, K, batch_size)
        ## the first incremental weight is just log normalizer since As is sampled from prior

        log_overall_weights[:, l] =  - log_qs(prior_true.repeat(batch_size, 1, 1), As)
        for m in range(mcmc_steps):
            As_prev = As.clone()
            variational, As = enc(Z_pairs, prior_true, batch_size)
            ## log_q(\phi_t | z_t-1)
            log_overall_weights[:, l] = log_overall_weights[:, l] - log_qs(variational, As)

            Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
            Z = smc_resamplings(Zs, log_weights, batch_size)
            Z_pairs = flatz(Z, T, K, batch_size)
            # log_qs_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
            variational, As = enc(Z_pairs, prior_true, batch_size)
            log_overall_weights[:, l] = log_overall_weights[:, l] + log_qs(variational, As_prev)

        log_overall_weights[:, l] = log_overall_weights[:, l] + log_normalizers + log_qs(prior_true.repeat(batch_size, 1, 1), As)

    conj_posts = conj_posterior(prior_true.repeat(batch_size, 1, 1), Zs_true, T, K, batch_size)
    Z_pairs_true = flatz(Zs_true, T, K, batch_size)
    variational_true, As_notusing = enc(Z_pairs_true, prior_true, batch_size)
    kls = log_qs(conj_posts, As_true) - log_qs(variational_true, As_true)

    weights_rws = torch.exp(log_overall_weights - logsumexp(log_overall_weights, dim=1).unsqueeze(1)).detach()
    ess = (1. / (weights_rws ** 2 ).sum(1)).mean()
    eubos = torch.mul(weights_rws, log_overall_weights).sum(1)
    elbos =  log_overall_weights.mean(-1)
    loss = torch.mul(torch.exp(log_overall_weights).detach(), log_overall_weights).sum(1)
    return  loss.mean(), eubos.mean(), elbos.mean(), ess, kls.mean()

def ag_adapt(enc, prior_true, As_true, Zs_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps, batch_size):
    log_overall_weights = torch.zeros((batch_size, num_particles_rws))
    Zs_candidates = torch.zeros((num_particles_rws, batch_size, T, K))
    As_candidates = torch.zeros((num_particles_rws, batch_size, K, K))
    for m in range(mcmc_steps):
        if m == 0:
            for l in range(num_particles_rws):
                ## As B * K * K
                As = initial_trans(prior_true, K, batch_size)
                As_candidates[l] = As
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                ## Z B * T * K
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[l] = Z
                ## z_pairs (B * T-1)-by-(2K)
                Z_pairs = flatz(Z, T, K, batch_size)
                ## the first incremental weight is just log normalizer since As is sampled from prior
                log_overall_weights[:, l] =  - log_qs(prior_true.repeat(batch_size, 1, 1), As)
        else:
            for l in range(num_particles_rws):

                variational, As = enc(Z_pairs, prior_true, batch_size)
                ## log_q(\phi_t | z_t-1)
                log_overall_weights[:, l] = log_overall_weights[:, l] - log_qs(variational, As)
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[l] = Z
                Z_pairs = flatz(Z, T, K, batch_size)
                # log_qs_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
                variational, As_notusing = enc(Z_pairs, prior_true, batch_size)
                log_overall_weights[:, l] = log_overall_weights[:, l] + log_qs(variational, As_candidates[l].clone())
                As_candidates[l] = As
                if m == mcmc_steps-1:
                    log_overall_weights[:, l] = log_overall_weights[:, l] + log_normalizers + log_qs(prior_true.repeat(batch_size, 1, 1), As)

    conj_posts = conj_posterior(prior_true.repeat(batch_size, 1, 1), Zs_true, T, K, batch_size)
    Z_pairs_true = flatz(Zs_true, T, K, batch_size)
    variational_true, As_notusing = enc(Z_pairs_true, prior_true, batch_size)
    kls = log_qs(conj_posts, As_true) - log_qs(variational_true, As_true)

    weights_rws = torch.exp(log_overall_weights - logsumexp(log_overall_weights, dim=1).unsqueeze(1)).detach()
    ess = (1. / (weights_rws ** 2 ).sum(1)).mean()
    eubos = torch.mul(weights_rws, log_overall_weights).sum(1)
    elbos =  log_overall_weights.mean(-1)
    loss = torch.mul(torch.exp(log_overall_weights).detach(), log_overall_weights).sum(1)
    return  loss.mean(), eubos.mean(), elbos.mean(), ess, kls.mean()


def ag_sis(enc, alpha_trans_0, prior_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps, batch_size):
    log_overall_weights = torch.zeros((batch_size, num_particles_rws))
    for l in range(num_particles_rws):
        ## As B * K * K
        As = initial_trans(prior_true, K, batch_size)
        Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
        ## Z B * T * K
        Z = smc_resamplings(Zs, log_weights, batch_size)
        ## z_pairs (B * T-1)-by-(2K)
        Z_pairs = flatz(Z, T, K, batch_size)
        ## the first incremental weight is just log normalizer since As is sampled from prior
        log_overall_weights[:, l] =  log_normalizers
        for m in range(mcmc_steps):
            As_prev = As.clone()
            variational, As = enc(Z_pairs, prior_true, batch_size)
            log_qs_phi = log_qs(variational, As)
            Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
            Z = smc_resamplings(Zs, log_weights, batch_size)
            Z_pairs = flatz(Z, T, K, batch_size)
            log_qs_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
            log_ps = log_joints(prior_true, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size).detach()

            log_overall_weights[:, l] = log_ps - log_qs_smc + log_normalizers - log_qs_phi + log_overall_weights[:, l]

    weights_rws = torch.exp(log_overall_weights - logsumexp(log_overall_weights, dim=1).unsqueeze(1)).detach()
    ess = (1. / (weights_rws ** 2 ).sum(1)).mean()
    loss =  torch.mul(weights_rws, log_overall_weights).sum(1)
    eubos = torch.mul(weights_rws, log_overall_weights).sum(1)
    elbos =  log_overall_weights.mean(-1)
    return  eubos.mean(), elbos.mean(), variational, ess

def encode_obs(enc, prior_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, batch_size):
    log_overall_weights = torch.zeros((batch_size, num_particles_rws))
    Y_pairs = flatz(Ys, T, D, batch_size)
    kls = torch.zeros(batch_size).float()
    for l in range(num_particles_rws):
        variational, As = enc(Y_pairs, prior_true, batch_size)
        log_qs_phi = log_qs(variational, As)
        Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
        Z = smc_resamplings(Zs, log_weights, batch_size)
        log_qs_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
        log_ps = log_joints(prior_true, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size).detach()
        log_overall_weights[:, l] = log_ps - log_qs_smc - log_qs_phi + log_normalizers
    for b in range(batch_size):
        posterior = prior_true + pairwise(Z[b], T).sum(0)
        kls[b] = kl_dirichlets(posterior, variational[b], K)

    weights_rws = torch.exp(log_overall_weights.detach() - logsumexp(log_overall_weights, dim=1).unsqueeze(1)).detach()
    eubos =  torch.mul(weights_rws, log_overall_weights).sum(1)
    elbos =  log_overall_weights.mean(-1)
    divergence = eubos - elbos
    ess = (1. / (weights_rws ** 2).sum(1)).mean()

    return divergence.mean(), eubos.mean(), elbos.mean(), variational, ess, kls.mean()
