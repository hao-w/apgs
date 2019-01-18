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

def ag_sis_adaptive(enc, prior_true, prior_mcmc, As_true, Zs_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps, batch_size):
    """
    adaptive resampling at each mcmc step
    compute the EUBO only using samples at final step
    """
    log_increment_weights = torch.zeros((batch_size, mcmc_steps, num_particles_rws))
    Zs_candidates = torch.zeros((batch_size, num_particles_rws, T, K))
    As_candidates = torch.zeros((batch_size, num_particles_rws, K, K))
    conj_posts = conj_posterior(prior_true.repeat(batch_size, 1, 1), Zs_true, T, K, batch_size)
    Z_pairs_true = flatz(Zs_true, T, K, batch_size)
    # log_normalizers_candidates = torch.zeros((batch_size, rws))

    for m in range(mcmc_steps):
        if m == 0:
            for l in range(num_particles_rws):
                ## As B * K * K
                As = initial_trans(prior_mcmc, K, batch_size)
                As_candidates[:, l, :, :] = As
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                ## Z B * T * K
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[:, l, :, :] = Z
                log_increment_weights[:, m, l] = log_normalizers
        else:
            for l in range(num_particles_rws):
                ## z_pairs (B * T-1)-by-(2K)
                Z_pairs = flatz(Zs_candidates[:, l, :, :], T, K, batch_size)
                variational, As = enc(Z_pairs, prior_mcmc, batch_size)
                As_candidates[:, l, :, :] = As
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[:, l, :, :] = Z
                # log_normalizers_candidates[:, l] = log_normalizers
                log_ps = log_joints(prior_mcmc, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
                log_ps_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
                log_increment_weights[:, m, l] =  log_ps.detach() + log_normalizers - log_qs(variational, As) - log_ps_smc.detach()

        if m != (mcmc_steps-1):
           Zs_candidates, As_candidates = resample_rws(As_candidates, Zs_candidates, log_increment_weights[:, m].detach(), num_particles_rws, T, K)

    variational_true, As_notusing = enc(Z_pairs_true, prior_mcmc, batch_size)
    kls = kl_dirichlets(variational_true, conj_posts).sum(-1)

    ave_log_weights = (logsumexp(log_increment_weights[:, :-1, :], dim=-1) - torch.log(torch.FloatTensor([num_particles_rws]))).sum(-1)
    log_overall_weights = ave_log_weights.unsqueeze(-1).repeat(1, num_particles_rws) + log_increment_weights[:, -1, :]
    weights_rws = torch.exp(log_overall_weights - logsumexp(log_overall_weights, dim=-1).unsqueeze(-1)).detach()

    ess = (1. / (weights_rws ** 2 ).sum(1)).mean()
    eubos = torch.mul(weights_rws, log_overall_weights).sum(1)
    elbos =  log_overall_weights.mean(-1)
    return eubos.mean(), elbos.mean(), ess, kls.mean()

def resample_rws(As, Zs, log_weights, num_particles, T, K):
    """
    As is B-rws-K-K Tensor
    Zs is B-rws-T-K Tensor
    log_normalizers is B-rws Tensor
    log_weights is B-rws
    """
    weights = torch.exp(log_weights - logsumexp(log_weights, dim=-1).unsqueeze(-1))
    ## ancesters is B-rws tensor
    ancesters = Categorical(weights).sample((num_particles, )).transpose(0,1)
    ## expand it to B-rws-T-K
    ancesters_z = ancesters.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, K)
    ## expand it to B-rws-K-K
    ancesters_A = ancesters.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, K)
    ##
    # ancesters_normalizer = ancesters
    ## resample using gather, Zs_resampled is B-rws-T-K
    Zs_resampled = torch.gather(Zs, 1, ancesters_z)
    As_resampled = torch.gather(As, 1, ancesters_A)
    # log_normalizers_resampled = torch.gather(log_normalizers, 1, ancesters_normalizer)
    return Zs_resampled, As_resampled

def ag_sis_stepwise(enc, prior_true, prior_mcmc, As_true, Zs_true, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps, batch_size):
    log_uptonow_weights = torch.zeros((batch_size, mcmc_steps, num_particles_rws))
    log_increment_weights = torch.zeros((batch_size, mcmc_steps, num_particles_rws))
    Zs_candidates = torch.zeros((num_particles_rws, batch_size, T, K))
    log_normalizers_candidates = torch.zeros((num_particles_rws, batch_size))

    conj_posts = conj_posterior(prior_true.repeat(batch_size, 1, 1), Zs_true, T, K, batch_size)
    Z_pairs_true = flatz(Zs_true, T, K, batch_size)

    for m in range(mcmc_steps):
        if m == 0:
            for l in range(num_particles_rws):
                ## As B * K * K
                As = initial_trans(prior_mcmc, K, batch_size)
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                log_normalizers_candidates[l] = log_normalizers
                ## Z B * T * K
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[l] = Z
                ## the first incremental weight is just log normalizer since As is sampled from prior
                log_increment_weights[:, m, l] = log_normalizers
                log_uptonow_weights[:, m, l] = log_normalizers
        else:
            for l in range(num_particles_rws):
                ## z_pairs (B * T-1)-by-(2K)
                Z_pairs = flatz(Zs_candidates[l], T, K, batch_size)
                variational, As = enc(Z_pairs, prior_mcmc, batch_size)
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_candidates[l] = Z
                log_ps = log_joints(prior_mcmc, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
                log_ps_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
                log_increment_weights[:, m, l] =  log_ps.detach() + log_normalizers - log_qs(variational, As) - log_ps_smc.detach()
                log_uptonow_weights[:, m, l] = log_uptonow_weights[:, m-1, l] + log_increment_weights[:, m, l]

    variational_true, As_notusing = enc(Z_pairs_true, prior_mcmc, batch_size)
    kls = kl_dirichlets(variational_true, conj_posts).sum(-1)

    log_overall_weights = log_uptonow_weights[:, -1, :]
    weights_rws = torch.exp(log_overall_weights - logsumexp(log_overall_weights, dim=1).unsqueeze(1)).detach()
    uptonow_weights = torch.exp(log_uptonow_weights - logsumexp(log_uptonow_weights, dim=-1).unsqueeze(-1)).detach()
    increment_weights = torch.exp(log_increment_weights - logsumexp(log_increment_weights, dim=-1).unsqueeze(-1)).detach()

    ess = (1. / (weights_rws ** 2 ).sum(1)).mean()
    eubos = torch.mul(increment_weights, log_increment_weights).sum(-1).mean(-1)
    elbos =  log_overall_weights.mean(-1)
    loss = torch.mul(uptonow_weights, log_increment_weights).sum(1).mean(1)
    return  loss.mean(), eubos.mean(), elbos.mean(), ess, kls.mean()




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
