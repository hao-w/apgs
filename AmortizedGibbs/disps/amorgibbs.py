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

def rws_nested(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples = initial_trans(alpha_trans_0, K)
        log_weights_rws[l] = log_weights_rws[l] - log_q_hmm(alpha_trans_0, A_samples)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        log_weights_rws[l] = log_weights_rws[l] - log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K) + log_normalizer
        for m in range(mcmc_steps):
            A_prev = A_samples.clone()
            variational, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
            log_q_curr = log_q_hmm(variational, A_samples)
            log_weights_rws[l] = log_weights_rws[l] - log_q_curr
            Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
            log_weights_rws[l] = log_weights_rws[l] - log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K) + log_normalizer
        posterior = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(posterior, variational, K)
        log_weights_rws[l] = log_weights_rws[l] + log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
    weights_rws = torch.exp(log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    elbo = log_weights_rws.mean()
    return eubo, kl, ess, variational, elbo

def rws_decompose2(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros(num_particles_rws)
    log_p_joints = torch.zeros(num_particles_rws)
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples = initial_trans(alpha_trans_0, K)
        log_weights_rws[l] = log_weights_rws[l] - log_q_hmm(alpha_trans_0, A_samples)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_samples.clone()
            variational_curr, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
            log_weights_rws[l] = log_weights_rws[l] - log_q_hmm(variational_curr, A_samples)
            Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
            variational_curr, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
            log_q_prev = log_q_hmm(variational_curr, A_prev)
            log_weights_rws[l] = log_weights_rws[l] + log_q_prev
        log_weights_rws[l] = log_weights_rws[l] + log_normalizer + log_q_hmm(alpha_trans_0, A_samples)
        posterior = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(posterior, variational_curr, K)

    weights_rws = torch.exp(log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    elbo = log_weights_rws.mean()
    return eubo, kl, ess, variational_curr, elbo

def rws_sis(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_sis = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples = initial_trans(alpha_trans_0, K)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_sample, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_p_joint = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
        log_q_smc = log_joint_smc(Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K)
        log_weights_sis[l] = log_weights_sis[l] + log_p_joint - log_q_smc + log_normalizer - log_q_hmm(alpha_trans_0, A_samples)

        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_sample
            variational, A_sample = enc(Z_ret_pairwise, alpha_trans_0)
            log_q_mlp = log_q_hmm(variational, A_sample)
            Zs, log_weights, log_normalizer = smc_hmm(Pi, A_sample, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            log_q_smc = log_joint_smc(Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K)
            log_p_joint = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
            log_weights_sis[l] = log_weights_sis[l] + log_p_joint - log_q_smc + log_normalizer - log_q_mlp

        alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(posterior, variational_curr, K)

    weights_rws = torch.exp(log_weights_sis - logsumexp(log_weights_sis, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_sis).sum()
    elbo =  log_weights_sis.mean()
    return eubo, kl, ess, variational, elbo


def rws_decompose1(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros(num_particles_rws)
    log_p_joints = torch.zeros(num_particles_rws)
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples = initial_trans(alpha_trans_0, K)
        log_weights_rws[l] = log_weights_rws[l] - log_q_hmm(alpha_trans_0, A_samples)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        log_weights_rws[l] = log_weights_rws[l] + log_normalizer.clone()
        log_p_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()

        for m in range(mcmc_steps):
            A_prev = A_samples.clone()
            log_p_prev = log_p_curr.clone()
            variational_curr, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
            Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
            log_p_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
            log_weights_rws[l] = log_weights_rws[l] - log_q_hmm(variational_curr, A_samples) + log_q_hmm(variational_curr, A_prev) + log_p_curr.clone() - log_p_prev.clone()
        posterior = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(posterior, variational_curr, K)

    weights_rws = torch.exp(log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    elbo = log_weights_rws.mean()
    return eubo, kl, ess, variational_curr, elbo

def rws_decompose1_rao(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros((num_particles_rws, mcmc_steps+1))
    log_p_joints = torch.zeros(num_particles_rws)
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples = initial_trans(alpha_trans_0, K)
        log_weights_rws[l] = log_weights_rws[l] - log_q_hmm(alpha_trans_0, A_samples)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        log_weights_rws[l] = log_weights_rws[l] + log_normalizer.clone()
        log_p_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()

        for m in range(mcmc_steps):
            A_prev = A_samples.clone()
            log_p_prev = log_p_curr.clone()
            variational_curr, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
            Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
            log_p_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
            log_weights_rws[l] = log_weights_rws[l] - log_q_hmm(variational_curr, A_samples) + log_q_hmm(variational_curr, A_prev) + log_p_curr.clone() - log_p_prev.clone()
        posterior = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(posterior, variational_curr, K)

    weights_rws = torch.exp(log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    elbo = log_weights_rws.mean()
    return eubo, kl, ess, variational_curr, elbo
