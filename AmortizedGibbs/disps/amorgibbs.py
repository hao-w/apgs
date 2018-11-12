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
        A_samples, log_prior = initial_trans(alpha_trans_0, K)
        # A_samples = A_init
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_weights_rws[l] = log_normalizer.detach()
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_samples
            latents_dirs, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
            log_q_curr = log_q_hmm(latents_dirs, A_samples)
            log_weights_rws[l] = log_weights_rws[l] - log_q_curr
            Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
        log_weights_rws[l] = log_weights_rws[l] + log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        
    log_weights_rws_norm = (log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    weights_rws = torch.exp(log_weights_rws_norm)
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    elbo = log_weights_rws.mean(0).sum()
    
    return enc, eubo, kl, ess, latents_dirs, Z_ret, elbo


def rws_rao(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Zs_true, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros((num_particles_rws, mcmc_steps+1))
    log_p_joints = torch.zeros(num_particles_rws)
    log_grads = torch.zeros((num_particles_rws, mcmc_steps))
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples, log_prior = initial_trans(alpha_trans_0, K)
        # A_samples = A_init
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
       
        log_weights_rws[l, 0] = log_normalizer.detach()
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_samples
            latents_dirs, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
            log_p_joint_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
            log_p_joint_prev = log_joint(alpha_trans_0, Z_ret, Pi, A_prev, mu_ks, cov_ks, Y, T, D, K).detach()
            log_q_curr = log_q_hmm(latents_dirs, A_samples)
            log_q_prev = log_q_hmm(latents_dirs, A_prev)
            log_weights_rws[l, m+1] = log_p_joint_curr - log_p_joint_prev - log_q_curr + log_q_prev
            #log_grads[l, m] = - log_q_curr + log_q_prev
            Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
        
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)

        alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
  
    log_weights_rws_norm = (log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    weights_rws = torch.exp(log_weights_rws_norm)
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    elbo = log_weights_rws.mean(0).sum()
    return enc, eubo, kl, ess, latents_dirs, Z_ret, elbo

def rws_rao2(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros((num_particles_rws, mcmc_steps+1))
    log_p_joints = torch.zeros(num_particles_rws)
    log_grads = torch.zeros((num_particles_rws, mcmc_steps))
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples, log_prior = initial_trans(alpha_trans_0, K)
        # A_samples = A_init
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_weights_rws[l, 0] = log_normalizer
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_samples
            latents_dirs, A_samples = enc(Z_ret_pairwise)
            log_q_curr = log_q_hmm(latents_dirs, A_samples)
            log_q_prev = log_q_hmm(latents_dirs, A_prev)
            log_weights_rws[l, m+1] = - log_q_curr + log_q_prev
            #log_grads[l, m] = - log_q_curr + log_q_prev
            Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)

        log_p_joints[l] = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        log_p_smcs[l] = log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K) - log_normalizer
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
    log_weights_rws[:, -1] =log_weights_rws[:, -1] + log_p_joints  - log_p_smcs
    log_weights_rws_norm = (log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    weights_rws = torch.exp(log_weights_rws_norm)
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    return enc, eubo, kl, ess, latents_dirs, Z_ret


def rws_rao_notnorm(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros((num_particles_rws, mcmc_steps+1))
    log_p_joints = torch.zeros(num_particles_rws)
    log_grads = torch.zeros((num_particles_rws, mcmc_steps))
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples, log_prior = initial_trans(alpha_trans_0, K)
        # A_samples = A_init
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_weights_rws[l, 0] = log_normalizer
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_samples
            latents_dirs, A_samples = enc(Z_ret_pairwise)
            log_p_joint_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
            log_p_joint_prev = log_joint(alpha_trans_0, Z_ret, Pi, A_prev, mu_ks, cov_ks, Y, T, D, K).detach()
            log_q_curr = log_q_hmm(latents_dirs, A_samples)
            log_q_prev = log_q_hmm(latents_dirs, A_prev)
            log_weights_rws[l, m+1] = log_p_joint_curr - log_p_joint_prev - log_q_curr + log_q_prev
            #log_grads[l, m] = - log_q_curr + log_q_prev
            Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
    kl = kls.mean()
    stepwise_weights = torch.exp(log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    ess = (1. / (stepwise_weights ** 2 ).sum(0))
    log_weights_rws_norm = (log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    weights_rws_eubo = torch.exp(log_weights_rws_norm)

    eubo =  torch.mul(weights_rws_eubo[:,1:], log_weights_rws[:,1:]).sum() + torch.mul(weights_rws_eubo[:,0], log_weights_rws[:,0]).sum().detach()
    
    gradient = torch.mul(torch.exp(log_weights_rws[:,1:]).detach(), log_weights_rws[:,1:]).sum()
    print(ess / num_particles_rws)
    return enc, eubo, kl, ess, latents_dirs, Z_ret, gradient



def rws_sis(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_sis = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_sample, log_prior = initial_trans(alpha_trans_0, K)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_sample, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_p_joint = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
        log_q_smc = log_joint_smc(Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K)
        log_weights_sis[l] = log_weights_sis[l] + log_p_joint - log_q_smc + log_normalizer - log_prior
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_sample
            latents_dirs, A_sample = enc(Z_ret_pairwise)
            log_q_mlp = log_q_hmm(latents_dirs, A_sample)
            Zs, log_weights, log_normalizer = smc_hmm(Pi, A_sample, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            log_q_smc = log_joint_smc(Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K)
            log_p_joint = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
            log_weights_sis[l] = log_weights_sis[l] + log_p_joint - log_q_smc + log_normalizer - log_q_mlp
            
        alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)

    weights_rws = torch.exp(log_weights_sis - logsumexp(log_weights_sis, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_sis).sum()
    elbo =  log_weights_sis.mean()
    return enc, eubo, kl, ess, latents_dirs, Z_ret, elbo
    