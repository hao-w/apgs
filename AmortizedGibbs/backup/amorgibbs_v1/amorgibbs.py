import matplotlib.pyplot as plt
import time
import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from smc import *

def initial_trans(alpha_trans_0, K, num_particles_rws):
    # A = torch.zeros((K, K)).float()
    # for k in range(K):
    #     A[k] = Dirichlet(alpha_trans_0[k]).sample()
    A = torch.ones((K, K)).float() / 10
    for k in range(K):
        A[k, k] = 1 / 7
    return A.repeat(num_particles_rws, 1, 1)

def initial_trans_prior(K):
    alpha_trans_0 = torch.ones((K, K))
    for k in range(K):
        alpha_trans_0[k,k] = K
    return alpha_trans_0

def pairwise(Zs, T):
    return torch.bmm(Zs[:T-1].unsqueeze(-1), Zs[1:].unsqueeze(1))

def log_joint(alpha_trans_0, Zs, Pi, A, mu_ks, cov_ks, Y, T, D, K):
    log_joint_prob = torch.zeros(1).float()
    labels = Zs.nonzero()[:, 1]
    log_joint_prob += (MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y)).sum() # likelihood of obs
    log_joint_prob += cat(Pi).log_prob(Zs[0]) # z_1 | pi
    log_joint_prob += (cat(A[labels[:-1]]).log_prob(Zs[1:])).sum()
    log_joint_prob += (Dirichlet(alpha_trans_0).log_prob(A)).sum() ## prior of A
    return log_joint_prob

def log_q_hmm(latents_dirs, A_samples):
    log_q = Dirichlet(latents_dirs).log_prob(A_samples)
    return log_q.sum()

def kl_dirichlets(alpha_trans_0, latents_dirs, Zs, T, K):
    alpha_trans_hat = alpha_trans_0 + pairwise(Zs, T).sum(0)
    kl = 0.0
    for k in range(K):
        kl += kl_dirichlet(alpha_trans_hat[k], latents_dirs[k])
    return kl

def kl_dirichlet(alpha1, alpha2):
    A = torch.lgamma(alpha1.sum()) - torch.lgamma(alpha2.sum())
    B = (torch.lgamma(alpha1) - torch.lgamma(alpha2)).sum()
    C = (torch.mul(alpha1 - alpha2, torch.digamma(alpha1) - torch.digamma(alpha1.sum()))).sum()
    kl = A - B + C
    return kl

def rws(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros(num_particles_rws)
    log_p_joints = torch.zeros(num_particles_rws)
    log_normalizers = torch.zeros(num_particles_rws)
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        A_samples = initial_trans(alpha_trans_0, K, 1)[0]
        # A_samples = A_init
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_weight_rws = log_normalizer
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_samples
            latents_dirs, A_samples = enc(Z_ret_pairwise, alpha_trans_0.sum().item(), T)
            log_p_joint_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
            log_p_joint_prev = log_joint(alpha_trans_0, Z_ret, Pi, A_prev, mu_ks, cov_ks, Y, T, D, K).detach()
            log_q_curr = log_q_hmm(latents_dirs, A_samples).detach()
            log_q_prev = log_q_hmm(latents_dirs, A_prev).detach()
            log_weight_rws += log_p_joint_curr - log_p_joint_prev - log_q_curr + log_q_prev
            #
            Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)

        log_weights_rws[l] = log_weight_rws.detach()
        log_p_smcs[l] = log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        log_normalizers[l] = log_normalizer.detach()
        alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
        log_p_joints[l] = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        log_qs[l] = log_q_hmm(latents_dirs, A_samples)

    log_weights_rws = log_weights_rws - logsumexp(log_weights_rws)
    weights_rws = torch.exp(log_weights_rws)
    kl =  torch.mul(weights_rws, kls).sum()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    loss_infer = - torch.mul(weights_rws, log_qs).sum()
    eubo =  torch.mul(weights_rws, log_p_joints - log_qs + log_normalizers - log_p_smcs).sum()
    return enc, loss_infer, eubo, kl, ess, latents_dirs, Z_ret
