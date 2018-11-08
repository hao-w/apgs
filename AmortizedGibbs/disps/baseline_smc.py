import matplotlib.pyplot as plt
import time
import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from smc import *

def save_params(KLs, EUBOs, ELBOs, PATH_ENC):
    with open(PATH_ENC + 'EUBOs.txt', 'w+') as feubo:
        for eubo in EUBOs:
            feubo.write("%s\n" % eubo)
    with open(PATH_ENC + 'KLs.txt', 'w+') as fkl:
        for kl in KLs:
            fkl.write("%s\n" % kl)
    with open(PATH_ENC + 'ELBOs.txt', 'w+') as ELBOs:
        for elbo in ELBOs:
            fess.write("%s\n" % elbo)
    feubo.close()
    ELBOs.close()
    fess.close()

def initial_trans_prior(K):
    alpha_trans_0 = torch.ones((K, K))
    return alpha_trans_0

def log_q_hmm(latents_dirs, A_samples):
    log_q = Dirichlet(latents_dirs).log_prob(A_samples)
    return log_q.sum()

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

def onesmc_sampling(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Z_true, Y, T, D, K, num_particles, num_particles_smc):
    log_weights_is = torch.zeros(num_particles)
    log_p_joints = torch.zeros(num_particles)
    kls = torch.zeros(num_particles)
    for l in range(num_particles):
        Y_pairwise = torch.cat((Y[:-1].unsqueeze(0), Y[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*D)
        latents_dirs, A_sample = enc(Y_pairwise)
        log_q_mlp = log_q_hmm(latents_dirs, A_sample)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_sample, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_q_smc = log_joint_smc(Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K)
        log_p_joints = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
        log_weights_is[l] = log_p_joints - log_q_smc - log_q_mlp + log_normalizer
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)

    log_weights_is_norm = log_weights_is - logsumexp(log_weights_is, dim=0)
    weights_is = torch.exp(log_weights_is_norm).detach()
    kl =  kls.mean()
    ess = (1. / (weights_is ** 2 ).sum()).item()
    eubo =  torch.mul(weights_is, log_weights_is).sum()
    elbo = log_weights_is.mean()
    return enc, eubo, kl, ess, latents_dirs, elbo

def twoshots_sampling(A_true, enc, alpha_trans_0, Pi, mu_ks, cov_ks, Z_true, Y, T, D, K, num_particles, num_particles_smc):
    log_weights_is = torch.zeros(num_particles)
    log_p_joints = torch.zeros(num_particles)
    kls = torch.zeros(num_particles)
    for l in range(num_particles):
        # Y_pairwise = torch.cat((Y[:-1].unsqueeze(0), Y[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*D)
        # latents_dirs, A_sample = enc(Y_pairwise)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_true, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_q_smc = log_joint_smc(Z_ret, Pi, A_true, mu_ks, cov_ks, Y, T, D, K)
        Z_pairwise = torch.cat((Z_ret[:-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        latents_dirs, A_sample = enc(Z_pairwise)
        log_q_mlp = log_q_hmm(latents_dirs, A_sample)

        log_p_joints = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
        log_weights_is[l] = log_p_joints - log_q_smc - log_q_mlp + log_normalizer
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)

    log_weights_is_norm = log_weights_is - logsumexp(log_weights_is, dim=0)
    weights_is = torch.exp(log_weights_is_norm).detach()
    kl =  torch.mul(weights_is, kls).sum()
    ess = (1. / (weights_is ** 2 ).sum()).item()
    eubo =  torch.mul(weights_is, log_weights_is).sum()
    return enc, eubo, kl, ess, latents_dirs