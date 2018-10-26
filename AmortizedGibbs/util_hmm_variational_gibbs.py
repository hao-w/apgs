import math
import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from scipy.stats import invwishart
import numpy as np
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch
from probtorch.util import log_sum_exp
import time


def initial_trans(alpha_trans_0, K):
    # A = torch.zeros((K, K)).float()
    # for k in range(K):
    #     A[k] = Dirichlet(alpha_trans_0[k]).sample()
    A = torch.ones((K, K)).float() / 10
    for k in range(K):
        A[k, k] = 1 / 7
    return A

def initial_trans_prior(K):
    alpha_trans_0 = torch.ones((K, K))
    for k in range(K):
        alpha_trans_0[k,k] = K
    return alpha_trans_0

def make_cov(L):
    L_star = L.tril(-1) + L.diag().pow(2.0).diag()
    C = torch.matmul(L, L.t())
    E = torch.diag(C).pow(-0.5).diag()
    return torch.matmul(E, torch.matmul(C, E))

def quad(a, B):
    return torch.mm(torch.mm(a.transpose(0, 1), B), a)

def csmc_hmm(Z_ret, Pi, A, mu_ks, cov_ks, Y, T, D, K, num_particles=1):
    Zs = torch.zeros((num_particles, T, K))
    log_weights = torch.zeros((num_particles, T))
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    log_normalizer = torch.zeros(1).float()

    for t in range(T):
        if t == 0:
            Zs[-1, t] = Z_ret[t]
            label = Z_ret[t].nonzero().item()
            likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
            log_weights[-1, t] = likelihood
            for n in range(num_particles-1):
                sample_z = cat(Pi).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood

        else:
            reweight = torch.exp(log_weights[:, t-1] - log_sum_exp(log_weights[:, t-1]))
            ancesters = Categorical(reweight).sample((num_particles-1,))
            Zs[:-1] = Zs[ancesters]
            Zs[-1, t] = Z_ret[t]
            label = Z_ret[t].nonzero().item()
            likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
            log_weights[-1, t] = likelihood
            for n in range(num_particles-1):
                label = torch.mm(Zs[n, t-1].unsqueeze(0), decode_onehot).int().item()
                sample_z = cat(A[label]).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood
        log_normalizer += log_sum_exp(log_weights[:, t]) - torch.log(torch.FloatTensor([num_particles]))
    return Zs, log_weights, log_normalizer

def log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K):
    log_joint = torch.zeros(1).float()
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    for t in range(T):
        label = Z_ret[t].nonzero().item()
        likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
        log_joint += likelihood
        if t == 0:
            log_joint += cat(Pi).log_prob(Z_ret[t])
        else:
            label_prev = Z_ret[t-1].nonzero().item()
            log_joint += cat(A_samples[label_prev]).log_prob(Z_ret[t])
    return log_joint

def smc_hmm(Pi, A, mu_ks, cov_ks, Y, T, D, K, num_particles=1):
    Zs = torch.zeros((num_particles, T, K))
    log_weights = torch.zeros((num_particles, T))
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    log_normalizer = torch.zeros(1).float()
    for t in range(T):
        if t == 0:
            for n in range(num_particles):
                sample_z = cat(Pi).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood
        else:
            ## resampling
            reweight = torch.exp(log_weights[:, t-1] - log_sum_exp(log_weights[:, t-1]))
            ancesters = Categorical(reweight).sample((num_particles,))
            Zs = Zs[ancesters]
            for n in range(num_particles):
                label = torch.mm(Zs[n, t-1].unsqueeze(0), decode_onehot).int().item()
                sample_z = cat(A[label]).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood
        log_normalizer += log_sum_exp(log_weights[:, t]) - torch.log(torch.FloatTensor([num_particles]))
        #
        # for n in range(num_particles):
        #     path = torch.mm(Zs[n, :t+1], decode_onehot).data.numpy()
        #     plt.plot(path, 'r-o')
        # plt.show()

    return Zs, log_weights, log_normalizer

def resampling_smc(Zs, log_weights):
    reweight = log_weights[:, -1] - log_sum_exp(log_weights[:, -1])
    ancester = Categorical(reweight).sample().item()
    return Zs[ancester]


## compute the I[z_t=i]I[z_t-1=j] by vectorization
def pairwise(Zs, T):
    return torch.bmm(Zs[:T-1].unsqueeze(-1), Zs[1:].unsqueeze(1))

def log_joint(alpha_trans_0, Zs, Pi, A, mu_ks, cov_ks, Y, T, D, K):
    log_joint_prob = 0.0
    decode_onehot = torch.arange(K).repeat(T, 1).float()
    labels = torch.bmm(decode_onehot.unsqueeze(1), Zs.unsqueeze(2)).squeeze(-1).squeeze(-1).int()
    for t in range(T):
        log_joint_prob += MultivariateNormal(mu_ks[labels[t].item()], cov_ks[labels[t].item()]).log_prob(Y[t]) # likelihood of obs
        if t == 0:
            log_joint_prob += cat(Pi).log_prob(Zs[t]) # z_1 | pi
        else:
            log_joint_prob += cat(A[labels[t-1].item()]).log_prob(Zs[t]) # z_t | z_t-1 = j*, A
    for k in range(K):
        log_joint_prob += Dirichlet(alpha_trans_0[k]).log_prob(A[k]) ## prior of A
    return log_joint_prob

# def inclusive_kl(A_samples, latents_dirs, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_encoder):
#     log_p_joint = torch.zeros(num_particles_encoder)
#     for n in range(num_particles_encoder):
#         A = A_samples[:, n, :]
#         log_p_joint[n] = log_joint(alpha_trans_0, Zs, Pi, A, mu_ks, cov_ks, Y, T, D, K).item()
#
#     alpha_trans_hat = alpha_trans_0 + pairwise(Zs, T).sum(0)
#     kl = kl_dirichlets(alpha_trans_0, latents_dirs, Zs, T, K)
#
#     log_q = log_q_hmm(latents_dirs, A_samples, K, num_particles_encoder)
#     log_weights = log_p_joint - log_q - log_sum_exp(log_p_joint - log_q)
#     log_weights = log_weights.detach()
#     weights = torch.exp(log_weights)
#     loss_inference = - torch.mul(weights, log_q).sum()
#     ess = (1. / (weights ** 2 ).sum()).item()
#     kl_est = torch.mul(weights, log_p_cond - log_q).sum().detach().item()
#     return loss_inference, kl, kl_est, ess

def log_q_hmm(latents_dirs, A_samples, K, num_particles):
    log_q = torch.zeros((K, num_particles))
    for k in range(K):
        log_q[k] = Dirichlet(latents_dirs[k]).log_prob(A_samples[k])
    if num_particles == 1:
        return log_q.sum()
    else:
        return log_q.sum(0)

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

def rws(enc, A_init, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros(num_particles_rws)
    log_p_joints = torch.zeros(num_particles_rws)
    log_normalizers = torch.zeros(num_particles_rws)
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
#         A_samples = initial_trans(alpha_trans_0, K)
        A_samples = A_init
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_weight_rws = log_normalizer
        Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        for m in range(mcmc_steps):
            A_prev = A_samples
            latents_dirs, A_samples = enc(Z_ret_pairwise, alpha_trans_0.sum().item(), T)
            log_p_joint_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
            log_p_joint_prev = log_joint(alpha_trans_0, Z_ret, Pi, A_prev, mu_ks, cov_ks, Y, T, D, K).detach()
            log_q_curr = log_q_hmm(latents_dirs, A_samples, K, 1).detach()
            log_q_prev = log_q_hmm(latents_dirs, A_prev, K, 1).detach()
            log_weight_rws += log_p_joint_curr - log_p_joint_prev - log_q_curr + log_q_prev

            Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)

        log_weights_rws[l] = log_weight_rws.detach()
        log_p_smcs[l] = log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        log_normalizers[l] = log_normalizer.detach()
        alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
        log_p_joints[l] = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        log_qs[l] = log_q_hmm(latents_dirs, A_samples, K, 1)


    log_weights_rws = log_weights_rws - log_sum_exp(log_weights_rws)
    weights_rws = torch.exp(log_weights_rws)
    kl =  torch.mul(weights_rws, kls).sum()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    loss_infer = - torch.mul(weights_rws, log_qs).sum()
    eubo =  torch.mul(weights_rws, log_p_joints - log_qs + log_normalizers - log_p_smcs).sum()
    return enc, loss_infer, eubo, kl, ess, latents_dirs, Z_ret

def rws2(enc, A_init, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    log_weights_rws = torch.zeros(num_particles_rws)
    log_p_joints = torch.zeros(num_particles_rws)
    log_normalizers = torch.zeros(num_particles_rws)
    log_p_smcs = torch.zeros(num_particles_rws)
    log_qs = torch.zeros(num_particles_rws)
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
#         A_samples = initial_trans(alpha_trans_0, K)
        A_samples = A_init
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_weight_rws = log_normalizer
        for m in range(mcmc_steps):
            A_prev = A_samples
            latents_dirs, A_samples = enc(Z_ret.contiguous().view(1, T*K), alpha_trans_0.sum().item())
            log_p_joint_curr = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
            log_p_joint_prev = log_joint(alpha_trans_0, Z_ret, Pi, A_prev, mu_ks, cov_ks, Y, T, D, K).detach()
            log_q_curr = log_q_hmm(latents_dirs, A_samples, K, 1).detach()
            log_q_prev = log_q_hmm(latents_dirs, A_prev, K, 1).detach()
            log_weight_rws += log_p_joint_curr - log_p_joint_prev - log_q_curr + log_q_prev

            Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)

        log_weights_rws[l] = log_weight_rws.detach()
        log_p_smcs[l] = log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        log_normalizers[l] = log_normalizer.detach()
        alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
        log_p_joints[l] = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach()
        log_qs[l] = log_q_hmm(latents_dirs, A_samples, K, 1)


    log_weights_rws = log_weights_rws - log_sum_exp(log_weights_rws)
    weights_rws = torch.exp(log_weights_rws)
    kl =  torch.mul(weights_rws, kls).sum()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    loss_infer = - torch.mul(weights_rws, log_qs).sum()
    eubo =  torch.mul(weights_rws, log_p_joints - log_qs + log_normalizers - log_p_smcs).sum()
    return enc, loss_infer, eubo, kl, ess, latents_dirs, Z_ret
