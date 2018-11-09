import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from smc_v import *

def sample_pixels(T, pixels, seq_ind):
    directory = "/home/hao/Research/amortized/AmortizedGibbs/images_test/"
    seq_imgs = torch.zeros((T+1, pixels, pixels))
    for t in range(T+1):
        filename = '%s-%s.png' % (str(seq_ind), str(t))
        img = plt.imread(directory+filename)[:, :, 0]
        img = np.where(img == 1.0, 0.0, 1.0)
        seq_imgs[t] = torch.from_numpy(img)
        # plt.imshow(img, cmap='gray')
        # plt.show()
    return seq_imgs

def initial_trans(alpha_trans_0, K, num_particles_rws):
    # A = torch.zeros((K, K)).float()
    # for k in range(K):
    #     A[k] = Dirichlet(alpha_trans_0[k]).sample()
    A = torch.ones((K, K)).float()
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

def log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws):
    log_joints = torch.zeros(num_particles_rws).float()
    labels = Z_ret.nonzero()
    labels_trans = (labels.view(num_particles_rws, T, -1)[:, :-1, :]).contiguous().view(num_particles_rws*(T-1), 3)
    Z_ret_trans = Z_ret[:, 1:, :].contiguous().view(num_particles_rws*(T-1), -1)
    Ys = Y.repeat(num_particles_rws, 1, 1).view(num_particles_rws*T, D)
    log_joints = log_joints + (MultivariateNormal(mu_ks[labels[:,-1]], cov_ks[labels[:,-1]]).log_prob(Ys)).view(num_particles_rws, T).sum(1)
    log_joints = log_joints + cat(Pi).log_prob(Z_ret[:, 0, :])
    log_joints = log_joints + (cat(A_samples[labels_trans[:,0], labels_trans[:,-1]]).log_prob(Z_ret_trans).view(num_particles_rws, T-1).sum(1))
    log_joints = log_joints + Dirichlet(alpha_trans_0).log_prob(A_samples).sum(1)
    return log_joints


def log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws):
    log_q = Dirichlet(latents_dirs).log_prob(A_samples)
    return log_q.sum(1)

def log_q_hmm(latents_dirs, A_samples):
    log_q = Dirichlet(latents_dirs).log_prob(A_samples)
    return log_q.sum()

def kl_dirichlets(alpha_trans_0, latents_dirs, Zs, T, K):
    alpha_trans_hat = alpha_trans_0 + pairwise(Zs, T).sum(0)
    kl = 0.0
    for k in range(K):
        kl += kl_dirichlet(alpha_trans_hat[k], latents_dirs[k])
    return kl

def kl_dirichlets_v(alpha_trans_0, latents_dirs, Z_ret, T, K, num_particles_rws):
    kls = torch.zeros(num_particles_rws)
    for l in range(num_particles_rws):
        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs[l], Z_ret[l], T, K)
    return kls

def kl_dirichlet(alpha1, alpha2):
    A = torch.lgamma(alpha1.sum()) - torch.lgamma(alpha2.sum())
    B = (torch.lgamma(alpha1) - torch.lgamma(alpha2)).sum()
    C = (torch.mul(alpha1 - alpha2, torch.digamma(alpha1) - torch.digamma(alpha1.sum()))).sum()
    kl = A - B + C
    return kl

def rws_v(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    A_samples = initial_trans(alpha_trans_0, K, num_particles_rws)
    Zs, log_weights, log_normalizers = smc_hmm_v(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
    Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
    log_weight_rwss = log_normalizers
    Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1) ## rws-by-2-by-T-1-by-K
    for m in range(mcmc_steps):
        A_prevs = A_samples
        A_samples = torch.zeros((num_particles_rws, K, K))
        latents_dirs = torch.zeros((num_particles_rws, K, K))
        for l in range(num_particles_rws):
            latents_dir, A_sample = enc(Z_ret_pair[l].transpose(0,1).contiguous().view(T-1, 2*K))
            A_samples[l] = A_sample
            latents_dirs[l] = latents_dir

        log_p_joint_currs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
        log_p_joint_prevs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_prevs, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()

        log_q_currs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws).detach()
        log_q_prevs = log_q_hmm_v(latents_dirs, A_prevs, K, num_particles_rws).detach()
        ## increment weights
        log_weight_rwss = log_weight_rwss + (log_p_joint_currs - log_p_joint_prevs - log_q_currs + log_q_prevs)
        ## csmc
        Zs, log_weights, log_normalizers = csmc_hmm_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
        Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
        Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1)

    kls = kl_dirichlets_v(alpha_trans_0, latents_dirs, Z_ret, T, K, num_particles_rws)

    log_p_smcss = log_joint_smc_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
    log_p_joints = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
    log_qs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws)

    log_weight_rwss_norm = log_weight_rwss - logsumexp(log_weight_rwss, dim=0)
    log_weight_rwss_norm = log_weight_rwss_norm.detach()
    weight_rwss = torch.exp(log_weight_rwss_norm)
    kl =  torch.mul(weight_rwss, kls).sum()
    ess = (1. / (weight_rwss ** 2 ).sum()).item()
    loss_infer = - torch.mul(weight_rwss, log_qs).sum()
    eubo =  torch.mul(weight_rwss, log_weight_rwss).sum()
    return enc, eubo, kl, ess, latents_dirs, Z_ret, loss_infer

def rws_v2(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
    A_samples = initial_trans(alpha_trans_0, K, num_particles_rws)
    Zs, log_weights, log_normalizers = smc_hmm_v(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
    Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
    log_weight_rwss = log_normalizers
    Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1) ## rws-by-2-by-T-1-by-K
    for m in range(mcmc_steps):
        A_prevs = A_samples
        A_samples = torch.zeros((num_particles_rws, K, K))
        latents_dirs = torch.zeros((num_particles_rws, K, K))
        for l in range(num_particles_rws):
            latents_dir, A_sample = enc(Z_ret_pair[l].transpose(0,1).contiguous().view(T-1, 2*K))
            A_samples[l] = A_sample
            latents_dirs[l] = latents_dir

        log_p_joint_currs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
        log_p_joint_prevs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_prevs, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()

        log_q_currs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws)
        log_q_prevs = log_q_hmm_v(latents_dirs, A_prevs, K, num_particles_rws).detach()
        ## increment weights
        log_weight_rwss = log_weight_rwss + (log_p_joint_currs - log_p_joint_prevs - log_q_currs + log_q_prevs)
        ## csmc
        Zs, log_weights, log_normalizers = csmc_hmm_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
        Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
        Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1)

    kls = kl_dirichlets_v(alpha_trans_0, latents_dirs, Z_ret, T, K, num_particles_rws)

    log_p_smcss = log_joint_smc_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
    log_p_joints = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
    log_qs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws)

    log_weight_rwss_norm = log_weight_rwss - logsumexp(log_weight_rwss, dim=0)
    log_weight_rwss_norm = log_weight_rwss_norm.detach()
    weight_rwss = torch.exp(log_weight_rwss_norm)
    kl =  torch.mul(weight_rwss, kls).sum()
    ess = (1. / (weight_rwss ** 2 ).sum()).item()
    loss_infer = - torch.mul(weight_rwss, log_qs).sum()
    eubo =  torch.mul(weight_rwss, log_weight_rwss).sum()
    return enc, eubo, kl, ess, latents_dirs, Z_ret, loss_infer

# def rws_v2(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, sis_steps):
#     A_samples = initial_trans(alpha_trans_0, K, num_particles_rws)
#     Zs, log_weights, log_normalizers = smc_hmm_v(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
#     Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
#     Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1) ## rws-by-2-by-T-1-by-K
#     log_p_joint_currs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#     log_p_smcss = log_joint_smc_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#     log_weight_rwss = log_p_joint_currs + log_normalizers - log_p_smcss
#     for m in range(sis_steps):
#         A_prevs = A_samples
#         A_samples = torch.zeros((num_particles_rws, K, K))
#         latents_dirs = torch.zeros((num_particles_rws, K, K))
#         for l in range(num_particles_rws):
#             latents_dir, A_sample = enc(Z_ret_pair[l].transpose(0,1).contiguous().view(T-1, 2*K))
#             A_samples[l] = A_sample
#             latents_dirs[l] = latents_dir
#         log_q_currs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws)
#
#         ## csmc
#         Zs, log_weights, log_normalizers = csmc_hmm_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
#         Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
#         Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1)
#         log_p_smcss = log_joint_smc_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#         log_p_joint_currs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#         log_weight_rwss = log_weight_rwss + log_p_joint_currs + log_normalizers - log_p_smcss - log_q_currs
#     kls = kl_dirichlets_v(alpha_trans_0, latents_dirs, Z_ret, T, K, num_particles_rws)
#
#     # log_p_smcss = log_joint_smc_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#     # log_p_joints = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#     # log_qs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws)
#
#     log_weight_rwss_norm = log_weight_rwss - logsumexp(log_weight_rwss, dim=0)
#     log_weight_rwss_norm = log_weight_rwss_norm.detach()
#     weight_rwss = torch.exp(log_weight_rwss_norm)
#     kl =  torch.mul(weight_rwss, kls).sum()
#     ess = (1. / (weight_rwss ** 2 ).sum()).item()
#     # loss_infer = - torch.mul(weight_rwss, log_qs).sum()
#     eubo =  torch.mul(weight_rwss, log_weight_rwss).sum()
#     return enc, eubo, kl, ess, latents_dirs, Z_ret


# def rws_pixels(amor_enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps):
#     A_samples = initial_trans(alpha_trans_0, K, num_particles_rws)
#     Zs, log_weights, log_normalizers = smc_hmm_v(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
#     Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
#     log_weight_rwss = log_normalizers
#     Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1) ## rws-by-2-by-T-1-by-K
#
#     for m in range(mcmc_steps):
#         A_prevs = A_samples
#         A_samples = torch.zeros((num_particles_rws, K, K))
#         latents_dirs = torch.zeros((num_particles_rws, K, K))
#         for l in range(num_particles_rws):
#             latents_dir, A_sample = amor_enc(Z_ret_pair[l].transpose(0,1).contiguous().view(T-1, 2*K))
#             A_samples[l] = A_sample
#             latents_dirs[l] = latents_dir
#
#         log_p_joint_currs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#         log_p_joint_prevs = log_joint_v(alpha_trans_0, Z_ret, Pi, A_prevs, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#
#         log_q_currs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws)
#         log_q_prevs = log_q_hmm_v(latents_dirs, A_prevs, K, num_particles_rws)
#         ## increment weights
#         log_weight_rwss = log_weight_rwss + (log_p_joint_currs - log_p_joint_prevs - log_q_currs + log_q_prevs)
#         ## csmc
#         Zs, log_weights, log_normalizers = csmc_hmm_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws)
#         Z_ret = resampling_smc_v(Zs, log_weights, num_particles_rws)
#         Z_ret_pair = torch.cat((Z_ret[:, :T-1, :].unsqueeze(1), Z_ret[:, 1:, :].unsqueeze(1)), 1)
#
#     kls = kl_dirichlets_v(alpha_trans_0, latents_dirs, Z_ret, T, K, num_particles_rws)
#
#     log_p_smcss = log_joint_smc_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#     log_p_joints = log_joint_v(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws).detach()
#     log_qs = log_q_hmm_v(latents_dirs, A_samples, K, num_particles_rws)
#
#     return amor_enc, log_weight_rwss, log_p_smcss, log_qs, log_p_joints, kls
