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
    A = torch.zeros((K, K)).float()
    for k in range(K):
        A[k] = Dirichlet(alpha_trans_0[k]).sample()
    return A

def make_cov(L):
    L_star = L.tril(-1) + L.diag().pow(2.0).diag()
    C = torch.matmul(L, L.t())
    E = torch.diag(C).pow(-0.5).diag()
    return torch.matmul(E, torch.matmul(C, E))


def pirors(Y, T, D, K):
    ## set up prior
    alpha_init_0 = torch.ones(K)
    # L = torch.ones((K, K))  / (2 *(K-1))
    #alpha_trans_0 = torch.cat((torch.cat((torch.eye(4)*0.5, torch.ones((4, K-4)) * (0.5 / (K-4))), 1), torch.ones((K-4, K)) * (1.0 / K)), 0)
    # m_0 = torch.FloatTensor([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * (1 / math.sqrt(2))
    alpha_trans_0 = torch.ones((K, K)) * (1/ K)
    m_0 = Y.mean(0).float()
    beta_0 = 1.0
    nu_0 = 6.0
    # W_0 =  (nu_0-D-1) * torch.mm((Y - Y.mean(0)).transpose(0,1), (Y - Y.mean(0))) / (T)
    W_0 =  torch.mm((Y - Y.mean(0)).transpose(0,1), (Y - Y.mean(0))) / (T * nu_0)
    return alpha_init_0, alpha_trans_0, m_0, beta_0, nu_0, W_0

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

def stats(Zs, Y, D, K):
    N_ks = Zs.sum(0)
    Y_ks = torch.zeros((K, D))
    S_ks = torch.zeros((K, D, D))

    Zs_expanded = Zs.repeat(D, 1, 1)
    for k in range(K):
        if N_ks[k].item() != 0:
            Y_ks[k] = torch.mul(Zs_expanded[:, :, k].transpose(0,1), Y).sum(0) / N_ks[k]
            Zs_expanded2 = Zs_expanded[:, :, k].repeat(D, 1, 1).permute(2, 1, 0)
            Y_diff = Y - Y_ks[k]
            Y_bmm = torch.bmm(Y_diff.unsqueeze(2), Y_diff.unsqueeze(1))
            S_ks[k] = torch.mul(Zs_expanded2, Y_bmm).sum(0) / (N_ks[k])
    return N_ks, Y_ks, S_ks

## compute the I[z_t=i]I[z_t-1=j] by vectorization
def pairwise(Zs, T):
    return torch.bmm(Zs[:T-1].unsqueeze(-1), Zs[1:].unsqueeze(1))


    return
def gibbs_global(Zs, alpha_init_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, T, D, K):
    ## Zs is N * K tensor where each row is one-hot encoding of a sample of latent state
    ## sample /pi
    alpha_init_hat = alpha_init_0 + N_ks
    Pi = Dirichlet(alpha_init_hat).sample()
    # sample mu_k and Sigma_k
    nu_ks = nu_0 + N_ks
    beta_ks = beta_0 + N_ks
    m_ks = torch.zeros((K, D))
    W_ks = torch.zeros((K, D, D))
    cov_ks = torch.zeros((K, D, D))
    mu_ks = torch.zeros((K, D))
    for k in range(K):
        m_ks[k] = (beta_0 * m_0 + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = (Y_ks[k] - m_0).view(D, 1)
        W_ks[k] = W_0 + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * torch.mm(temp2, temp2.transpose(0, 1))
        ## sample mu_k and Sigma_k from posterior
        cov_ks[k] = torch.from_numpy(invwishart.rvs(df=nu_ks[k].item(), scale=W_ks[k].data.numpy()))
        mu_ks[k] = MultivariateNormal(loc=m_ks[k], covariance_matrix=cov_ks[k] / beta_ks[k].item()).sample()
    return Pi, mu_ks, cov_ks

def log_joint(alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, Pi, A, mu_ks, cov_ks, Y, T, D, K):
    log_joint_prob = 0.0
    ## some vectorization to pick up the k-th global  for each state by one hot encoding
    decode_onehot = torch.arange(K).repeat(T, 1).float()
    labels = torch.bmm(decode_onehot.unsqueeze(1), Zs.unsqueeze(2)).squeeze(-1).squeeze(-1).int()
    # Y_ll_means = torch.bmm(Zs.unsqueeze(1), mu_ks.repeat(T, 1, 1)).squeeze(1)
    # Y_ll_covs = torch.mul(Zs.transpose(0,1).repeat(D, D, 1, 1).permute(-1, 2, 1, 0),  cov_ks.repeat(T, 1, 1, 1)).squeeze(1)
    ## start compute LL
    for t in range(T):
        log_joint_prob += MultivariateNormal(mu_ks[labels[t].item()], cov_ks[labels[t].item()]).log_prob(Y[t]) # likelihood of obs
        if t == 0:
            log_joint_prob += cat(Pi).log_prob(Zs[t]) # z_1 | pi
        else:
            log_joint_prob += cat(A[labels[t-1].item()]).log_prob(Zs[t]) # z_t | z_t-1 = j*, A
    log_joint_prob += Dirichlet(alpha_init_0).log_prob(Pi)
    for k in range(K):
        log_joint_prob += Dirichlet(alpha_trans_0[k]).log_prob(A[k]) ## prior of A
        log_joint_prob += MultivariateNormal(m_0, cov_ks[k] / beta_0).log_prob(mu_ks[k])# prior of mu_ks
        log_joint_prob += invwishart.logpdf(cov_ks[k].data.numpy(), nu_0, W_0.data.numpy())# prior of cov_ks
    return log_joint_prob


def inclusive_kl(A_samples, latents_dirs, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_encoder):
    log_p_joint = torch.zeros(num_particles_encoder)
    for n in range(num_particles_encoder):
        A = A_samples[:, n, :]
        log_p_joint[n] = log_joint(alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, Pi, A, mu_ks, cov_ks, Y, T, D, K).item()

    log_p_cond = torch.zeros((K, num_particles_encoder))
    alpha_trans_hat = alpha_trans_0 + pairwise(Zs, T).sum(0)
    kl = kl_dirichlets(alpha_trans_0, latents_dirs, Zs, T, K)
    for k in range(K):
        log_p_cond[k] = Dirichlet(alpha_trans_hat[k]).log_prob(A_samples[k])
    #
    log_p_cond = log_p_cond.sum(0)
    log_q = log_q_hmm(latents_dirs, A_samples, K, num_particles_encoder)
    log_weights = log_p_joint - log_q - log_sum_exp(log_p_joint - log_q)
    log_weights = log_weights.detach()
    weights = torch.exp(log_weights)
    loss_inference = - torch.mul(weights, log_q).sum()
    ess = (1. / (weights ** 2 ).sum()).item()
    kl_est = torch.mul(weights, log_p_cond - log_q).sum().detach().item()
    return loss_inference, kl, kl_est, ess

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

# def inclusive_kl2(enc, A_samples, latents_dirs, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_rws, num_particles_encoder, num_particles_smc):
#     log_weights_rws = torch.zeros(num_particles_rws)
#     log_qs = torch.zeros(num_particles_rws)
#     log_p_conds = torch.zeros(num_particles_rws)
#     kl = 0.0
#     for l in range(num_particles_rws):
#         # sample a obs sequence
# #         Y, mu_true, cov_true, Zs_true, Pi_true, A_true = sampling_hmm(T, K, D)
#         # initialize A from prior
# #         A_samples = initial_trans(alpha_trans_0, K)
#         # SMC to generate a weighted sample set for local states
#         Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
#         # draw a sample from the sample set
#         Z_ret = resampling_smc(Zs, log_weights)
#         latents_dirs, A_samples_new = enc(Z_ret.contiguous().view(-1, T*K), num_particles_encoder)
#         loss_inference, kl, kl_est, ess, weights = inclusive_kl(A_samples_new, latents_dirs, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles_encoder)
#         log_p_joint = torch.zeros(num_particles_encoder)
#         for n in range(num_particles_encoder):
#             A = A_samples[:, n, :]
#             log_p_joint[n] = log_joint(alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, Pi, A, mu_ks, cov_ks, Y, T, D, K).item()
#
#         log_p_cond = torch.zeros((K, num_particles_rws))
#         alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
#     kl = kl_dirichlets(alpha_trans_0, latents_dirs, Zs, T, K)
#     for k in range(K):
#         log_p_cond[k] = Dirichlet(alpha_trans_hat[k]).log_prob(A_samples[k])
#     #
#     log_p_cond = log_p_cond.sum(0)
#     log_q = log_q_hmm(latents_dirs, A_samples, K, num_particles_rws)
#     log_weights = log_p_joint - log_q - log_sum_exp(log_p_joint - log_q)
#     log_weights = log_weights.detach()
#     weights = torch.exp(log_weights)
#     loss_inference = - torch.mul(weights, log_q).sum()
#     ess = (1. / (weights ** 2 ).sum()).item()
#     kl_est = torch.mul(weights, log_p_cond - log_q).sum().detach().item()
#
#
#         log_p_joint_curr = log_joint(alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Z_ret, Pi, A_samples_new, mu_ks, cov_ks, Y, T, D, K).detach().item()
#
#
#         log_p_cond = 0.0
#         alpha_trans_hat = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
#         for k in range(K):
#             log_p_cond += Dirichlet(alpha_trans_hat[k]).log_prob(A_samples_new[k])
#         kl += kl_dirichlets(alpha_trans_0, latents_dirs, Z_ret, T, K)
#         log_p_conds[l] = log_p_cond
#
#         log_qs[l] = log_q_hmm(latents_dirs, A_samples_new, K)
#         log_weights_rws[l] = log_p_joint_curr - log_q_hmm(latents_dirs, A_samples_new, K) + log_normalizer
#     kl /= num_particles_rws
#
#     log_weights_rws = (log_weights_rws - log_sum_exp(log_weights_rws)).detach()
#     weights_rws = torch.exp(log_weights_rws)
#     ess = (1. / (weights_rws ** 2 ).sum()).item()
#     loss_infer = - torch.mul(weights_rws, log_qs).sum()
#     kl_est = torch.mul(weights_rws, log_p_conds - log_qs).sum().detach().item()
