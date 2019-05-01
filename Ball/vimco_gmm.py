import matplotlib.pyplot as plt
import time
import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from smc import *
from util import *
from data import *

def flatz(Z, T, K, batch_size):
    return torch.cat((Z[:, :T-1, :].unsqueeze(2), Z[:, 1:, :].unsqueeze(2)), 2).view(batch_size * (T-1), 2*K)

def init_mus(K, D, batch_size):
    mus = Normal(torch.zeros((batch_size, K, D)), torch.ones((batch_size, K, D))  * 0.5).sample()
    log_p_init = Normal(torch.zeros((batch_size, K, D)), torch.ones((batch_size, K, D))  * 0.5).log_prob(mus).sum(-1).sum(-1)
    return mus, log_p_init

def E_step_gmm(mus, Sigmas, Ys, T, D, K):
    ## Ys is B-by-T-by-D
    for k in range(K):
        gammas = Normal(mus.unsqueeze(0).repeat(T, 1, 1, 1).transpose(0,2), Sigmas.unsqueeze(0).repeat(T, 1, 1, 1).transpose(0,2)).log_prob(Ys).sum(-1).transpose(0,1).transpose(1,2)
    Z = cat(F.softmax(gammas, -1)).sample()
    log_qz = cat(F.softmax(gammas, -1)).log_prob(Z).sum(-1)
    return Z, log_qz

def log_qs_gmm(mean, std, mus):
    log_q = Normal(mean, std).log_prob(mus).sum(-1).sum(-1)
    return log_q

def log_joints_gmm(Z, Pi, mus, Sigmas, Ys, T, D, K, batch_size):
    log_probs = torch.zeros(batch_size).float()
    ## mus B-by-K-by-D
    log_probs = log_probs + Normal(torch.zeros((batch_size, K, D)), torch.ones((batch_size, K, D)) * 0.5).log_prob(mus).sum(-1).sum(-1)
    ## Z B-by-T-by-K
    log_probs = log_probs + cat(Pi).log_prob(Z).sum(-1)
    labels = Z.nonzero()
    log_probs = log_probs + Normal(mus[labels[:, 0], labels[:, -1]].view(batch_size, T, D), Sigmas[labels[:, 0], labels[:, -1]].view(batch_size, T, D)).log_prob(Ys).sum(-1).sum(-1)
    return log_probs

def eubo_hmm_rws(prior, Pi, mus, covs, Ys, T, D, K, rws_samples, smc_samples, steps, batch_size):
    log_final_weights = torch.zeros((rws_samples, batch_size)).float()
    for m in range(steps):
        log_increment_weights = torch.zeros((rws_samples, batch_size)).float()
        Zs_cand = torch.zeros((rws_samples, batch_size, T, K))
        if m == 0:
            for r in range(rws_samples):
                As = initial_trans(prior, K, batch_size)
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mus, covs, Ys, T, D, K, smc_samples, batch_size)
                Z = smc_resamplings(Zs, log_weights, batch_size)
                Zs_cand[r] = Z
                log_increment_weights[r] = log_normalizers
            Zs_samples = adapt_resampling(Zs_cand, log_increment_weights, rws_samples)

        elif m == (steps-1):
            for r in range(rws_samples):
                Z_pairs = flatz(Zs_samples[r], T, K, batch_size)
                alphas, As = enc(Z_pairs, prior, batch_size)
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mus, covs, Ys, T, D, K, smc_samples, batch_size)
                Z = smc_resamplings(Zs, log_weights, batch_size)
                log_p_prior = Dirichlet(prior).log_prob(As).sum(-1)
                log_q_enc = Dirichlet(alphas).log_prob(As).sum(-1)
                log_final_weights[r] =  log_p_prior - log_q_enc + log_normalizers


        else:
            for r in range(rws_samples):
                Z_pairs = flatz(Zs_samples[r], T, K, batch_size)
                alphas, As = enc(Z_pairs, prior, batch_size)
                Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mus, covs, Ys, T, D, K, smc_samples, batch_size)
                Z = smc_resamplings(Zs, log_weights, batch_size)
                log_p_prior = Dirichlet(prior).log_prob(As).sum(-1)
                log_q_enc = Dirichlet(alphas).log_prob(As).sum(-1)
                log_increment_weights[r] =  log_p_prior - log_q_enc + log_normalizers
            Zs_samples = adapt_resampling(Zs_cand, log_increment_weights, rws_samples)
        weights = torch.exp(log_final_weights - logsumexp(log_final_weights, dim=0)).detach()
        eubo = torch.mul(weights, log_final_weights).sum(0).mean()
        elbo = log_final_weights.mean(0).mean()
        ess = (1. / (weights ** 2).sum(0)).mean()

    return eubo, elbo, ess
