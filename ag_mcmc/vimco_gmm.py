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

def ag_mcmc_vimco_gmm(enc, Zs_true, Pi, Sigmas, Ys, T, D, K, num_samples, mcmc_steps, batch_size):
    """
    mcmc sampling scheme 2
    vimco gradient estimator
    """
    log_increment_weights = torch.zeros((batch_size, num_samples)).float()
    log_q_mcmc = torch.zeros((batch_size, num_samples)).float()
    Zs_candidates = torch.zeros((num_samples, batch_size, T, K)).float()
    mus_candidates = torch.zeros((num_samples, batch_size, K, D)).float()

    for m in range(mcmc_steps):
        if m == 0:
            for l in range(num_samples):
                mus, log_p_init = init_mus(K, D, batch_size)
                mus_candidates[l] = mus
                Z, log_qz = E_step_gmm(mus, Sigmas, Ys, T, D, K)
                Zs_candidates[l] = Z
                ## the first incremental weight is just log normalizer since As is sampled from prior
                log_increment_weights[:, l] = - log_p_init
                log_q_mcmc[:, l] = log_p_init + log_qz
                # exclusive_kls[:, l] = exclusive_kls[:, l] + log_qs(conj_posts, As) - log_qs(prior_mcmc, As)
        else:
            for l in range(num_samples):
                Ys_Z = torch.cat((Ys, Zs_candidates[l]), -1)
                mean, std, mus = enc(Ys_Z, T, K, batch_size)
                mus_candidates[l] = mus
                ## Sigmas needs to be B-K-D
                Z, log_qz = E_step_gmm(mus, Sigmas, Ys, T, D, K)
                Zs_candidates[l] = Z
                Ys_Z = torch.cat((Ys, Z), -1)
                mean_new, std_new, mus_notusing = enc(Ys_Z, T, K, batch_size)
                mus_prev = mus_candidates[l].clone()
                log_increment_weights[:, l] =  log_increment_weights[:, l] - log_qs_gmm(mean, std, mus) + log_qs_gmm(mean_new, std_new, mus_prev)
                mus_candidates[l] = mus
                log_q_mcmc[:, l] = log_q_mcmc[:, l] + log_qs_gmm(mean, std, mus) + log_qz
                if m == (mcmc_steps-1):
                    log_increment_weights[:, l] =  log_increment_weights[:, l] - log_qz + log_joints_gmm(Z, Pi, mus, Sigmas, Ys, T, D, K, batch_size)

    elbos = log_increment_weights.mean(-1)

    log_sum_weights = logsumexp(log_increment_weights, -1)
    log_K = torch.log(torch.FloatTensor([num_samples]))
    term1_A = (log_sum_weights - log_K).unsqueeze(-1).repeat(1, num_samples).detach()
    term1_B1 = (log_increment_weights.sum(-1).unsqueeze(-1).repeat(1, num_samples) - log_increment_weights) / (num_samples - 1.0)
    expand = log_increment_weights.unsqueeze(1).repeat(1, num_samples, 1)
    extra_col = - torch.diagonal(expand, offset=0, dim1=1, dim2=2)
    term1_B2 = logsumexp(torch.cat((expand, extra_col.unsqueeze(-1)), -1), dim=-1)

    term1_B = (logsumexp(torch.cat((term1_B1.unsqueeze(-1), term1_B2.unsqueeze(-1)), -1), dim=-1) - log_K).detach()
    term1 = torch.mul(term1_A - term1_B, log_q_mcmc).sum(-1)
    term2 = log_sum_weights - log_K
    gradient = term1 + term2
    return gradient.mean(), elbos.mean()
