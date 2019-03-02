import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plots import *
from kls import *
from torch._six import inf
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from torch import logsumexp
import sys
import time
import datetime

def shuffler(batch_Xs, N, K, D, batch_size):
    indices = torch.cat([torch.randperm(N).unsqueeze(0) for b in range(batch_size)])
    indices_Xs = indices.unsqueeze(-1).repeat(1, 1, D)
    return torch.gather(batch_Xs, 1, indices_Xs)

def init_priors(K, D, batch_size):
    prior_mean = torch.zeros((batch_size, K, D))
    prior_nu = torch.ones((batch_size, K, D)) * 0.3
    prior_alpha = torch.ones((batch_size, K, D)) * 3.0
    prior_beta = torch.ones((batch_size, K, D)) * 3.0
    return prior_mean, prior_nu, prior_alpha, prior_beta


def post_global(Xs, Zs, prior_mean, prior_nu, prior_alpha, prior_beta, N, K, D, batch_size):
    stat1 = Zs.sum(1).unsqueeze(-1).repeat(1, 1, D) ## B * K * D
    xz_nk = torch.mul(Zs.unsqueeze(-1).repeat(1, 1, 1, D), Xs.unsqueeze(-1).repeat(1, 1, 1, K).transpose(-1, -2)) # B*N*K*D
    stat2 = xz_nk.sum(1) ## B*K*D
    stat3 = torch.mul(Zs.unsqueeze(-1).repeat(1, 1, 1, D), torch.mul(Xs, Xs).unsqueeze(-1).repeat(1, 1, 1, K).transpose(-1, -2)).sum(1) 
    stat1_nonzero = stat1
    stat1_nonzero[stat1_nonzero == 0.0] = 1.0
    x_bar = stat2 / stat1
    posterior_beta = prior_beta + (stat3 - (stat2 ** 2) / stat1_nonzero) / 2. + (stat1 * prior_nu / (stat1 + prior_nu)) * ((prior_nu**2) + x_bar**2 - 2 * x_bar *  prior_nu) / 2.
    posterior_nu = prior_nu + stat1
    posterior_mean = (prior_mean * prior_nu + stat2) / (prior_nu + stat1) 
    posterior_alpha = prior_alpha + (stat1 / 2.)
#     posterior_sigma = torch.sqrt(posterior_nu * (posterior_beta / posterior_alpha))
    return posterior_mean, posterior_nu, posterior_alpha, posterior_beta

def post_local(Xs, Pi, mus, precisions, N, K, D, batch_size):
    sigma = 1. / torch.sqrt(precisions)
    mus_expand = mus.unsqueeze(2).repeat(1, 1, N, 1)
    sigma_expand = sigma.unsqueeze(2).repeat(1, 1, N, 1)
    Xs_expand = Xs.unsqueeze(1).repeat(1, K, 1, 1)
    log_gammas = Normal(mus_expand, sigma_expand).log_prob(Xs_expand).sum(-1).transpose(-1, -2) # B * N * K
    log_pis = log_gammas - logsumexp(log_gammas, dim=-1).unsqueeze(-1)
    return log_pis

def log_joints_gmm(X, Z, Pi, mus, precisions, N, D, K, prior_mean, prior_nu, prior_alpha, prior_beta, batch_size):
    log_probs = torch.zeros(batch_size).float()
    ## priors on mus and sigmas size B
    log_probs = log_probs + Gamma(prior_alpha, prior_beta).log_prob(precisions).sum(-1).sum(-1)
    prior_sigma = 1. / torch.sqrt(prior_nu * precisions)
    log_probs = log_probs + Normal(prior_mean, prior_sigma).log_prob(mus).sum(-1).sum(-1)
    ## Z B-by-T-by-K
    log_probs = log_probs + cat(Pi).log_prob(Z).sum(-1)
    labels = Z.nonzero()
    sigmas = 1. / torch.sqrt(precisions)
    log_probs = log_probs + Normal(mus[labels[:, 0], labels[:, -1], :].view(batch_size, N, D), 
                                   sigmas[labels[:, 0], labels[:, -1], :].view(batch_size, N, D)).log_prob(X).sum(-1).sum(-1)
    return log_probs

def E_step(enc_local, X, mus, precisions, N, D, K, batch_size):
    mus_flat = mus.view(-1, K*D).unsqueeze(1).repeat(1, N, 1)
    sigma = 1. / torch.sqrt(precisions)
    sigma_flat = sigma.view(-1, K*D).unsqueeze(1).repeat(1, N, 1)
    data = torch.cat((X, mus_flat, sigma_flat), -1).view(batch_size*N, -1)
    zs_pi, zs, log_q_z = enc_local(data, N, K, D, 1, batch_size)
    return zs_pi, zs[0], log_q_z[0]

def M_step(enc_global, X, N, D, K, batch_size):
    data = X.view(batch_size*N, -1)
    q_mean, q_nu, q_alpha, q_beta, q_sigma, mus, precisions = enc_global(data, K, D, 1, batch_size)  
    log_q_eta =  Normal(q_mean[0], q_sigma[0]).log_prob(mus[0]).sum(-1).sum(-1) + Gamma(q_alpha, q_beta).log_prob(precisions[0]).sum(-1).sum(-1)## B
    return q_mean[0], q_nu, q_alpha, q_beta, q_sigma[0], mus[0], precisions[0], log_q_eta


def save_results(eubos, elbos, esses, kls_global_ex, kls_global_in, kls_local_ex, kls_local_in, PATH, filename):
    fout = open('results/logs-' + PATH + filename + '.txt', 'w+')
    fout.write('EUBOs, ELBOs, ESSs, KLs_eta_ex, KLs_eta_in, KL_z_ex, KL_z_in\n')
    for i in range(len(eubos)):
        fout.write(str(eubos[i]) + ', ' + str(elbos[i]) + ', ' + str(esses[i]) 
                   + str(kls_global_ex[i]) + str(kls_global_in[i]) + str(kls_local_ex[i]) + str(kls_local_in[i]) + '\n')
    fout.close()
    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1e-1)   