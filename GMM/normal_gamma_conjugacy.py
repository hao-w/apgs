import torch
import torch.nn as nn
import torch.nn.functional as F
from normal_gamma_kls import *
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from torch import logsumexp

def params_to_nats(alpha, beta, mu, nu):
    return alpha - (1./2), - beta - (nu * (mu**2) / 2), nu * mu, - nu / 2

def nats_to_params(nat1, nat2, nat3, nat4):
    alpha = nat1 + (1./2)
    nu = -2 * nat4
    mu = nat3 / nu
    beta = - nat2 - (nu * (mu**2) / 2)
    return alpha, beta, mu, nu

def data_to_stats(obs, states, K, D):
    """
    stat1 : sum of I[z_n=k], S * B * K
    stat2 : sum of I[z_n=k]*x_n, S * B * K * D
    stat3 : sum of I[z_n=k]*x_n^2, S * B * K * D
    """
    stat1 = states.sum(2)
    states_expand = states.unsqueeze(-1).repeat(1, 1, 1, 1, D)
    obs_expand = obs.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2)
    stat2 = (states_expand * obs_expand).sum(2)
    stat3 = (states_expand * (obs_expand**2)).sum(2)
    return stat1, stat2, stat3

def Post_mu_tau_nat(obs, stat1, stat2, stat3, prior_nat1, prior_nat2, prior_nat3, prior_nat4, K, D):
    """
    natural parameters of conjugate posterior given the priors and sufficient statistics
    """
    stat1_expand = stat1.unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
    return prior_nat1 + (stat1_expand / 2.), prior_nat2 - (stat3 / 2.), prior_nat3 + stat2, prior_nat4 - (stat1_expand / 2.)

def Post_eta(obs, states, prior_alpha, prior_beta, prior_mu, prior_nu, K, D):
    """
    distribution parameters of posterior given priors and obs and states
    """
    stat1, stat2, stat3 = data_to_stats(obs, states, K, D)
    prior_nat1, prior_nat2, prior_nat3, prior_nat4 = params_to_nats(prior_alpha, prior_beta, prior_mu, prior_nu) ## K * D
    post_nat1, post_nat2, post_nat3, post_nat4 = Post_mu_tau_nat(obs, stat1, stat2, stat3 , prior_nat1, prior_nat2, prior_nat3, prior_nat4, K, D)
    post_alpha, post_beta, post_mu, post_nu = nats_to_params(post_nat1, post_nat2, post_nat3, post_nat4)
    return post_alpha, post_beta, post_mu, post_nu

# def Post_mu_tau(stat1, stat2, stat3, prior_alpha, prior_beta, prior_mu, prior_nu, D):
#     """
#     distribution parameters of conjugate posterior given priors and sufficient statistics
#     """
#     stat1_expand = stat1.unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
#     stat1_nonzero = stat1_expand
#     stat1_nonzero[stat1_nonzero == 0.0] = 1.0
#     x_bar = stat2 / stat1_nonzero
#     post_beta = prior_beta + (stat3 - (stat2 ** 2) / stat1_nonzero) / 2. + (stat1_expand * prior_nu / (stat1_expand + prior_nu)) * ((x_bar - prior_nu)**2) / 2.
#     post_nu = prior_nu + stat1_expand
#     post_mu = (prior_mu * prior_nu + stat2) / (prior_nu + stat1_expand)
#     post_alpha = prior_alpha + (stat1_expand / 2.)
#     return post_alpha, post_beta, post_mu, post_nu

def Post_z(obs, obs_sigma, obs_mu, prior_pi, N, K):
    """
    conjugate posterior p(z | mu, tau, x) given mu, tau, x
    """
    obs_mu_expand = obs_mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    obs_sigma_expand = obs_sigma.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    obs_expand = obs.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
    log_gammas = Normal(obs_mu_expand, obs_sigma_expand).log_prob(obs_expand).sum(-1).transpose(-1, -2) * prior_pi # S * B * N * K
    post_logits = F.softmax(log_gammas, dim=-1).log()
    return post_logits


def Post_mu(stat1, stat2, prior_mu, prior_sigma, obs_sigma, D):
    """
    conjugate posterior p(mu | sigma, z, x) given sigma, z, x
    """
    post_sigma2 = 1. / ((1. / (prior_sigma**2)) + stat1.unsqueeze(-1).repeat(1,1,1,D) / (obs_sigma**2))
    post_sigma = post_sigma2.sqrt()
    post_mu = prior_mu / (prior_sigma**2) + (stat2 / (obs_sigma**2))
    post_mu = post_mu * post_sigma2
    return post_mu, post_sigma

def Post_tau(stat1, stat2, stat3, prior_alpha, prior_beta, obs_mu, D):
    """
    conjugate posterior p(tau | mu, z, x) given mu, z, x
    """
    stat1_expand = stat1.unsqueeze(-1).repeat(1,1,1,D)
    post_alpha = prior_alpha +  stat1_expand / 2
    post_beta = prior_beta + stat3 + (obs_mu**2) * stat1_expand - 2 * obs_mu * stat2
    return post_alpha, post_beta
