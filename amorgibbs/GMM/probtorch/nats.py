import torch
import torch.nn as nn
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

def data_to_stats(x, z, N, K, D):
    stat1 = z.sum(2) ## S * B * K, sum of I[z_n=k]
    z_expand = z.unsqueeze(-1).repeat(1, 1, 1, 1, D) ## S * B * N * K * D
    x_expand = x.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2) ## S * B * N * K * D
    stat2 = (z_expand * x_expand).sum(2) ## S * B * K * D, sum of I[z_n=k]*x_n
    stat3 = (z_expand * (x_expand**2)).sum(2) ## S * B * K * D, sum of I[z_n=k]*x_n^2
    return stat1, stat2, stat3

def post_nats(stat_ones, stat_x, stat_x2, states, prior_alpha, prior_beta, prior_mu, prior_nu, N, K, D):
    stat1 = (states * stat_ones).sum(-2) ## S * B * K
    z_expand = states.unsqueeze(-2).repeat(1, 1, 1, D, 1) ## S * B * N * D * K
    stat2 = (z_expand * stat_x.unsqueeze(-1)).transpose(-1, -2).sum(2) ## S * B * K * D
    stat3 = (z_expand * stat_x2.unsqueeze(-1)).transpose(-1, -2).sum(2) ## S * B * K * D
    prior_nat1, prior_nat2, prior_nat3, prior_nat4 = params_to_nats(prior_alpha, prior_beta, prior_mu, prior_nu) ## K * D
    stat1_expand = stat1.unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
    post_nat1 = prior_nat1 + (stat1_expand / 2.) ## S * B * K * D
    post_nat2 = prior_nat2 - (stat3 / 2.)
    post_nat3 = prior_nat3 + stat2
    post_nat4 = prior_nat4 - (stat1_expand / 2.)
    alpha, beta, mu, nu = nats_to_params(post_nat1, post_nat2, post_nat3, post_nat4)
    return alpha, beta, mu, nu

def merge_stats(stat_ones, stat_x, stat_x2, states):
    s, b, n, d = stat_x.shape
    _, _, _, k = states.shape
    stat1 = (states * stat_ones).sum(-2) ## S * B * K
    z_expand = states.unsqueeze(-2).repeat(1, 1, 1, d, 1) ## S * B * N * D * K
    stat2 = (z_expand * stat_x.unsqueeze(-1)).transpose(-1, -2).sum(2).view(s, b, (k*D)) ## S * B * (K*D)
    stat3 = (z_expand * stat_x2.unsqueeze(-1)).transpose(-1, -2).sum(2).view(s, b, (k*D)) ## S * B * (K*D)
    return torch.cat((stat1.unsqueeze(-1), stat2, stat3), -1) ## S * B * K * (1 + K*D + K*D)
