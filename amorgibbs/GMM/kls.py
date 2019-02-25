import torch
import torch.nn as nn
from torch._six import inf
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma

def kl_normal_normal(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

def kls_gaussians(mus_mean, mus_sigma, posterior_mean, posterior_sigma):
    Kl_exclusive = kl_normal_normal(mus_mean, mus_sigma, posterior_mean, posterior_sigma).sum(-1).sum(-1).mean()
    Kl_inclusive = kl_normal_normal(posterior_mean, posterior_sigma, mus_mean, mus_sigma).sum(-1).sum(-1).mean()
    return Kl_exclusive, Kl_inclusive

def kl_gamma_gamma(p_alpha, p_beta, q_alpha, q_beta):
    t1 = q_alpha * (p_beta / q_beta).log()
    t2 = torch.lgamma(q_alpha) - torch.lgamma(p_alpha)
    t3 = (p_alpha - q_alpha) * torch.digamma(p_alpha)
    t4 = (q_beta - p_beta) * (p_alpha / p_beta)
    return t1 + t2 + t3 + t4

def kls_gammas(precsions_alpha, precisions_beta, posterior_alpha, posterior_beta):
    KL_exclusive = kl_gamma_gamma(precsions_alpha, precisions_beta, posterior_alpha, posterior_beta).sum(-1).sum(-1).mean()
    KL_inclusive = kl_gamma_gamma(posterior_alpha, posterior_beta, precsions_alpha, precisions_beta).sum(-1).sum(-1).mean()
    return KL_exclusive, KL_inclusive

def kl_NG_NG(p_mean, p_nu, p_alpha, p_beta, q_mean, q_nu, q_alpha, q_beta):
    diff = q_mean - p_mean
    t1 = (1. / 2) * ((p_alpha / p_beta) *  (diff ** 2) * q_nu + (q_nu / p_nu) - (torch.log(q_nu) - torch.log(p_nu)) - 1)
    t2 = q_alpha * (torch.log(p_beta) - torch.log(q_beta)) - (torch.lgamma(p_alpha) - torch.lgamma(q_alpha)) 
    t3 = (p_alpha - q_alpha) * torch.digamma(p_alpha) - (p_beta - q_beta) * p_alpha / p_beta
    return t1 + t2 + t3

def kls_NGs(p_mean, p_nu, p_alpha, p_beta, q_mean, q_nu, q_alpha, q_beta):
    kl_exclusive = kl_NG_NG(q_mean, q_nu, q_alpha, q_beta, p_mean, p_nu, p_alpha, p_beta).sum(-1).sum(-1)
    kl_inclusive = kl_NG_NG(p_mean, p_nu, p_alpha, p_beta, q_mean, q_nu, q_alpha, q_beta).sum(-1).sum(-1)
    return kl_exclusive, kl_inclusive


def kl_cat_cat(p_logits, q_logits, EPS=1e-8):
    p_probs = torch.exp(p_logits)
    q_probs = torch.exp(q_logits) + EPS
    t = p_probs * (p_logits - q_logits)
    t[(q_probs == 0).expand_as(t)] = inf
    t[(p_probs == 0).expand_as(t)] = 0
    return t.sum(-1)

def kls_cats(p_logits, q_logits, EPS=1e-8):
    KL_ex = kl_cat_cat(q_logits, p_logits + EPS).sum(-1)
    KL_in = kl_cat_cat(p_logits, q_logits + EPS).sum(-1)
    return KL_ex, KL_in