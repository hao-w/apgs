import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
def params_to_nats(alpha, beta, mu, nu):
    return alpha - (1./2), - beta - (nu * (mu**2) / 2), nu * mu, - nu / 2

def nats_to_params(nat1, nat2, nat3, nat4):
    alpha = nat1 + (1./2)
    nu = -2 * nat4
    mu = nat3 / nu
    beta = - nat2 - (nu * (mu**2) / 2)
    return alpha, beta, mu, nu

def data_to_stats(ob, state):
    """
    stat1 : sum of I[z_n=k], S * B * K
    stat2 : sum of I[z_n=k]*x_n, S * B * K * D
    stat3 : sum of I[z_n=k]*x_n^2, S * B * K * D
    """
    stat1 = state.sum(2)
    state_expand = state.unsqueeze(-1).repeat(1, 1, 1, 1, ob.shape[-1])
    ob_expand = ob.unsqueeze(-1).repeat(1, 1, 1, 1, state.shape[-1]).transpose(-1, -2)
    stat2 = (state_expand * ob_expand).sum(2)
    stat3 = (state_expand * (ob_expand**2)).sum(2)
    return stat1, stat2, stat3

def Post_eta(ob, state, prior_alpha, prior_beta, prior_mu, prior_nu):
    stat1, stat2, stat3 = data_to_stats(ob, state)
    stat1_expand = stat1.unsqueeze(-1).repeat(1, 1, 1, ob.shape[-1]) ## S * B * K * D
    stat1_nonzero = stat1_expand
    stat1_nonzero[stat1_nonzero == 0.0] = 1.0
    x_bar = stat2 / stat1_nonzero
    post_alpha = prior_alpha + stat1_expand / 2
    post_nu = prior_nu + stat1_expand
    post_mu = (prior_mu * prior_nu + stat2) / (stat1_expand + prior_nu)
    post_beta = prior_beta + (stat3 - (stat2 ** 2) / stat1_nonzero) / 2. + (stat1_expand * prior_nu / (stat1_expand + prior_nu)) * ((x_bar - prior_nu)**2) / 2.
    return post_alpha, post_beta, post_mu, post_nu

def Post_z(ob, ob_tau, ob_mu, prior_pi):
    """
    conjugate posterior p(z | mu, tau, x) given mu, tau, x
    """
    N = ob.shape[-2]
    K = ob_mu.shape[-2]
    ob_sigma = 1. / ob_tau.sqrt()
    ob_mu_expand = ob_mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    ob_sigma_expand = ob_sigma.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    ob_expand = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
    log_gammas = Normal(ob_mu_expand, ob_sigma_expand).log_prob(ob_expand).sum(-1).transpose(-1, -2) + prior_pi.log() # S * B * N * K
    post_logits = F.softmax(log_gammas, dim=-1).log()
    return post_logits

## some standard KL-divergence functions
def kl_normal_normal(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

def kls_normals(q_mean, q_sigma, p_mean, p_sigma):
    Kl_ex = kl_normal_normal(q_mean, q_sigma, p_mean, p_sigma).sum(-1)
    Kl_in = kl_normal_normal(p_mean, p_sigma, q_mean, q_sigma).sum(-1)
    return Kl_ex, Kl_in

def kl_gamma_gamma(p_alpha, p_beta, q_alpha, q_beta):
    t1 = q_alpha * (p_beta / q_beta).log()
    t2 = torch.lgamma(q_alpha) - torch.lgamma(p_alpha)
    t3 = (p_alpha - q_alpha) * torch.digamma(p_alpha)
    t4 = (q_beta - p_beta) * (p_alpha / p_beta)
    return t1 + t2 + t3 + t4

def kls_gammas(q_alpha, q_beta, p_alpha, p_beta):
    KL_ex = kl_gamma_gamma(q_alpha, q_beta, p_alpha, p_beta).sum(-1)
    KL_in = kl_gamma_gamma(p_alpha, p_beta, q_alpha, q_beta).sum(-1)
    return KL_ex, KL_in


def kl_NG_NG(p_alpha, p_beta, p_mu, p_nu, q_alpha, q_beta, q_mu, q_nu):
    diff = q_mu - p_mu
    t1 = (1. / 2) * ((p_alpha / p_beta) *  (diff ** 2) * q_nu + (q_nu / p_nu) - (torch.log(q_nu) - torch.log(p_nu)) - 1)
    t2 = q_alpha * (torch.log(p_beta) - torch.log(q_beta)) - (torch.lgamma(p_alpha) - torch.lgamma(q_alpha))
    t3 = (p_alpha - q_alpha) * torch.digamma(p_alpha) - (p_beta - q_beta) * p_alpha / p_beta
    return t1 + t2 + t3

def kls_NGs(q_alpha, q_beta, q_mu, q_nu, p_alpha, p_beta, p_mu, p_nu):
    kl_ex = kl_NG_NG(q_alpha, q_beta, q_mu, q_nu, p_alpha, p_beta, p_mu, p_nu).sum(-1)
    kl_in = kl_NG_NG(p_alpha, p_beta, p_mu, p_nu, q_alpha, q_beta, q_mu, q_nu).sum(-1)
    return kl_ex, kl_in

from torch._six import inf

def kl_cat_cat(p_logits, q_logits, EPS):
    p_probs= p_logits.exp()
    ## To prevent from infinite KL due to ill-defined support of q
    q_logits[q_logits == -inf] = EPS
    t = p_probs * (p_logits - q_logits)
    # t[(q_probs == 0).expand_as(t)] = inf
    t[(p_probs == 0).expand_as(t)] = 0
    return t.sum(-1)

def kls_cats(q_logits, p_logits, EPS):
    KL_ex = kl_cat_cat(q_logits, p_logits, EPS)
    KL_in = kl_cat_cat(p_logits, q_logits, EPS)
    return KL_ex, KL_in
