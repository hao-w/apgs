import torch
import torch.nn as nn
# from torch._six import inf
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from torch import logsumexp

def kls_step(x, enc_init, enc_global, enc_local, q_eta, q_z, N, K, D, num_samples, batch_size):
    #### samples
    means = q_eta['means'].value.view(num_samples, batch_size, K, D)
    precisions = q_eta['precisions'].value.view(num_samples, batch_size, K, D)
    #### prior parameters
    p_alpha = enc_init.prior_alpha.view(K, D).unsqueeze(0).unsqueeze(0).repeat(num_samples, batch_size, 1, 1)
    p_beta = enc_init.prior_beta.view(K, D).unsqueeze(0).unsqueeze(0).repeat(num_samples, batch_size, 1, 1)
    p_mean = enc_init.prior_mean.view(K, D).unsqueeze(0).unsqueeze(0).repeat(num_samples, batch_size, 1, 1)
    p_nu = enc_init.prior_nu.view(K, D).unsqueeze(0).unsqueeze(0).repeat(num_samples, batch_size, 1, 1)
    p_pi = enc_local.prior_pi.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(num_samples, batch_size, N, 1)
    #### variational parameters
    q_logits = q_z['zs'].dist.probs.log()
    p_logits = post_local(x, p_pi, means, precisions, N, K, D, batch_size)
    kl_z_ex, kl_z_in = kls_cats(p_logits, q_logits)
    ## move one step forward given z
    zs = q_z['zs'].value
    q_eta, p_eta = enc_global(q_z, x)
    q_mean = q_eta['means'].dist.loc.view(num_samples, batch_size, K, D)
    q_nu = 1. / ((q_eta['means'].dist.scale.view(num_samples, batch_size, K, D) ** 2) * precisions)
    q_alpha = q_eta['precisions'].dist.concentration.view(num_samples, batch_size, K, D)
    q_beta = q_eta['precisions'].dist.rate.view(num_samples, batch_size, K, D)
    post_mean, post_nu, post_alpha, post_beta = post_global(x, zs, p_mean, p_nu, p_alpha, p_beta, N, K, D)
    kl_eta_ex, kl_eta_in = kls_NGs(post_mean, post_nu, post_alpha, post_beta, q_mean, q_nu, q_alpha, q_beta)
    return kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in

def kl_normal_normal(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

def kls_normals(q_mean, q_sigma, p_mean, p_sigma):
    Kl_ex = kl_normal_normal(q_mean, q_sigma, p_mean, p_sigma).sum(-1)
    Kl_in = kl_normal_normal(p_mean, p_sigma, q_mean, q_sigma).sum(-1)
    return Kl_ex, Kl_in

def Post_mu(stat1, stat2, prior_mu, prior_sigma, obs_sigma, D):
    post_sigma2 = 1. / ((1. / (prior_sigma**2)) + stat1.unsqueeze(-1).repeat(1,1,1,D) / (obs_sigma**2))
    post_sigma = post_sigma2.sqrt()
    post_mu = prior_mu / (prior_sigma**2) + (stat2 / (obs_sigma**2))
    post_mu = post_mu * post_sigma2
    return post_mu, post_sigma

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

def Post_tau(stat1, stat2, stat3, prior_alpha, prior_beta, obs_mu, D):
    stat1_expand = stat1.unsqueeze(-1).repeat(1,1,1,D)
    post_alpha = prior_alpha +  stat1_expand / 2
    post_beta = prior_beta + stat3 + (obs_mu**2) * stat1_expand - 2 * obs_mu * stat2
    return post_alpha, post_beta

def kl_NG_NG(p_mean, p_nu, p_alpha, p_beta, q_mean, q_nu, q_alpha, q_beta):
    diff = q_mean - p_mean
    t1 = (1. / 2) * ((p_alpha / p_beta) *  (diff ** 2) * q_nu + (q_nu / p_nu) - (torch.log(q_nu) - torch.log(p_nu)) - 1)
    t2 = q_alpha * (torch.log(p_beta) - torch.log(q_beta)) - (torch.lgamma(p_alpha) - torch.lgamma(q_alpha))
    t3 = (p_alpha - q_alpha) * torch.digamma(p_alpha) - (p_beta - q_beta) * p_alpha / p_beta
    return t1 + t2 + t3

def kls_NGs(q_mean, q_nu, q_alpha, q_beta, p_mean, p_nu, p_alpha, p_beta):
    kl_ex = kl_NG_NG(q_mean, q_nu, q_alpha, q_beta, p_mean, p_nu, p_alpha, p_beta).sum(-1)
    kl_in = kl_NG_NG(p_mean, p_nu, p_alpha, p_beta, q_mean, q_nu, q_alpha, q_beta).sum(-1)
    return kl_ex, kl_in

def Post_mu_tau(stat1, stat2, stat3, prior_mu, prior_nu, prior_alpha, prior_beta, D):
    stat1_expand = stat1.unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
    stat1_nonzero = stat1_expand
    stat1_nonzero[stat1_nonzero == 0.0] = 1.0
    x_bar = stat2 / stat1_nonzero
    post_beta = prior_beta + (stat3 - (stat2 ** 2) / stat1_nonzero) / 2. + (stat1_expand * prior_nu / (stat1_expand + prior_nu)) * ((x_bar - prior_nu)**2) / 2.
    post_nu = prior_nu + stat1_expand
    post_mu = (prior_mu * prior_nu + stat2) / (prior_nu + stat1_expand)
    post_alpha = prior_alpha + (stat1_expand / 2.)
    return post_mu, post_nu, post_alpha, post_beta


def kl_cat_cat(p_logits, q_logits, EPS=1e-8):
    p_probs = torch.exp(p_logits)
    q_probs = torch.exp(q_logits) + EPS
    t = p_probs * (p_logits - q_logits)
    # t[(q_probs == 0).expand_as(t)] = inf
    t[(p_probs == 0).expand_as(t)] = 0
    return t.sum(-1)

def kls_cats(p_logits, q_logits, EPS=1e-8):
    KL_ex = kl_cat_cat(q_logits, p_logits + EPS).sum(-1)
    KL_in = kl_cat_cat(p_logits, q_logits + EPS).sum(-1)
    return KL_ex, KL_in

def post_global(Xs, Zs, prior_mean, prior_nu, prior_alpha, prior_beta, N, K, D):
    Zs_fflat = Zs.unsqueeze(-1).repeat(1, 1, 1, 1, D)
    Xs_fflat = Xs.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2)
    stat1 = Zs.sum(2).unsqueeze(-1).repeat(1, 1, 1, D) ## S * B * K * D
    xz_nk = torch.mul(Zs_fflat, Xs_fflat) # S*B*N*K*D
    stat2 = xz_nk.sum(2) ## S*B*K*D
    stat3 = torch.mul(Zs_fflat, torch.mul(Xs_fflat, Xs_fflat)).sum(2) # S*B*K*D
    stat1_nonzero = stat1
    stat1_nonzero[stat1_nonzero == 0.0] = 1.0
    x_bar = stat2 / stat1
    posterior_beta = prior_beta + (stat3 - (stat2 ** 2) / stat1_nonzero) / 2. + (stat1 * prior_nu / (stat1 + prior_nu)) * ((prior_nu**2) + x_bar**2 - 2 * x_bar *  prior_nu) / 2.
    posterior_nu = prior_nu + stat1
    posterior_mean = (prior_mean * prior_nu + stat2) / (prior_nu + stat1)
    posterior_alpha = prior_alpha + (stat1 / 2.)
#     posterior_sigma = torch.sqrt(posterior_nu * (posterior_beta / posterior_alpha))
    return posterior_mean, posterior_nu, posterior_alpha, posterior_beta

def post_local(Xs, Pi, means, precisions, N, K, D, batch_size):
    sigmas = 1. / torch.sqrt(precisions)
    means_expand = means.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    sigmas_expand = sigmas.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    Xs_expand = Xs.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
    log_gammas = Normal(means_expand, sigmas_expand).log_prob(Xs_expand).sum(-1).transpose(-1, -2) # S * B * N * K
    logits = log_gammas - logsumexp(log_gammas, dim=-1).unsqueeze(-1)
    return logits
