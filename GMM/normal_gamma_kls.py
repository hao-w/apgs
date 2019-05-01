import torch

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

def kl_cat_cat(p_logits, q_logits, EPS=1e-12):
    p_probs= p_logits.exp()
    ## To prevent from infinite KL due to ill-defined support of q
    q_logits[q_logits == -inf] = torch.log(torch.FloatTensor([EPS]))
    t = p_probs * (p_logits - q_logits)
    # t[(q_probs == 0).expand_as(t)] = inf
    t[(p_probs == 0).expand_as(t)] = 0
    return t.sum(-1)

def kls_cats(q_logits, p_logits):
    KL_ex = kl_cat_cat(q_logits, p_logits)
    KL_in = kl_cat_cat(p_logits, q_logits)
    return KL_ex, KL_in
