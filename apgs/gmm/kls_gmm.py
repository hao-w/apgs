import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

def kls_eta(models, ob, z):
    """
    compute the KL divergence KL(p(\eta | x, z)|| q(\eta | x, z))
    """
    (_, _, enc_apg_eta, generative) = models   
    q_f_eta = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=True)
    mu = q_f_eta['means'].value
    tau = q_f_eta['precisions'].value
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_alpha = q_f_eta['precisions'].dist.concentration
    q_beta = q_f_eta['precisions'].dist.rate
    q_mu = q_f_eta['means'].dist.loc
    q_std = q_f_eta['means'].dist.scale
    q_nu = 1. / (tau * (q_std**2)) # nu*tau = 1 / std**2

    posterior_alpha, posterior_beta, posterior_mu, posterior_nu = posterior_eta(ob=ob,
                                                                                z=z,
                                                                                prior_alpha=generative.prior_alpha,
                                                                                prior_beta=generative.prior_beta,
                                                                                prior_mu=generative.prior_mu,
                                                                                prior_nu=generative.prior_nu)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha=q_alpha,
                                   q_beta=q_beta,
                                   q_mu=q_mu,
                                   q_nu=q_nu,
                                   p_alpha=posterior_alpha,
                                   p_beta=posterior_beta,
                                   p_mu=posterior_mu,
                                   p_nu=posterior_nu)

    inckl = kl_eta_in.sum(-1).mean().detach()
    exckl = kl_eta_ex.sum(-1).mean().detach()
    return exckl, inckl

# def kl_eta_and_z(enc_apg_eta, enc_apg_z, generative, ob, z):
#     q_f_eta = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=True)
#     mu = q_f_eta['means'].value
#     tau = q_f_eta['precisions'].value
#     ## KLs for mu and sigma based on Normal-Gamma prior
#     q_alpha = q_f_eta['precisions'].dist.concentration
#     q_beta = q_f_eta['precisions'].dist.rate
#     q_mu = q_f_eta['means'].dist.loc
#     q_std = q_f_eta['means'].dist.scale
#     q_nu = 1. / (tau * (q_std**2)) # nu*tau = 1 / std**2

#     q_f_z = enc_apg_z(ob=ob, tau=tau, mu=mu, sampled=True)
#     q_pi = q_f_z['states'].dist.probs
    
#     posterior_alpha, posterior_beta, posterior_mu, posterior_nu = posterior_eta(ob=ob,
#                                                                                 z=z,
#                                                                                 prior_alpha=generative.prior_alpha,
#                                                                                 prior_beta=generative.prior_beta,
#                                                                                 prior_mu=generative.prior_mu,
#                                                                                 prior_nu=generative.prior_nu)
#     kl_eta_ex, kl_eta_in = kls_NGs(q_alpha=q_alpha,
#                                    q_beta=q_beta,
#                                    q_mu=q_mu,
#                                    q_nu=q_nu,
#                                    p_alpha=posterior_alpha,
#                                    p_beta=posterior_beta,
#                                    p_mu=posterior_mu,
#                                    p_nu=posterior_nu)
#     posterior_logits = posterior_z(ob=ob,
#                                    tau=tau,
#                                    mu=mu,
#                                    prior_pi=generative.prior_pi)
#     kl_z_ex, kl_z_in = kls_cats(q_logits=q_pi.log(),
#                                 p_logits=posterior_logits)
#     inckls = {"inckl_eta" : kl_eta_in.sum(-1).mean(0).detach(),"inckl_z" : kl_z_in.sum(-1).mean(0).detach() }
#     return inckls

def params_to_nats(alpha, beta, mu, nu):
    """
    distribution parameters to natural parameters
    """
    return alpha - (1./2), - beta - (nu * (mu**2) / 2), nu * mu, - nu / 2

def nats_to_params(nat1, nat2, nat3, nat4):
    """
    natural parameters to distribution parameters
    """
    alpha = nat1 + (1./2)
    nu = -2 * nat4
    mu = nat3 / nu
    beta = - nat2 - (nu * (mu**2) / 2)
    return alpha, beta, mu, nu

def data_to_stats(ob, z):
    """
    pointwise sufficient statstics
    stat1 : sum of I[z_n=k], S * B * K * 1
    stat2 : sum of I[z_n=k]*x_n, S * B * K * D
    stat3 : sum of I[z_n=k]*x_n^2, S * B * K * D
    """
    stat1 = z.sum(2).unsqueeze(-1)
    z_expand = z.unsqueeze(-1).repeat(1, 1, 1, 1, ob.shape[-1])
    ob_expand = ob.unsqueeze(-1).repeat(1, 1, 1, 1, z.shape[-1]).transpose(-1, -2)
    stat2 = (z_expand * ob_expand).sum(2)
    stat3 = (z_expand * (ob_expand**2)).sum(2)
    return stat1, stat2, stat3

def posterior_eta(ob, z, prior_alpha, prior_beta, prior_mu, prior_nu):
    """
    conjugate postrior of eta, given the normal-gamma prior
    """
    stat1, stat2, stat3 = data_to_stats(ob, z)
    stat1_expand = stat1.repeat(1, 1, 1, ob.shape[-1]) ## S * B * K * D
    stat1_nonzero = stat1_expand
    stat1_nonzero[stat1_nonzero == 0.0] = 1.0
    x_bar = stat2 / stat1_nonzero
    post_alpha = prior_alpha + stat1_expand / 2
    post_nu = prior_nu + stat1_expand
    post_mu = (prior_mu * prior_nu + stat2) / (stat1_expand + prior_nu)
    post_beta = prior_beta + (stat3 - (stat2 ** 2) / stat1_nonzero) / 2. + (stat1_expand * prior_nu / (stat1_expand + prior_nu)) * ((x_bar - prior_nu)**2) / 2.
    return post_alpha, post_beta, post_mu, post_nu

def posterior_z(ob, tau, mu, prior_pi):
    """
    posterior of z, given the Gaussian likelihood and the uniform prior
    """
    N = ob.shape[-2]
    K = mu.shape[-2]
    sigma = 1. / tau.sqrt()
    mu_expand = mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    sigma_expand = sigma.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    ob_expand = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
    log_gammas = Normal(mu_expand, sigma_expand).log_prob(ob_expand).sum(-1).transpose(-1, -2) + prior_pi.log() # S * B * N * K
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

def kl_cat_cat(p_logits, q_logits, EPS=-1e14):
    p_probs= p_logits.exp()
    ## To prevent from infinite KL due to ill-defined support of q
    q_logits[q_logits == -inf] = EPS
    t = p_probs * (p_logits - q_logits)
    # t[(q_probs == 0).expand_as(t)] = inf
    t[(p_probs == 0).expand_as(t)] = 0
    return t.sum(-1)

def kls_cats(q_logits, p_logits):
    KL_ex = kl_cat_cat(q_logits, p_logits)
    KL_in = kl_cat_cat(p_logits, q_logits)
    return KL_ex, KL_in
