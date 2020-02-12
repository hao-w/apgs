import torch
import torch.nn.functional as F
# from resampling import resample
from kls_gmm import kl_gmm, posterior_eta, posterior_z
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat

"""
True Gibbs objective in GMM problem
==========
sampling scheme:
1. start with 'one-shot' predicting eta and z, resample
2. For m = 1 : apg_sweeps:
    update eta given z
    update z given eta
==========
abbreviations:
K -- number of clusters
D -- data dimensions (D=2 in GMM)
S -- sample size
B -- batch size
N -- number of data points in one (GMM) dataset
==========
variables:
ob : S * B * N * D, observations, as data points
tau: S * B * K * D, cluster precisions, as global variables
mu: S * B * K * D, cluster means, as global variables
eta := {tau, mu} global block
z : S * B * N * K, cluster assignments, as local variables
==========
"""
def gibbs_objective(model, gibbs_sweeps, ob, mode_required=False, density_required=False):
    trace = dict() ## a dictionary that tracks variables needed during the sweeping
    if mode_required:
        trace['E_tau'] = []
        trace['E_mu'] = []
        trace['E_z'] = []
    if density_required:
        trace['density'] = []
        # trace['ll'] = []

    (enc_rws_eta, enc_apg_z, generative) = model
    tau, mu, z, trace = rws(enc_rws_eta=enc_rws_eta,
                                   enc_rws_z=enc_apg_z,
                                   generative=generative,
                                   ob=ob,
                                   trace=trace,
                                   mode_required=mode_required,
                                   density_required=density_required)
    for m in range(gibbs_sweeps):
        tau, mu, z, trace = gibbs_sweep(generative=generative,
                                                ob=ob,
                                                z=z,
                                                trace=trace,
                                                mode_required=mode_required,
                                                density_required=density_required)
    if mode_required:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0) # (1 + apg_sweeps) * B * K * D
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)  # (1 + apg_sweeps) * B * K * D
        trace['E_z'] = torch.cat(trace['E_z'], 0) # (1 + apg_sweeps) * B * N * K
    if density_required:
        trace['density'] = torch.cat(trace['density'], 0) # (1 + apg_sweeps) * B
        # trace['ll'] = torch.cat(trace['ll'], 0) # (1 + apg_sweeps) * B
    return trace

def rws(enc_rws_eta, enc_rws_z, generative, ob, trace, mode_required, density_required):
    """
    One-shot for eta and z, like a normal RWS
    """
    q_eta = enc_rws_eta(ob=ob, prior_ng=generative.prior_ng, sampled=True)
    p_eta = generative.eta_prior(q=q_eta)
    log_q_eta = q_eta['means'].log_prob.sum(-1) + q_eta['precisions'].log_prob.sum(-1)
    log_p_eta = p_eta['means'].log_prob.sum(-1) + p_eta['precisions'].log_prob.sum(-1) # S * B * K
    tau = q_eta['precisions'].value
    mu = q_eta['means'].value
    q_z = enc_rws_z(ob, tau=tau, mu=mu, sampled=True)
    p_z = generative.z_prior(q=q_z)
    log_q_z = q_z['states'].log_prob
    log_p_z = p_z['states'].log_prob # S * B * N
    z = q_z['states'].value ## S * B * N * K
    ll = generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=False) # log likelihood
    log_p = ll.sum(-1) +  log_p_eta.sum(-1) + log_p_z.sum(-1)
    log_q =  log_q_eta.sum(-1) + log_q_z.sum(-1)

    if mode_required:
        E_tau = (q_eta['precisions'].dist.concentration / q_eta['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta['means'].dist.loc.mean(0).detach()
        E_z = q_z['states'].dist.probs.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
    if density_required:
        log_joint = log_p.detach()
        trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
        # trace['ll'].append(ll.sum(-1).mean(0).unsqueeze(0))

    return tau, mu, z, trace

def gibbs_sweep(generative, ob, z, trace, mode_required, density_required):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    post_alpha, post_beta, post_mu, post_nu = posterior_eta(ob=ob,
                                                              z=z,
                                                              prior_alpha=generative.prior_alpha,
                                                              prior_beta=generative.prior_beta,
                                                              prior_mu=generative.prior_mu,
                                                              prior_nu=generative.prior_nu)

    E_tau = (post_alpha / post_beta).mean(0)
    E_mu = post_mu.mean(0)

    tau = Gamma(post_alpha, post_beta).sample()
    mu = Normal(post_mu, 1. / (post_nu * tau).sqrt()).sample()

    posterior_logits = posterior_z(ob=ob,
                                   tau=tau,
                                   mu=mu,
                                   prior_pi=generative.prior_pi)
    E_z = posterior_logits.exp().mean(0)
    z = cat(logits=posterior_logits).sample()

    ll = generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=True)
    log_prior_tau = Gamma(generative.prior_alpha, generative.prior_beta).log_prob(tau).sum(-1).sum(-1)
    log_prior_mu = Normal(generative.prior_mu, 1. / (generative.prior_nu * tau).sqrt()).log_prob(mu).sum(-1).sum(-1)
    log_prior_z = cat(probs=generative.prior_pi).log_prob(z).sum(-1)

    log_joint = ll + log_prior_tau + log_prior_mu + log_prior_z
    if mode_required:
        trace['E_tau'].append(E_tau.unsqueeze(0)) # (1 + apg_sweeps) * B * K * D
        trace['E_mu'].append(E_mu.unsqueeze(0))  # (1 + apg_sweeps) * B * K * D
        trace['E_z'].append(E_z.unsqueeze(0)) # (1 + apg_sweeps) * B * N * K

    if density_required:
        trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
        # trace['ll'].append(ll.mean(0).unsqueeze(0))
    return tau, mu, z, trace
