import torch
import torch.nn.functional as F
import probtorch
from resampling import resample
from kls_gmm import kl_gmm
"""
Amortized Population Gibbs objective in GMM problem
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
sampling scheme:
1. start with 'one-shot' predicting eta by encoder os_eta
2. For m = 1 : apg_sweeps:
    update eta given z
    resample
    update z given eta
    resample
==========
"""
def apg_objective(models, apg_sweeps, ob, loss_required=True, ess_required=True, mode_required=False, density_required=False, kl_required=True):
    trace = dict() ## a dictionary that tracks variables needed during the sweeping
    if loss_required:
        trace['loss'] = []
    if ess_required:
        trace['ess_rws'] = []
        trace['ess_eta'] = []
        trace['ess_z'] = []
    if kl_required:
        trace['inckl_eta'] = []
        trace['inckl_z'] = []
    if mode_required:
        trace['E_tau'] = []
        trace['E_mu'] = []
        trace['E_z'] = []
    if density_required:
        trace['density'] = []

    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    w, tau, mu, z, trace = rws(enc_rws_eta=enc_rws_eta,
                               enc_rws_z=enc_apg_z,
                               generative=generative,
                               ob=ob,
                               trace=trace,
                               loss_required=loss_required,
                               ess_required=ess_required,
                               mode_required=mode_required,
                               density_required=density_required)
    tau = resample(var=tau, weights=w, dim_expand=False)
    mu = resample(var=mu, weights=w, dim_expand=False)
    z = resample(var=z, weights=w, dim_expand=False)
    for m in range(apg_sweeps):
        w, tau, mu, trace = apg_eta(enc_apg_eta=enc_apg_eta,
                                       generative=generative,
                                       ob=ob,
                                       z=z,
                                       tau_old=tau,
                                       mu_old=mu,
                                       trace=trace,
                                       loss_required=loss_required,
                                       ess_required=ess_required,
                                       mode_required=mode_required,
                                       density_required=density_required)
        tau = resample(var=tau, weights=w, dim_expand=True)
        mu = resample(var=mu, weights=w, dim_expand=True)
        w, z, trace = apg_z(enc_apg_z=enc_apg_z,
                            generative=generative,
                            ob=ob,
                            tau=tau,
                            mu=mu,
                            z_old=z,
                            trace=trace,
                            loss_required=loss_required,
                            ess_required=ess_required,
                            mode_required=mode_required,
                            density_required=density_required)
        z = resample(var=z, weights=w, dim_expand=True)

    if kl_required:
        inckls = kl_gmm(enc_apg_eta=enc_apg_eta,
                        enc_apg_z=enc_apg_z,
                        generative=generative,
                        ob=ob,
                        z=z)
        trace['inckl_eta'] = inckls['inckl_eta']
        trace['inckl_z'] = inckls['inckl_z'] # B-length vector

    if loss_required:
        trace['loss'] = torch.cat(trace['loss'], 0) # (1+apg_sweeps) * 1
    if ess_required:
        trace['ess_eta'] = torch.cat(trace['ess_eta'], 0) # apg_sweeps * B
        trace['ess_z'] = torch.cat(trace['ess_z'], 0) # apg_sweeps * B
    if mode_required:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0) # (1 + apg_sweeps) * B * K * D
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)  # (1 + apg_sweeps) * B * K * D
        trace['E_z'] = torch.cat(trace['E_z'], 0) # (1 + apg_sweeps) * B * N * K
    if density_required:
        trace['density'] = torch.cat(trace['density'], 0) # (1 + apg_sweeps) * B
    return trace

def rws(enc_rws_eta, enc_rws_z, generative, ob, trace, loss_required, ess_required, mode_required, density_required):
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
    w = F.softmax(log_p - log_q, 0).detach()
    if loss_required :
        loss = (w * (- log_q)).sum(0).mean() # 1-by-1 scalar
        trace['loss'].append(loss.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0)) # B-length tensor
        trace['ess_rws'].append(ess)
    if mode_required:
        E_tau = (q_eta['precisions'].dist.concentration / q_eta['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta['means'].dist.loc.mean(0).detach()
        E_z = q_z['zs'].dist.probs.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
    if density_required:
        log_joint = log_p.mean(0).detach()
        trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
    return w, tau, mu, z, trace

def apg_eta(enc_apg_eta, generative, ob, z, tau_old, mu_old, trace, loss_required, ess_required, mode_required, density_required):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=True) ## forward kernel
    p_f = generative.eta_prior(q=q_f)
    log_q_f = q_f['means'].log_prob.sum(-1) + q_f['precisions'].log_prob.sum(-1)
    log_p_f = p_f['means'].log_prob.sum(-1) + p_f['precisions'].log_prob.sum(-1)
    tau = q_f['precisions'].value
    mu = q_f['means'].value
    ll_f = generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f + log_p_f - log_q_f
    ## backward
    q_b = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    p_b = generative.eta_prior(q=q_b)
    log_q_b = q_b['means'].log_prob.sum(-1) + q_b['precisions'].log_prob.sum(-1)
    log_p_b = p_b['means'].log_prob.sum(-1) + p_b['precisions'].log_prob.sum(-1)
    ll_b = generative.log_prob(ob=ob, z=z, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b + log_p_b - log_q_b
    w = F.softmax(log_w_f - log_w_b, 0).detach()
    if loss_required:
        loss = (w * (- log_q_f)).sum(0).sum(-1).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0)).mean(-1)
        trace['ess_eta'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if mode_required:
        E_tau = (q_f['precisions'].dist.concentration / q_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_f['means'].dist.loc.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
    if density_required:
        trace['density'].append(log_p_f.sum(-1).mean(0).unsqueeze(0)) # 1-by-B-length vector
        return w, tau, mu, trace

def apg_z(enc_apg_z, generative, ob, tau, mu, z_old, trace, loss_required, ess_required, mode_required, density_required):
    """
    Given the current samples of global variable (eta = mu + tau),
    update local variable (state).
    """
    q_f = enc_apg_z(ob=ob, tau=tau, mu=mu, sampled=True)
    p_f = generative.z_prior(q=q_f)
    log_q_f = q_f['states'].log_prob
    log_p_f = p_f['states'].log_prob
    z = q_f['states'].value
    ll_f = generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=False)
    log_w_f = ll_f + log_p_f - log_q_f
    ## backward
    q_b = enc_apg_z(ob=ob, tau=tau, mu=mu, sampled=False, z_old=z_old)
    p_b = generative.z_prior(q=q_b)
    log_q_b = q_b['states'].log_prob
    log_p_b = p_b['states'].log_prob
    ll_b = generative.log_prob(ob=ob, z=z_old, tau=tau, mu=mu, aggregate=False)
    log_w_b = ll_b + log_p_b - log_q_b
    w = F.softmax(log_w_f - log_w_b, 0).detach()
    if loss_required:
        loss = (w * (- log_q_f)).sum(0).sum(-1).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0)).mean(-1)
        trace['ess_z'].append(ess.unsqueeze(0))
    if mode_required:
        E_z = q_f['states'].dist.probs.mean(0).detach()
        trace['E_z'].append(E_z.unsqueeze(0))
    if density_required:
        trace['density'][-1] = trace['density'][-1] + (ll_f + log_p_f).sum(-1).mean(0).unsqueeze(0)
    return w, z, trace
