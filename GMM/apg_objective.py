import torch
import torch.nn.functional as F
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
"""
def apg_objective(model, resampler, block, apg_sweeps, ob, loss_required=True, ess_required=True, mode_required=False, density_required=False):
    trace = dict() ## a dictionary that tracks variables needed during the sweeping
    if loss_required:
        trace['loss'] = []
    if ess_required:
        trace['ess'] = []
    if mode_required:
        trace['E_tau'] = []
        trace['E_mu'] = []
        trace['E_z'] = []
    if density_required:
        trace['density'] = []

    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = model
    log_w, tau, mu, z, trace = rws(enc_rws_eta=enc_rws_eta,
                                   enc_rws_z=enc_apg_z,
                                   generative=generative,
                                   ob=ob,
                                   trace=trace,
                                   loss_required=loss_required,
                                   ess_required=ess_required,
                                   mode_required=mode_required,
                                   density_required=density_required)
    ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
    tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
    mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
    z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    if block == 'small':
        for m in range(apg_sweeps):
            log_w_eta, tau, mu, trace = apg_eta(enc_apg_eta=enc_apg_eta,
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
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_eta)
            tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
            mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
            z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
            log_w_z, z, trace = apg_z(enc_apg_z=enc_apg_z,
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
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_z)
            tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
            mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
            z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    elif block == 'large':
        for m in range(apg_sweeps):
            log_w, tau, mu, z, trace = apg_eta_z(enc_apg_eta=enc_apg_eta,
                                                enc_apg_z=enc_apg_z,
                                                generative=generative,
                                                ob=ob,
                                                z_old=z,
                                                tau_old=tau,
                                                mu_old=mu,
                                                trace=trace,
                                                loss_required=loss_required,
                                                ess_required=ess_required,
                                                mode_required=mode_required,
                                                density_required=density_required)
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
            tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
            mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
            z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    else:
        print('ERROR! Unexpected block strategy.')
        exit()

    if loss_required:
        trace['loss'] = torch.cat(trace['loss'], 0) # (1+apg_sweeps) * 1
    if ess_required:
        trace['ess'] = torch.cat(trace['ess'], 0)
    if mode_required:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0) # (1 + apg_sweeps) * B * K * D
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)  # (1 + apg_sweeps) * B * K * D
        trace['E_z'] = torch.cat(trace['E_z'], 0) # (1 + apg_sweeps) * B * N * K
    if density_required:
        trace['density'] = torch.cat(trace['density'], 0) # (1 + apg_sweeps) * S * B
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
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    if loss_required :
        loss = (w * (- log_q)).sum(0).mean() # 1-by-1 scalar
        trace['loss'].append(loss.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0)) # B-length tensor
        trace['ess'].append(ess.unsqueeze(0))
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
    return log_w, tau, mu, z, trace

def apg_eta_z(enc_apg_eta, enc_apg_z, generative, ob, z_old, tau_old, mu_old, trace, loss_required, ess_required, mode_required, density_required):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f_eta = enc_apg_eta(ob=ob, z=z_old, prior_ng=generative.prior_ng, sampled=True) ## forward kernel
    p_f_eta = generative.eta_prior(q=q_f_eta)
    log_q_f_eta = q_f_eta['means'].log_prob.sum(-1).sum(-1) + q_f_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_f_eta = p_f_eta['means'].log_prob.sum(-1).sum(-1) + p_f_eta['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f_eta['precisions'].value
    mu = q_f_eta['means'].value
    q_f_z = enc_apg_z(ob=ob, tau=tau, mu=mu, sampled=True)
    p_f_z = generative.z_prior(q=q_f_z)
    log_q_f_z = q_f_z['states'].log_prob.sum(-1)
    log_p_f_z = p_f_z['states'].log_prob.sum(-1)
    z = q_f_z['states'].value
    ll_f = generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f + log_p_f_eta - log_q_f_eta - log_q_f_z + log_p_f_z
    ## backward
    q_b_z = enc_apg_z(ob=ob, tau=tau, mu=mu, sampled=False, z_old=z_old)
    p_b_z = generative.z_prior(q=q_b_z)
    log_q_b_z = q_b_z['states'].log_prob.sum(-1)
    log_p_b_z = p_b_z['states'].log_prob.sum(-1)
    q_b_eta = enc_apg_eta(ob=ob, z=z_old, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    p_b_eta = generative.eta_prior(q=q_b_eta)
    log_q_b_eta = q_b_eta['means'].log_prob.sum(-1).sum(-1) + q_b_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_b_eta = p_b_eta['means'].log_prob.sum(-1).sum(-1) + p_b_eta['precisions'].log_prob.sum(-1).sum(-1)
    ll_b = generative.log_prob(ob=ob, z=z_old, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b + log_p_b_eta - log_q_b_eta + log_p_b_z - log_q_b_z
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if loss_required:
        loss = (w * (- log_q_f_eta - log_q_f_z)).sum(0).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if mode_required:
        E_tau = (q_f['precisions'].dist.concentration / q_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_f['means'].dist.loc.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        E_z = q_f['states'].dist.probs.mean(0).detach()
        trace['E_z'].append(z.unsqueeze(0))
    if density_required:
        log_joint = (ll_f + log_p_f_eta + log_p_f_z).detach()
        trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, z, trace


def apg_eta(enc_apg_eta, generative, ob, z, tau_old, mu_old, trace, loss_required, ess_required, mode_required, density_required):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=True) ## forward kernel
    p_f = generative.eta_prior(q=q_f)
    log_q_f = q_f['means'].log_prob.sum(-1).sum(-1) + q_f['precisions'].log_prob.sum(-1).sum(-1)
    log_p_f = p_f['means'].log_prob.sum(-1).sum(-1) + p_f['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f['precisions'].value
    mu = q_f['means'].value
    ll_f = generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f + log_p_f - log_q_f
    ## backward
    q_b = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    p_b = generative.eta_prior(q=q_b)
    log_q_b = q_b['means'].log_prob.sum(-1).sum(-1) + q_b['precisions'].log_prob.sum(-1).sum(-1)
    log_p_b = p_b['means'].log_prob.sum(-1).sum(-1) + p_b['precisions'].log_prob.sum(-1).sum(-1)
    ll_b = generative.log_prob(ob=ob, z=z, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b + log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if loss_required:
        loss = (w * (- log_q_f)).sum(0).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if mode_required:
        E_tau = (q_f['precisions'].dist.concentration / q_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_f['means'].dist.loc.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
    if density_required:
        trace['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, trace

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
    log_w_local = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w_local, 0).detach()
    log_w = log_w_local.sum(-1)
    if loss_required:
        loss = (w * (- log_q_f)).sum(0).sum(-1).mean()
        trace['loss'][-1] = trace['loss'][-1] + loss.unsqueeze(0)
    if mode_required:
        E_z = q_f['states'].dist.probs.mean(0).detach()
        trace['E_z'].append(E_z.unsqueeze(0))
    if density_required:
        trace['density'][-1] = trace['density'][-1] + (ll_f + log_p_f).sum(-1).unsqueeze(0)
    return log_w, z, trace
