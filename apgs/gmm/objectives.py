import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from apgs.gmm.kls_gmm import kls_eta, posterior_eta, posterior_z

def apg_objective(models, x, result_flags, num_sweeps, block, resampler):
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
    trace = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    log_w, tau, mu, z, trace = oneshot(enc_rws_eta, enc_apg_z, generative, x, trace, result_flags)
    tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w)
    for m in range(num_sweeps-1):
        if block == 'decomposed':
            log_w_eta, tau, mu, trace = apg_update_eta(enc_apg_eta, generative, x, z, tau, mu, trace, result_flags)       
            tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_eta)
            log_w_z, z, trace = apg_update_z(enc_apg_z, generative, x, tau, mu, z, trace, result_flags)
            tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_z)
        elif block == 'joint':
            log_w, tau, mu, z, trace = apg_update_joint(enc_apg_z, enc_apg_eta, generative, x, z, tau, mu, trace, result_flags)
            tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w)
        else:
            raise ValueError
    if result_flags['loss_required']:
        trace['loss'] = torch.cat(trace['loss'], 0)
    if result_flags['ess_required']:
        trace['ess'] = torch.cat(trace['ess'], 0)
    if result_flags['mode_required']:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0) 
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)
        trace['E_z'] = torch.cat(trace['E_z'], 0)  # (num_sweeps) * B * N * K
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0)  # (num_sweeps) * S * B
    return trace

def rws_objective(models, x, result_flags):
    """
    The objective of RWS method
    """
    trace = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} 
    (enc_rws_eta, enc_rws_z, generative) = models
    w, tau, mu, z, trace = oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags)
    if result_flags['loss_required']:
        trace['loss'] = torch.cat(trace['loss'], 0)
    if result_flags['ess_required']:
        trace['ess'] = torch.cat(trace['ess'], 0)
    if result_flags['mode_required']:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0)
        trace['E_mu'] = torch.cat(trace['E_mu'], 0) 
        trace['E_z'] = torch.cat(trace['E_z'], 0) 
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0) 
    return trace

def oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags):
    """
    One-shot for eta and z, like a normal RWS
    """
    q_eta = enc_rws_eta(x, prior_ng=generative.prior_ng, sampled=True)
    p_eta = generative.eta_prior(q=q_eta)
    log_q_eta = q_eta['means'].log_prob.sum(-1).sum(-1) + q_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_eta = p_eta['means'].log_prob.sum(-1).sum(-1) + p_eta['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_eta['precisions'].value
    mu = q_eta['means'].value
    q_z = enc_rws_z(x, tau=tau, mu=mu, sampled=True)
    p_z = generative.z_prior(q=q_z)
    log_q_z = q_z['states'].log_prob.sum(-1)
    log_p_z = p_z['states'].log_prob.sum(-1) 
    z = q_z['states'].value 
    ll = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_p = ll + log_p_eta + log_p_z
    log_q =  log_q_eta + log_q_z
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q)).sum(0).mean() 
        trace['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0)) 
        trace['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_tau = (q_eta['precisions'].dist.concentration / q_eta['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta['means'].dist.loc.mean(0).detach()
        E_z = q_z['states'].dist.probs.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
    if result_flags['density_required']:
        log_joint = log_p.detach()
        trace['density'].append(log_joint.unsqueeze(0)) 
    return log_w, tau, mu, z, trace

def apg_update_joint(enc_apg_z, enc_apg_eta, generative, x, z_old, tau_old, mu_old, trace, result_flags):
    """
    Jointly update all the variables
    """
    q_f_eta = enc_apg_eta(x, z=z_old, prior_ng=generative.prior_ng, sampled=True) 
    p_f_eta = generative.eta_prior(q=q_f_eta)
    log_q_f_eta = q_f_eta['means'].log_prob.sum(-1).sum(-1) + q_f_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_f_eta = p_f_eta['means'].log_prob.sum(-1).sum(-1) + p_f_eta['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f_eta['precisions'].value
    mu = q_f_eta['means'].value
    q_f_z = enc_apg_z(x, tau=tau, mu=mu, sampled=True)
    p_f_z = generative.z_prior(q=q_f_z)
    log_q_f_z = q_f_z['states'].log_prob.sum(-1)
    log_p_f_z = p_f_z['states'].log_prob.sum(-1)
    z = q_f_z['states'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f + log_p_f_eta - log_q_f_eta - log_q_f_z + log_p_f_z
    ## backward
    q_b_z = enc_apg_z(x, tau=tau, mu=mu, sampled=False, z_old=z_old)
    p_b_z = generative.z_prior(q=q_b_z)
    log_q_b_z = q_b_z['states'].log_prob.sum(-1)
    log_p_b_z = p_b_z['states'].log_prob.sum(-1)
    q_b_eta = enc_apg_eta(x, z=z_old, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    p_b_eta = generative.eta_prior(q=q_b_eta)
    log_q_b_eta = q_b_eta['means'].log_prob.sum(-1).sum(-1) + q_b_eta['precisions'].log_prob.sum(-1).sum(-1)
    log_p_b_eta = p_b_eta['means'].log_prob.sum(-1).sum(-1) + p_b_eta['precisions'].log_prob.sum(-1).sum(-1)
    ll_b = generative.log_prob(x, z=z_old, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b + log_p_b_eta - log_q_b_eta + log_p_b_z - log_q_b_z
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f_eta - log_q_f_z)).sum(0).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0)) 
    if result_flags['mode_required']:
        E_tau = (q_f['precisions'].dist.concentration / q_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_f['means'].dist.loc.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        E_z = q_f['states'].dist.probs.mean(0).detach()
        trace['E_z'].append(z.unsqueeze(0))
    if result_flags['density_required']:
        log_joint = (ll_f + log_p_f_eta + log_p_f_z).detach()
        trace['density'].append(log_joint.unsqueeze(0))
    return log_w, tau, mu, z, trace


def apg_update_eta(enc_apg_eta, generative, x, z, tau_old, mu_old, trace, result_flags):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f = enc_apg_eta(x, z=z, prior_ng=generative.prior_ng, sampled=True) ## forward kernel
    p_f = generative.eta_prior(q=q_f)
    log_q_f = q_f['means'].log_prob.sum(-1).sum(-1) + q_f['precisions'].log_prob.sum(-1).sum(-1)
    log_p_f = p_f['means'].log_prob.sum(-1).sum(-1) + p_f['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f['precisions'].value
    mu = q_f['means'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f + log_p_f - log_q_f
    ## backward
    q_b = enc_apg_eta(x, z=z, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    p_b = generative.eta_prior(q=q_b)
    log_q_b = q_b['means'].log_prob.sum(-1).sum(-1) + q_b['precisions'].log_prob.sum(-1).sum(-1)
    log_p_b = p_b['means'].log_prob.sum(-1).sum(-1) + p_b['precisions'].log_prob.sum(-1).sum(-1)
    ll_b = generative.log_prob(x, z=z, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b + log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f)).sum(0).mean()
        trace['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if result_flags['mode_required']:
        E_tau = (q_f['precisions'].dist.concentration / q_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_f['means'].dist.loc.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
    if result_flags['density_required']:
        trace['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, trace

def apg_update_z(enc_apg_z, generative, x, tau, mu, z_old, trace, result_flags):
    """
    Given the current samples of global variable (eta = mu + tau),
    update local variable state i.e. z
    """
    q_f = enc_apg_z(x, tau=tau, mu=mu, sampled=True)
    p_f = generative.z_prior(q=q_f)
    log_q_f = q_f['states'].log_prob
    log_p_f = p_f['states'].log_prob
    z = q_f['states'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=False)
    log_w_f = ll_f + log_p_f - log_q_f
    ## backward
    q_b = enc_apg_z(x, tau=tau, mu=mu, sampled=False, z_old=z_old)
    p_b = generative.z_prior(q=q_b)
    log_q_b = q_b['states'].log_prob
    log_p_b = p_b['states'].log_prob
    ll_b = generative.log_prob(x, z=z_old, tau=tau, mu=mu, aggregate=False)
    log_w_b = ll_b + log_p_b - log_q_b
    log_w_local = (log_w_f - log_w_b).detach()
    w_local = F.softmax(log_w_local, 0).detach()
    log_w = log_w_local.sum(-1)
    if result_flags['loss_required']:
        loss = (w_local * (- log_q_f)).sum(0).sum(-1).mean()
        trace['loss'][-1] = trace['loss'][-1] + loss.unsqueeze(0)
    if result_flags['mode_required']:
        E_z = q_f['states'].dist.probs.mean(0).detach()
        trace['E_z'].append(E_z.unsqueeze(0))
    if result_flags['density_required']:
        trace['density'][-1] = trace['density'][-1] + (ll_f + log_p_f).sum(-1).unsqueeze(0)
    return log_w, z, trace

def resample_variables(resampler, tau, mu, z, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
    mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
    z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    return tau, mu, z


def gibbs_objective(models, x, result_flags, num_sweeps):
    """
    The Gibbs sampler objective 
    """
    trace = {'density' : []} 
    (enc_rws_eta, enc_rws_z, _, generative) = models
    _, tau, mu, z, trace = oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags)
    for m in range(num_sweeps-1):
        tau, mu, z, trace = gibbs_sweep(generative, x, z, trace)
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0)  # (num_sweeps) * S * B
    return trace

def gibbs_sweep(generative, x, z, trace):
    """
    Gibbs updates
    """
    post_alpha, post_beta, post_mu, post_nu = posterior_eta(x,
                                                            z=z,
                                                            prior_alpha=generative.prior_alpha,
                                                            prior_beta=generative.prior_beta,
                                                            prior_mu=generative.prior_mu,
                                                            prior_nu=generative.prior_nu)

    E_tau = (post_alpha / post_beta).mean(0)
    E_mu = post_mu.mean(0)
    tau = Gamma(post_alpha, post_beta).sample()
    mu = Normal(post_mu, 1. / (post_nu * tau).sqrt()).sample()
    posterior_logits = posterior_z(x, tau, mu, generative.prior_pi)
    E_z = posterior_logits.exp().mean(0)
    z = cat(logits=posterior_logits).sample()
    ll = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    log_prior_tau = Gamma(generative.prior_alpha, generative.prior_beta).log_prob(tau).sum(-1).sum(-1)
    log_prior_mu = Normal(generative.prior_mu, 1. / (generative.prior_nu * tau).sqrt()).log_prob(mu).sum(-1).sum(-1)
    log_prior_z = cat(probs=generative.prior_pi).log_prob(z).sum(-1)
    log_joint = ll + log_prior_tau + log_prior_mu + log_prior_z
    trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
    return tau, mu, z, trace

def hmc_objective(models, x, result_flags, hmc_sampler, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps):
    """
    HMC + marginalization over discrete variables in GMM problem
    """
    trace = {'density' : []} 
    (enc_rws_eta, enc_rws_z, _, generative) = models
    _, tau, mu, z, trace = oneshot(enc_rws_eta, enc_rws_z, generative, x, trace, result_flags)
    log_tau, mu, trace = hmc_sampler.hmc_sampling(generative,
                                                  x,
                                                  log_tau=tau.log(),
                                                  mu=mu,
                                                  trace=trace,
                                                  hmc_num_steps=hmc_num_steps,
                                                  leapfrog_step_size=leapfrog_step_size,
                                                  leapfrog_num_steps=leapfrog_num_steps)
    trace['density'] = torch.cat(trace['density'], 0)
    return log_tau.exp(), mu, trace

def bpg_objective(models, x, result_flags, num_sweeps, resampler):
    """
    bpg objective
    """
    trace = {'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    log_w, tau, mu, z, trace = oneshot(enc_rws_eta, enc_apg_z, generative, x, trace, result_flags)
    tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w)
    for m in range(num_sweeps-1):
        log_w_eta, tau, mu, trace = bpg_update_eta(generative, x, z, tau, mu, trace)
        tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_eta)
        log_w_z, z, trace = apg_update_z(enc_apg_z, generative, x, tau, mu, z, trace, result_flags)
        tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_z)
    trace['density'] = torch.cat(trace['density'], 0)  # (num_sweeps) * S * B
    return trace

def bpg_update_eta(generative, x, z, tau_old, mu_old, trace):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f = generative.eta_sample_prior(S=x.shape[0], B=x.shape[1])
    ## Not typo, here p is q since we sample from prior
    log_p_f = q_f['means'].log_prob.sum(-1).sum(-1) + q_f['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f['precisions'].value
    mu = q_f['means'].value
    ll_f = generative.log_prob(x, z=z, tau=tau, mu=mu, aggregate=True)
    ll_b = generative.log_prob(x, z=z, tau=tau_old, mu=mu_old, aggregate=True)
    log_w = (ll_f - ll_b).detach()
    trace['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, trace