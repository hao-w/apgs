import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.beta import Beta
import math

def apg_objective(models, x, K, result_flags, num_sweeps, resampler):
    """
    Amortized Population Gibbs objective in DGMM problem
    ==========
    abbreviations:
    K -- number of clusters
    D -- data dimensions (D=2 in NCMM)
    S -- sample size
    B -- batch size
    N -- number of data points in one (NCMM) dataset
    ==========
    variables:
    ob : S * B * N * D, observations, as data points
    mu: S * B * K * D, cluster means, as global variables
    z : S * B * N * K, cluster assignments, as local variables
    beta : S * B * N * 1 angle, as local variables
    local : {z, beta} is block of local variables
    ==========
    sampling scheme:
    1. start with 'one-shot' predicting mu, z and beta
    2. For m = 1 : apg_sweeps:
        update mu given z, beta
        resample
        update z and beta given mu
        resample
    ==========
    """
    trace = {'loss_phi' : [], 'loss_theta' : [], 'ess' : [], 'E_mu' : [], 'E_z' : [], 'E_recon' : [], 'density' : []}
    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = models
    log_w, mu, z, beta, trace = oneshot(enc_rws_mu, enc_apg_local, dec, x, K, trace, result_flags)
    mu, z, beta = resample_variables(resampler, mu, z, beta, log_weights=log_w)
    for m in range(num_sweeps-1):
        log_w_mu, mu, trace = apg_update_mu(enc_apg_mu, dec, x, z, beta, mu, K, trace, result_flags)
        mu, z, beta = resample_variables(resampler, mu, z, beta, log_weights=log_w_mu)
        log_w_z, z, beta, trace = apg_update_local(enc_apg_local, dec, x, mu, z, beta, K, trace, result_flags)
        mu, z, beta = resample_variables(resampler, mu, z, beta, log_weights=log_w_z)
    if result_flags['loss_required']:
        trace['loss_phi'] = torch.cat(trace['loss_phi'], 0) 
        trace['loss_theta'] = torch.cat(trace['loss_theta'], 0) 
    if result_flags['ess_required']:
        trace['ess'] = torch.cat(trace['ess'], 0) 
    if result_flags['mode_required']:
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)  
        trace['E_z'] = torch.cat(trace['E_z'], 0)
        trace['E_recon'] = torch.cat(trace['E_recon'], 0) 
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0) 
    return trace

def rws_objective(models, x, K, result_flags):
    """
    RWS objective 
    """
    trace = {'loss_phi' : [], 'loss_theta' : [], 'ess' : [], 'E_mu' : [], 'E_z' : [], 'E_recon' : [], 'density' : []}
    (enc_rws_mu, enc_rws_local, dec) = models
    log_w, mu, z, beta, trace = oneshot(enc_rws_mu, enc_apg_local, dec, x, K, trace, result_flags)
    mu, z, beta = resample_variables(resampler, mu, z, beta, log_weights=log_w)
    if result_flags['loss_required']:
        trace['loss_phi'] = torch.cat(trace['loss_phi'], 0) 
        trace['loss_theta'] = torch.cat(trace['loss_theta'], 0) 
    trace['loss'] = trace['loss_phi'].sum() + trace['loss_theta'][-1]
    if result_flags['ess_required']:
        trace['ess'] = torch.cat(trace['ess'], 0) 
    if result_flags['mode_required']:
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)  
        trace['E_z'] = torch.cat(trace['E_z'], 0)
        trace['E_recon'] = torch.cat(trace['E_recon'], 0) 
    if result_flags['density_required']:
        trace['density'] = torch.cat(trace['density'], 0) 
    return trace

def oneshot(enc_rws_mu, enc_rws_local, dec, x, K, trace, result_flags):
    """
    One-shot predicts mu, like a normal RWS
    """
    q_mu = enc_rws_mu(x, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=True)
    mu = q_mu['means'].value
    q_local = enc_rws_local(x, mu=mu, K=K, sampled=True)
    beta = q_local['angles'].value
    z = q_local['states'].value
    p = dec(x, mu=mu, z=z, beta=beta)
    log_q = q_mu['means'].log_prob.sum(-1).sum(-1) + q_local['states'].log_prob.sum(-1) + q_local['angles'].log_prob.sum(-1).sum(-1)
    ll = p['likelihood'].log_prob.sum(-1).sum(-1)
    log_p = ll + p['means'].log_prob.sum(-1).sum(-1) + p['states'].log_prob.sum(-1) + p['angles'].log_prob.sum(-1).sum(-1)
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (- ll)).sum(0).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. /(w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_mu =  q_mu['means'].dist.loc.mean(0).detach()
        E_mu_sigma = q_mu['means'].dist.scale.mean()
        E_z = q_local['states'].dist.probs.mean(0).detach()
        E_recon = p['likelihood'].dist.loc.mean(0).detach()
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
        trace['E_recon'].append(E_recon.unsqueeze(0))
    if result_flags['density_required']:
        log_joint  = log_p.detach()
        trace['density'].append(log_joint.unsqueeze(0))
    return log_w, mu, z, beta, trace

def apg_update_mu(enc_apg_mu, dec, x, z, beta, mu_old, K, trace, result_flags):
    """
    Given local variable {z, beta}, update global variables mu
    """
    q_f = enc_apg_mu(x, z=z, beta=beta, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=True) ## forward kernel
    mu = q_f['means'].value
    log_q_f = q_f['means'].log_prob.sum(-1).sum(-1) # S * B
    p_f = dec(x, mu=mu, z=z, beta=beta)
    ll_f = p_f['likelihood'].log_prob.sum(-1).sum(-1)
    log_priors_f = p_f['means'].log_prob.sum(-1).sum(-1)
    log_p_f = log_priors_f + ll_f
    log_w_f =  log_p_f - log_q_f
    ## backward
    q_b = enc_apg_mu(x, z=z, beta=beta, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=False, mu_old=mu_old)
    log_q_b = q_b['means'].log_prob.sum(-1).sum(-1).detach()
    p_b = dec(x, mu=mu_old, z=z, beta=beta)
    ll_b = p_b['likelihood'].log_prob.sum(-1).sum(-1).detach()
    log_prior_b = p_b['means'].log_prob.sum(-1).sum(-1)
    log_p_b = log_prior_b + ll_b
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss_phi = (w * (- log_q_f)).sum(0).mean()
        loss_theta = (w * (- ll_f)).sum(0).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        trace['ess'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if result_flags['mode_required']:
        E_mu =  q_f['means'].dist.loc.mean(0).detach()
        trace['E_mu'].append(E_mu.unsqueeze(0))
    if result_flags['density_required']:
        trace['density'].append(log_priors_f.unsqueeze(0))
    return log_w, mu, trace

def apg_update_local(enc_apg_local, dec, x, mu, z_old, beta_old, K, trace, result_flags):
    """
    Given the current samples of global variable mu
    update local variables {z, beta}
    """
    q_f = enc_apg_local(x, mu=mu, K=K, sampled=True)
    beta = q_f['angles'].value
    z = q_f['states'].value
    p_f = dec(x, mu=mu, z=z, beta=beta)
    log_q_f = q_f['states'].log_prob + q_f['angles'].log_prob.sum(-1)
    ll_f = p_f['likelihood'].log_prob.sum(-1)
    log_p_f = ll_f + p_f['states'].log_prob + p_f['angles'].log_prob.sum(-1)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_b = enc_apg_local(x, mu=mu, K=K, sampled=False, z_old=z_old, beta_old=beta_old)
    p_b = dec(x, mu=mu, z=z_old, beta=beta_old)
    log_q_b = q_b['states'].log_prob.detach() + q_b['angles'].log_prob.sum(-1).detach()
    ll_b = p_b['likelihood'].log_prob.sum(-1).detach()
    log_p_b = ll_b + p_b['states'].log_prob + p_b['angles'].log_prob.sum(-1)
    log_w_b = log_p_b - log_q_b
    log_w_local = (log_w_f - log_w_b).detach()
    log_w = log_w_local.sum(-1)
    w_local = F.softmax(log_w_local, 0).detach()
    if result_flags['loss_required']:
        loss_phi = (w_local * (- log_q_f)).sum(0).sum(-1).mean()
        loss_theta = (w_local * (- ll_f)).sum(0).sum(-1).mean()
        trace['loss_phi'][-1] = trace['loss_phi'][-1] + loss_phi.unsqueeze(0)  
        trace['loss_theta'][-1] = trace['loss_theta'][-1] + loss_theta.unsqueeze(0)  
    if result_flags['mode_required']:
        E_z = q_f['states'].dist.probs.mean(0).detach()
        E_recon = p_f['likelihood'].dist.loc.mean(0).detach()
        trace['E_z'].append(E_z.unsqueeze(0))
        trace['E_recon'].append(E_recon.unsqueeze(0))
    if result_flags['density_required']:
        trace['density'][-1] = trace['density'][-1] + log_p_f.sum(-1).unsqueeze(0)
    return log_w, z, beta, trace

def resample_variables(resampler, mu, z, beta, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
    z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    beta = resampler.resample_4dims(var=beta, ancestral_index=ancestral_index)
    return mu, z, beta


def hmc_objective(models, x, K, result_flags, hmc_sampler):
    """
    HMC objective
    """
    trace = {'density' : []} 
    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = models
    _, mu, z, beta, trace = oneshot(enc_rws_mu, enc_apg_local, dec, x, K, trace, result_flags)
    trace = hmc_sampler.hmc_sampling(x, mu, z, beta, trace)
    trace['density'] = torch.cat(trace['density'], 0)
    return trace

def bpg_objective(models, x, K, result_flags, num_sweeps, resampler):
    """
    bpg objective
    """
    trace = {'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = models
    log_w, mu, z, beta, trace = oneshot(enc_rws_mu, enc_apg_local, dec, x, K, trace, result_flags)
    mu, z, beta = resample_variables(resampler, mu, z, beta, log_weights=log_w)
    for m in range(num_sweeps-1):
        log_w_mu, mu, trace = bpg_update_mu(enc_apg_mu, dec, x, z, beta, mu, K, trace)
        mu, z, beta = resample_variables(resampler, mu, z, beta, log_weights=log_w_mu)
        log_w_z, z, beta, trace = apg_update_local(enc_apg_local, dec, x, mu, z, beta, K, trace, result_flags)
        mu, z, beta = resample_variables(resampler, mu, z, beta, log_weights=log_w_z)
    trace['density'] = torch.cat(trace['density'], 0) 
    return trace

def bpg_update_mu(enc_apg_mu, dec, x, z, beta, mu_old, K, trace):
    q = Normal(dec.prior_mu_mu, dec.prior_mu_sigma)
    S, B, K, D = mu_old.shape
    mu = q.sample((S, B, K, ))
    log_p = q.log_prob(mu).sum(-1).sum(-1)
    p_f = dec(x, mu=mu, z=z, beta=beta)
    ll_f = p_f['likelihood'].log_prob.sum(-1).sum(-1)
    p_b = dec(x, mu=mu_old, z=z, beta=beta)
    ll_b = p_b['likelihood'].log_prob.sum(-1).sum(-1).detach()
    log_w = (ll_f - ll_b).detach()
    trace['density'].append(log_p.unsqueeze(0)) # 1-by-B-length vector
    return log_w, mu, trace
