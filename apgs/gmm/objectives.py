import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from apgs.gmm.kls_gmm import kls_eta, posterior_eta, posterior_z
import probtorch

def apg_objective(models, x, result_flags, num_sweeps, resampler):
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
    metrics = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    log_w, q_eta_z, metrics = oneshot(enc_rws_eta, enc_apg_z, generative, x, metrics, result_flags)
    q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w)
    for m in range(num_sweeps-1):
            log_w_eta, q_eta_z, metrics = apg_update_eta(enc_apg_eta, generative, q_eta_z, x, metrics, result_flags)       
            q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_eta)
            log_w_z, q_eta_z, metrics = apg_update_z(enc_apg_z, generative, q_eta_z, x, metrics, result_flags)
            q_eta_z = resample_variables(resampler, q_eta_z, log_weights=log_w_z)
    if result_flags['loss_required']:
        metrics['loss'] = torch.cat(metrics['loss'], 0)
    if result_flags['ess_required']:
        metrics['ess'] = torch.cat(metrics['ess'], 0)
    if result_flags['mode_required']:
        metrics['E_tau'] = torch.cat(metrics['E_tau'], 0) 
        metrics['E_mu'] = torch.cat(metrics['E_mu'], 0)
        metrics['E_z'] = torch.cat(metrics['E_z'], 0)  # (num_sweeps) * B * N * K
    if result_flags['density_required']:
        metrics['density'] = torch.cat(metrics['density'], 0)  # (num_sweeps) * S * B
    return metrics

def rws_objective(models, x, result_flags, num_sweeps):
    """
    The objective of RWS method
    """
    metrics = {'loss' : [], 'ess' : [], 'E_tau' : [], 'E_mu' : [], 'E_z' : [], 'density' : []} 
    (enc_rws_eta, enc_rws_z, generative) = models
    log_w, q_eta_z, metrics = oneshot(enc_rws_eta, enc_rws_z, generative, x, metrics, result_flags)
    if result_flags['loss_required']:
        metrics['loss'] = torch.cat(metrics['loss'], 0)
    if result_flags['ess_required']:
        metrics['ess'] = torch.cat(metrics['ess'], 0)
    if result_flags['mode_required']:
        metrics['E_tau'] = torch.cat(metrics['E_tau'], 0)
        metrics['E_mu'] = torch.cat(metrics['E_mu'], 0) 
        metrics['E_z'] = torch.cat(metrics['E_z'], 0) 
    if result_flags['density_required']:
        metrics['density'] = torch.cat(metrics['density'], 0) 
    return metrics

def oneshot(enc_rws_eta, enc_rws_z, generative, x, metrics, result_flags):
    """
    One-shot for eta and z, like a normal RWS
    """
    q_eta_z = enc_rws_eta(x, prior_ng=generative.prior_ng)
    q_eta_z = enc_rws_z(q_eta_z, x)
    p = generative.forward(q_eta_z, x)
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) ## it is annoying to repeat these same arguments every time I call .log_joint
    log_q = q_eta_z.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) 
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q)).sum(0).mean() 
        metrics['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0)) 
        metrics['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_tau = (q_eta_z['precisions'].dist.concentration / q_eta_z['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta_z['means'].dist.loc.mean(0).detach()
        E_z = q_eta_z['states'].dist.probs.mean(0).detach()
        metrics['E_tau'].append(E_tau.unsqueeze(0))
        metrics['E_mu'].append(E_mu.unsqueeze(0))
        metrics['E_z'].append(E_z.unsqueeze(0))
    if result_flags['density_required']:
        log_joint = log_p.detach()
        metrics['density'].append(log_joint.unsqueeze(0)) 
    return log_w, q_eta_z, metrics


def apg_update_eta(enc_apg_eta, generative, q_eta_z, x, metrics, result_flags):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    # forward
    q_eta_z_f = enc_apg_eta(q_eta_z, x, prior_ng=generative.prior_ng) ## forward kernel
    p_f = generative.forward(q_eta_z_f, x)
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_eta(q_eta_z, x, prior_ng=generative.prior_ng)
    p_b = generative.forward(q_eta_z_b, x)
    log_q_b = q_eta_z_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f)).sum(0).mean()
        metrics['loss'].append(loss.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        metrics['ess'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if result_flags['mode_required']:
        E_tau = (q_eta_z_f['precisions'].dist.concentration / q_eta_z_f['precisions'].dist.rate).mean(0).detach()
        E_mu = q_eta_z_f['means'].dist.loc.mean(0).detach()
        metrics['E_tau'].append(E_tau.unsqueeze(0))
        metrics['E_mu'].append(E_mu.unsqueeze(0))
    if result_flags['density_required']:
        metrics['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, q_eta_z_f, metrics

def apg_update_z(enc_apg_z, generative, q_eta_z, x, metrics, result_flags):
    """
    Given the current samples of global variable (eta = mu + tau),
    update local variable state i.e. z
    """
    # forward
    q_eta_z_f = enc_apg_z(q_eta_z, x)
    p_f = generative.forward(q_eta_z_f, x)
    log_q_f = q_eta_z_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_eta_z_b = enc_apg_z(q_eta_z, x)
    p_b = generative.forward(q_eta_z_b, x)
    log_q_b = q_eta_z_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss = (w * (- log_q_f)).sum(0).sum(-1).mean()
        metrics['loss'][-1] = metrics['loss'][-1] + loss.unsqueeze(0)
    if result_flags['mode_required']:
        E_z = q_eta_z_f['states'].dist.probs.mean(0).detach()
        metrics['E_z'].append(E_z.unsqueeze(0))
    if result_flags['density_required']:
        metrics['density'].append(log_p_f.unsqueeze(0))
    return log_w, q_eta_z_f, metrics


def resample_variables(resampler, q, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    tau = q['precisions'].value
    tau_concentration = q['precisions'].dist.concentration
    tau_rate = q['precisions'].dist.rate
    mu = q['means'].value
    mu_loc = q['means'].dist.loc
    mu_scale = q['means'].dist.scale
    z = q['states'].value
    z_probs = q['states'].dist.probs
    tau = resampler.resample_4dims(var=tau, ancestral_index=ancestral_index)
    tau_concentration = resampler.resample_4dims(var=tau_concentration, ancestral_index=ancestral_index)
    tau_rate = resampler.resample_4dims(var=tau_rate, ancestral_index=ancestral_index)
    mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
    mu_loc = resampler.resample_4dims(var=mu_loc, ancestral_index=ancestral_index)
    mu_scale = resampler.resample_4dims(var=mu_scale, ancestral_index=ancestral_index)
    z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    z_probs = resampler.resample_4dims(var=z_probs, ancestral_index=ancestral_index)
    q_resampled = probtorch.Trace()
    q_resampled.gamma(tau_concentration,
                      tau_rate,
                      value=tau,
                      name='precisions')
    q_resampled.normal(mu_loc,
                       mu_scale,
                       value=mu,
                       name='means')
    _ = q_resampled.variable(cat, probs=z_probs, value=z, name='states')
    return q_resampled


def gibbs_objective(models, x, result_flags, num_sweeps):
    """
    The Gibbs sampler objective 
    """
    metrics = {'density' : []} 
    (enc_rws_eta, enc_rws_z, _, generative) = models
    _, tau, mu, z, metrics = oneshot(enc_rws_eta, enc_rws_z, generative, x, metrics, result_flags)
    for m in range(num_sweeps-1):
        tau, mu, z, metrics = gibbs_sweep(generative, x, z, metrics)
    if result_flags['density_required']:
        metrics['density'] = torch.cat(metrics['density'], 0)  # (num_sweeps) * S * B
    return metrics

def gibbs_sweep(generative, x, z, metrics):
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
    metrics['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
    return tau, mu, z, metrics

def hmc_objective(models, x, result_flags, hmc_sampler):
    """
    HMC + marginalization over discrete variables in GMM problem
    """
    metrics = {'density' : []} 
    (enc_rws_eta, enc_rws_z, _, generative) = models
    _, tau, mu, z, metrics = oneshot(enc_rws_eta, enc_rws_z, generative, x, metrics, result_flags)
    log_tau, mu, metrics = hmc_sampler.hmc_sampling(generative,
                                                  x,
                                                  log_tau=tau.log(),
                                                  mu=mu,
                                                  metrics=metrics)
    metrics['density'] = torch.cat(metrics['density'], 0)
    return log_tau.exp(), mu, metrics

def bpg_objective(models, x, result_flags, num_sweeps, resampler):
    """
    bpg objective
    """
    metrics = {'density' : []} ## a dictionary that tracks things needed during the sweeping
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    log_w, tau, mu, z, metrics = oneshot(enc_rws_eta, enc_apg_z, generative, x, metrics, result_flags)
    tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w)
    for m in range(num_sweeps-1):
        log_w_eta, tau, mu, metrics = bpg_update_eta(generative, x, z, tau, mu, metrics)
        tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_eta)
        log_w_z, z, metrics = apg_update_z(enc_apg_z, generative, x, tau, mu, z, metrics, result_flags)
        tau, mu, z = resample_variables(resampler, tau, mu, z, log_weights=log_w_z)
    metrics['density'] = torch.cat(metrics['density'], 0)  # (num_sweeps) * S * B
    return metrics

def bpg_update_eta(generative, x, z, tau_old, mu_old, metrics):
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
    metrics['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, metrics