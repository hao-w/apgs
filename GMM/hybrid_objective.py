import torch
import torch.nn.functional as F
from kls_gmm import kl_gmm, posterior_eta, posterior_z
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat

def hybrid_objective(model, flags, hmc, resampler, apg_sweeps, ob):
    densities = dict()
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = model
    log_w_rws, tau_rws, mu_rws, z_rws, log_joint_rws = rws(enc_rws_eta=enc_rws_eta,
                                                           enc_rws_z=enc_apg_z,
                                                           generative=generative,
                                                           ob=ob)
    if flags['apg']:
        print('Running APG updates..')
        trace_apg = {'density' : []}
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_rws)
        tau_apg = resampler.resample_4dims(var=tau_rws, ancestral_index=ancestral_index)
        mu_apg = resampler.resample_4dims(var=mu_rws, ancestral_index=ancestral_index)
        z_apg = resampler.resample_4dims(var=z_rws, ancestral_index=ancestral_index)
        for m in range(apg_sweeps):
            log_w_apg, tau_apg, mu_apg, trace_apg = apg_eta(enc_apg_eta=enc_apg_eta,
                                                            generative=generative,
                                                            ob=ob,
                                                            z=z_apg,
                                                            tau_old=tau_apg,
                                                            mu_old=mu_apg,
                                                            trace=trace_apg)
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_apg)
            tau_apg = resampler.resample_4dims(var=tau_apg, ancestral_index=ancestral_index)
            mu_apg = resampler.resample_4dims(var=mu_apg, ancestral_index=ancestral_index)
            z_apg = resampler.resample_4dims(var=z_apg, ancestral_index=ancestral_index)
            # trace['elbo'].append(log_w)
            log_w_apg, z_apg, trace_apg = apg_z(enc_apg_z=enc_apg_z,
                                                generative=generative,
                                                ob=ob,
                                                tau=tau_apg,
                                                mu=mu_apg,
                                                z_old=z_apg,
                                                trace=trace_apg)
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_apg)
            tau_apg = resampler.resample_4dims(var=tau_apg, ancestral_index=ancestral_index)
            mu_apg = resampler.resample_4dims(var=mu_apg, ancestral_index=ancestral_index)
            z_apg = resampler.resample_4dims(var=z_apg, ancestral_index=ancestral_index)

        densities['apg'] = torch.cat([log_joint_rws.unsqueeze(0)] + trace_apg['density'], 0) # (1 + apg_sweeps) * B

    if flags['gibbs']:
        trace_gibbs = {'density' : []}
        print('Running Gibbs+RWS updates..')
        for m in range(apg_sweeps):
            if m == 0:
                tau_gibbs, mu_gibbs, z_gibbs, trace_gibbs = gibbs_sweep(generative=generative,
                                                                            ob=ob,
                                                                            z=z_rws,
                                                                            trace=trace_gibbs)
            else:
                tau_gibbs, mu_gibbs, z_gibbs, trace_gibbs = gibbs_sweep(generative=generative,
                                                                        ob=ob,
                                                                        z=z_gibbs,
                                                                        trace=trace_gibbs)
        densities['gibbs'] = torch.cat([log_joint_rws.unsqueeze(0)] + trace_gibbs['density'], 0)

    if flags['hmc']:
        print('Running HMC+RWS updates..')

        _, _, density_list = hmc.hmc_sampling(ob=ob, log_tau=tau_rws.log(), mu=mu_rws)

        densities['hmc'] =  torch.cat([log_joint_rws.unsqueeze(0)]+density_list, 0)

    if flags['bpg']:
        print('Running Boostraped Population Gibbs updates..')
        trace_bpg = {'density' : []}
        factor = 10 ## increase the number of particles by 100x
        resampler_bpg = resampler
        resampler_bpg.S = resampler_bpg.S * factor
        log_w_bpg = log_w.repeat(factor, 1)
        tau_bpg = tau_rws.repeat(factor, 1, 1, 1)
        mu_bpg = mu_rws.repeat(factor, 1, 1, 1)
        z_bpg = z_rws.repeat(factor, 1, 1, 1)
        ob_bpg = ob.repeat(factor, 1, 1, 1)
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_rws)
        tau_bpg = resampler.resample_4dims(var=tau_rws, ancestral_index=ancestral_index)
        mu_bpg = resampler.resample_4dims(var=mu_rws, ancestral_index=ancestral_index)
        z_bpg = resampler.resample_4dims(var=z_rws, ancestral_index=ancestral_index)
        for m in range(apg_sweeps):
            log_w_bpg, tau_bpg, mu_bpg, trace_bpg = bpg_eta(generative=generative,
                                                                ob=ob,
                                                                z=z_bpg,
                                                                tau_old=tau_bpg,
                                                                mu_old=mu_bpg,
                                                                trace=trace_bpg)
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_bpg)
            tau_bpg = resampler.resample_4dims(var=tau_bpg, ancestral_index=ancestral_index)
            mu_bpg = resampler.resample_4dims(var=mu_bpg, ancestral_index=ancestral_index)
            z_bpg = resampler.resample_4dims(var=z_bpg, ancestral_index=ancestral_index)
            low_w_bpg, z_bpg, trace_bpg = apg_z(enc_apg_z=enc_apg_z,
                                                generative=generative,
                                                ob=ob,
                                                tau=tau_bpg,
                                                mu=mu_bpg,
                                                z_old=z_bpg,
                                                trace=trace_bpg)
            ancestral_index = resampler.sample_ancestral_index(log_weights=low_w_bpg)
            tau_bpg = resampler.resample_4dims(var=tau_bpg, ancestral_index=ancestral_index)
            mu_bpg = resampler.resample_4dims(var=mu_bpg, ancestral_index=ancestral_index)
            z_bpg = resampler.resample_4dims(var=z_bpg, ancestral_index=ancestral_index)
        densities['bpg'] = torch.cat([log_joint_rws.unsqueeze(0)]+ trace_bpg['density'], 0)

    return densities

def rws(enc_rws_eta, enc_rws_z, generative, ob):
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
    log_joint = log_p.detach()
    return log_w, tau, mu, z, log_joint

def apg_eta(enc_apg_eta, generative, ob, z, tau_old, mu_old, trace):
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
    trace['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, trace

def bpg_eta(generative, ob, z, tau_old, mu_old, trace):
    """
    Given local variable z, update global variables eta := {mu, tau}.
    """
    q_f = generative.eta_sample_prior(S=ob.shape[0], B=ob.shape[1])
    ## Not typo, here p is q since we sample from prior
    log_p_f = q_f['means'].log_prob.sum(-1).sum(-1) + q_f['precisions'].log_prob.sum(-1).sum(-1)
    # log_p_f = p_f['means'].log_prob.sum(-1).sum(-1) + p_f['precisions'].log_prob.sum(-1).sum(-1)
    tau = q_f['precisions'].value
    mu = q_f['means'].value
    ll_f = generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=True)
    log_w_f = ll_f
    ## backward
    # q_b = generative.eta_sample_prior(S=ob.shape[0], B=ob.shape[1], sampled=False, tau)
    # q_b = enc_apg_eta(ob=ob, z=z, prior_ng=generative.prior_ng, sampled=False, tau_old=tau_old, mu_old=mu_old)
    # p_b = generative.eta_prior(q=q_b)
    # log_q_b = q_b['means'].log_prob.sum(-1).sum(-1) + q_b['precisions'].log_prob.sum(-1).sum(-1)
    # log_p_b = p_b['means'].log_prob.sum(-1).sum(-1) + p_b['precisions'].log_prob.sum(-1).sum(-1)
    ll_b = generative.log_prob(ob=ob, z=z, tau=tau_old, mu=mu_old, aggregate=True)
    log_w_b = ll_b
    log_w = (log_w_f - log_w_b).detach()
    trace['density'].append(log_p_f.unsqueeze(0)) # 1-by-B-length vector
    return log_w, tau, mu, trace

def apg_z(enc_apg_z, generative, ob, tau, mu, z_old, trace):
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
    log_w_joint = (log_w_f - log_w_b).detach().sum(-1)
    trace['density'][-1] = trace['density'][-1] + (ll_f + log_p_f).sum(-1).unsqueeze(0)
        # trace['ll'].append(ll_f.sum(-1).unsqueeze(0))
    return log_w_joint, z, trace



def gibbs_sweep(generative, ob, z, trace):
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

    trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
    return tau, mu, z, trace
