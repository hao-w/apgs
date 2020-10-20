import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat


def hybrid_objective(model, flags, hmc, resampler, resampler_bps, apg_sweeps, ob, K, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps):
    trace_apg = dict() ## a dictionary that tracks variables needed during the sweeping
    trace_apg['density'] = []

    trace_hmc = dict() ## a dictionary that tracks variables needed during the sweeping
    trace_hmc['density'] = []
    trace_hmc['marginal'] = []
    trace_bps = dict() ## a dictionary that tracks variables needed during the sweeping
    trace_bps['density'] = []
    density_dict = dict()

    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = model
    log_w, mu_rws, z_rws, beta_rws, log_joint = rws(enc_rws_mu=enc_rws_mu,
                                                    enc_rws_local=enc_apg_local,
                                                    dec=dec,
                                                    ob=ob,
                                                    K=K)
    if flags['hmc']:
        trace_hmc['density'].append(log_joint.unsqueeze(0))
        # print('Running HMC+Gibbs updates..')
        _, trace_hmc = hmc.hmc_sampling(ob=ob,
                                              mu=mu_rws,
                                              z=z_rws,
                                              beta=beta_rws,
                                              trace=trace_hmc,
                                              hmc_num_steps=hmc_num_steps,
                                              leapfrog_step_size=leapfrog_step_size,
                                              leapfrog_num_steps=leapfrog_num_steps)
        # density_dict['hmc'] =  torch.cat(trace_hmc['density'], 0)
        density_dict['hmc'] =  torch.cat(trace_hmc['density'], 0)
#         print(hmc.smallest_accept_ratio)

    if flags['bps']:
        factor = 1
        trace_bps['density'].append(log_joint.repeat(factor, 1).unsqueeze(0))
        # print('Running Boostraped Population Sampling updates..')
        ## increase the number of particles by 10x
        ob_bps = ob.repeat(factor, 1, 1, 1)
        log_w_bps = log_w.repeat(factor, 1)
        mu_bps = mu_rws.repeat(factor, 1, 1, 1)
        z_bps = z_rws.repeat(factor, 1, 1, 1)
        beta_bps = beta_rws.repeat(factor, 1, 1, 1)
        ancestral_index = resampler_bps.sample_ancestral_index(log_weights=log_w_bps)
        mu_bps = resampler_bps.resample_4dims(var=mu_bps, ancestral_index=ancestral_index)
        z_bps = resampler_bps.resample_4dims(var=z_bps, ancestral_index=ancestral_index)
        beta_bps = resampler_bps.resample_4dims(var=beta_bps, ancestral_index=ancestral_index)

        for m in range(apg_sweeps):
            log_w_bps, mu_bps, trace_bps = bps_mu(dec=dec,
                                                    ob=ob_bps,
                                                    z=z_bps,
                                                    beta=beta_bps,
                                                    mu_old=mu_bps,
                                                    K=K,
                                                    trace=trace_bps)
            ancestral_index = resampler_bps.sample_ancestral_index(log_weights=log_w_bps)
            mu_bps = resampler_bps.resample_4dims(var=mu_bps, ancestral_index=ancestral_index)
            z_bps = resampler_bps.resample_4dims(var=z_bps, ancestral_index=ancestral_index)
            beta_bps = resampler_bps.resample_4dims(var=beta_bps, ancestral_index=ancestral_index)

            log_w_bps, z_bps, beta_bps, trace_bps = apg_local(enc_apg_local=enc_apg_local,
                                                              dec=dec,
                                                              ob=ob_bps,
                                                              mu=mu_bps,
                                                              z_old=z_bps,
                                                              beta_old=beta_bps,
                                                              K=K,
                                                              trace=trace_bps)
            ancestral_index = resampler_bps.sample_ancestral_index(log_weights=log_w_bps)
            mu_bps = resampler_bps.resample_4dims(var=mu_bps, ancestral_index=ancestral_index)
            z_bps = resampler_bps.resample_4dims(var=z_bps, ancestral_index=ancestral_index)
            beta_bps = resampler_bps.resample_4dims(var=beta_bps, ancestral_index=ancestral_index)

        density_dict['bps'] = torch.cat(trace_bps['density'], 0)


    if flags['apg']:
        trace_apg['density'].append(log_joint.unsqueeze(0))
        # print('Running APG updates..')
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
        mu_apg = resampler.resample_4dims(var=mu_rws, ancestral_index=ancestral_index)
        z_apg = resampler.resample_4dims(var=z_rws, ancestral_index=ancestral_index)
        beta_apg = resampler.resample_4dims(var=beta_rws, ancestral_index=ancestral_index)
        for m in range(apg_sweeps):
            log_w_apg, mu_apg, trace_apg = apg_mu(enc_apg_mu=enc_apg_mu,
                                                    dec=dec,
                                                    ob=ob,
                                                    z=z_apg,
                                                    beta=beta_apg,
                                                    mu_old=mu_apg,
                                                    K=K,
                                                    trace=trace_apg)
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_apg)
            mu_apg = resampler.resample_4dims(var=mu_apg, ancestral_index=ancestral_index)
            z_apg = resampler.resample_4dims(var=z_apg, ancestral_index=ancestral_index)
            beta_apg = resampler.resample_4dims(var=beta_apg, ancestral_index=ancestral_index)

            log_w_apg, z_apg, beta_apg, trace_apg = apg_local(enc_apg_local=enc_apg_local,
                                                              dec=dec,
                                                              ob=ob,
                                                              mu=mu_apg,
                                                              z_old=z_apg,
                                                              beta_old=beta_apg,
                                                              K=K,
                                                              trace=trace_apg)

            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_apg)
            mu_apg = resampler.resample_4dims(var=mu_apg, ancestral_index=ancestral_index)
            z_apg = resampler.resample_4dims(var=z_apg, ancestral_index=ancestral_index)
            beta_apg = resampler.resample_4dims(var=beta_apg, ancestral_index=ancestral_index)
        density_dict['apg'] = torch.cat(trace_apg['density'], 0) # (1 + apg_sweeps) * B

    return density_dict

def rws(enc_rws_mu, enc_rws_local, dec, ob, K):
    """
    One-shot predicts mu, like a normal RWS
    """
    ## mu
    q_mu = enc_rws_mu(ob=ob, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=True)
    mu = q_mu['means'].value
    q_local = enc_rws_local(ob=ob, mu=mu, K=K, sampled=True)
    beta = q_local['angles'].value
    z = q_local['states'].value
    p = dec(ob=ob, mu=mu, z=z, beta=beta)
    log_q = q_mu['means'].log_prob.sum(-1).sum(-1) + q_local['states'].log_prob.sum(-1) + q_local['angles'].log_prob.sum(-1).sum(-1)
    ll = p['likelihood'].log_prob.sum(-1).sum(-1)
    log_p = ll + p['means'].log_prob.sum(-1).sum(-1) + p['states'].log_prob.sum(-1) + p['angles'].log_prob.sum(-1).sum(-1)
    log_w = (log_p - log_q).detach()

    return log_w, mu, z, beta, log_p.detach()

def apg_mu(enc_apg_mu, dec, ob, z, beta, mu_old, K, trace):
    """
    Given local variable {z, beta}, update global variables mu
    """
    q_f = enc_apg_mu(ob=ob, z=z, beta=beta, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=True) ## forward kernel
    mu = q_f['means'].value
    log_q_f = q_f['means'].log_prob.sum(-1).sum(-1) # S * B
    p_f = dec(ob=ob, mu=mu, z=z, beta=beta)
    ll_f = p_f['likelihood'].log_prob.sum(-1).sum(-1)
    # ll_f_collapsed = torch.cat([((z.argmax(-1)==k).float() * ll_f).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_priors_f = p_f['means'].log_prob.sum(-1).sum(-1)
    log_p_f = log_priors_f + ll_f
    log_w_f =  log_p_f - log_q_f
    ## backward
    q_b = enc_apg_mu(ob=ob, z=z, beta=beta, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=False, mu_old=mu_old)
    log_q_b = q_b['means'].log_prob.sum(-1).sum(-1).detach()
    p_b = dec(ob=ob, mu=mu_old, z=z, beta=beta)
    ll_b = p_b['likelihood'].log_prob.sum(-1).sum(-1).detach()
    # ll_b_collapsed = torch.cat([((z.argmax(-1)==k).float() * ll_b).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_prior_b = p_b['means'].log_prob.sum(-1).sum(-1)
    log_p_b =  log_prior_b + ll_b
    log_w_b =  log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()

    trace['density'].append(log_priors_f.unsqueeze(0)) # 1-by-B-length vector

    return log_w, mu, trace

def bps_mu(dec, ob, z, beta, mu_old, K, trace):
    """
    Given local variable {z, beta}, update global variables mu
    """
    q = Normal(dec.prior_mu_mu, dec.prior_mu_sigma)
    S, B, K, D = mu_old.shape
    mu = q.sample((S, B, K, ))
    log_p = q.log_prob(mu).sum(-1).sum(-1)
    p_f = dec(ob=ob, mu=mu, z=z, beta=beta)
    ll_f = p_f['likelihood'].log_prob.sum(-1).sum(-1)
    log_w_f =  ll_f
    ## backward
    p_b = dec(ob=ob, mu=mu_old, z=z, beta=beta)
    ll_b = p_b['likelihood'].log_prob.sum(-1).sum(-1).detach()
    log_w_b =  ll_b
    log_w = (log_w_f - log_w_b).detach()

    trace['density'].append(log_p.unsqueeze(0)) # 1-by-B-length vector

    return log_w, mu, trace

def apg_local(enc_apg_local, dec, ob, mu, z_old, beta_old, K, trace):
    """
    Given the current samples of global variable mu
    update local variables {z, beta}
    """
    q_f = enc_apg_local(ob=ob, mu=mu, K=K, sampled=True)
    beta = q_f['angles'].value
    z = q_f['states'].value
    p_f = dec(ob=ob, mu=mu, z=z, beta=beta)
    log_q_f = q_f['states'].log_prob + q_f['angles'].log_prob.sum(-1)
    ll_f = p_f['likelihood'].log_prob.sum(-1)
    log_p_f = ll_f + p_f['states'].log_prob + p_f['angles'].log_prob.sum(-1)
    log_w_f = log_p_f - log_q_f
    ## backward
    q_b = enc_apg_local(ob=ob, mu=mu, K=K, sampled=False, z_old=z_old, beta_old=beta_old)
    p_b = dec(ob=ob, mu=mu, z=z_old, beta=beta_old)
    log_q_b = q_b['states'].log_prob.detach() + q_b['angles'].log_prob.sum(-1).detach()
    ll_b = p_b['likelihood'].log_prob.sum(-1).detach()
    log_p_b = ll_b + p_b['states'].log_prob + p_b['angles'].log_prob.sum(-1)
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).sum(-1).detach()
    trace['density'][-1] = trace['density'][-1] + log_p_f.sum(-1).unsqueeze(0)
    return log_w, z, beta, trace
