import torch
import torch.nn.functional as F
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
def apg_objective(model, resampler, apg_sweeps, ob, K, loss_required=True, ess_required=True, mode_required=False, density_required=False):
    trace = dict()
    if loss_required:
        trace['loss_phi'] = []
        trace['loss_theta'] = []
    if ess_required:
        trace['ess_rws'] = []
        trace['ess_mu'] = []
        trace['ess_local'] = []

    if mode_required:
        trace['E_mu'] = []
        trace['E_z'] = []
        trace['E_recon'] = []

    if density_required:
        trace['density'] = []
#         trace['zs'] = []
#         trace['mus'] = []
#         trace['elbo'] = []
#         trace['log_qf'] = []
#         trace['log_pf'] =[]
#         trace['log_qb'] = []
#         trace['log_pb'] =[]
#         trace['ll_b'] = []
#         trace['log_prior_b'] =[]
#         trace['beta'] = []
    S, B, N, D = ob.shape

    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = model
    log_w, mu, z, beta, trace = rws(enc_rws_mu=enc_rws_mu,
                                    enc_rws_local=enc_apg_local,
                                    dec=dec,
                                    ob=ob,
                                    K=K,
                                    trace=trace,
                                    loss_required=loss_required,
                                    ess_required=ess_required,
                                    mode_required=mode_required,
                                    density_required=density_required)
    ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
    mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
    z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
    beta = resampler.resample_4dims(var=beta, ancestral_index=ancestral_index)
    for m in range(apg_sweeps):
        log_w, mu, z, beta, trace = apg_mu_local(enc_apg_mu=enc_apg_mu,
                                                    enc_apg_local=enc_apg_local,
                                                    dec=dec,
                                                    ob=ob,
                                                    z_old=z,
                                                    beta_old=beta,
                                                    mu_old=mu,
                                                    K=K,
                                                    trace=trace,
                                                    loss_required=loss_required,
                                                    ess_required=ess_required,
                                                    mode_required=mode_required,
                                                    density_required=density_required)

        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
        mu = resampler.resample_4dims(var=mu, ancestral_index=ancestral_index)
        z = resampler.resample_4dims(var=z, ancestral_index=ancestral_index)
        beta = resampler.resample_4dims(var=beta, ancestral_index=ancestral_index)


    if loss_required:
        trace['loss_phi'] = torch.cat(trace['loss_phi'], 0) # (1+apg_sweeps) * 1
        trace['loss_theta'] = torch.cat(trace['loss_theta'], 0) # (1+apg_sweeps) * 1
    if ess_required:
        trace['ess_rws'] = torch.cat(trace['ess_rws'], 0) # apg_sweeps * B
    if mode_required:
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)  # (1 + apg_sweeps) * B * K * D
        trace['E_z'] = torch.cat(trace['E_z'], 0) # (1 + apg_sweeps) * B * N * K
        trace['E_recon'] = torch.cat(trace['E_recon'], 0) # (1 + apg_sweeps) * B * N * K
    if density_required:
        trace['density'] = torch.cat(trace['density'], 0) # (1 + apg_sweeps) * B
    return trace

def rws(enc_rws_mu, enc_rws_local, dec, ob, K, trace, loss_required, ess_required, mode_required, density_required):
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
    w = F.softmax(log_w, 0).detach()
    if loss_required:
        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (- ll)).sum(0).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if ess_required:
        ess = (1. /(w**2).sum(0))
        trace['ess_rws'].append(ess.unsqueeze(0))
    if mode_required:
        E_mu =  q_mu['means'].dist.loc.mean(0).detach()
        E_mu_sigma = q_mu['means'].dist.scale.mean()
        E_z = q_local['states'].dist.probs.mean(0).detach()
        E_recon = p['likelihood'].dist.loc.mean(0).detach()
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
        trace['E_recon'].append(E_recon.unsqueeze(0))
    if density_required:
        log_joint  = log_p.mean(0).detach()
        trace['density'].append(log_joint.unsqueeze(0))
    return log_w, mu, z, beta, trace

def apg_mu_local(enc_apg_mu, enc_apg_local, dec, ob, z_old, beta_old, mu_old, K, trace, loss_required, ess_required, mode_required, density_required):
    """
    Given local variable {z, beta}, update global variables mu
    """
    q_f_mu = enc_apg_mu(ob=ob, z=z_old, beta=beta_old, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=True) ## forward kernel
    mu = q_f_mu['means'].value
    log_q_f_mu = q_f_mu['means'].log_prob.sum(-1).sum(-1) # S * B
    q_f_z = enc_apg_local(ob=ob, mu=mu, K=K, sampled=True)
    beta = q_f_z['angles'].value
    z = q_f_z['states'].value
    log_q_f_z = q_f_z['states'].log_prob.sum(-1) + q_f_z['angles'].log_prob.sum(-1).sum(-1)
    p_f = dec(ob=ob, mu=mu, z=z, beta=beta)
    ll_f = p_f['likelihood'].log_prob.sum(-1).sum(-1)

    log_p_f = ll_f + p_f['states'].log_prob.sum(-1) + p_f['angles'].log_prob.sum(-1).sum(-1) + p_f['means'].log_prob.sum(-1).sum(-1)
    log_q_f = log_q_f_z + log_q_f_mu
    log_w_f = log_p_f  - log_q_f
    ## backward
    q_b_z = enc_apg_local(ob=ob, mu=mu, K=K, sampled=False, z_old=z_old, beta_old=beta_old)
    log_q_b_z = q_b_z['states'].log_prob.sum(-1).detach() + q_b_z['angles'].log_prob.sum(-1).sum(-1).detach()

    # ll_b = p_b['likelihood'].log_prob.sum(-1).detach()
    q_b_mu = enc_apg_mu(ob=ob, z=z_old, beta=beta_old, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=False, mu_old=mu_old)
    log_q_b_mu = q_b_mu['means'].log_prob.sum(-1).sum(-1).detach()


    p_b = dec(ob=ob, mu=mu_old, z=z_old, beta=beta_old)
    log_p_b = p_b['likelihood'].log_prob.sum(-1).sum(-1).detach() + p_b['means'].log_prob.sum(-1).sum(-1) + p_b['states'].log_prob.sum(-1) + p_b['angles'].log_prob.sum(-1).sum(-1)
    log_q_b = log_q_b_z + log_q_b_mu
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    # w = F.softmax((log_w_f - log_w_b).sum(-1), 0).detach()
    if loss_required:
        loss_phi = (w * (- log_q_f)).sum(0).mean()
        loss_theta = (w * (- ll_f)).sum(0).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0))
        trace['ess_rws'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if mode_required:
        E_mu =  q_f['means'].dist.loc.mean(0).detach()
        trace['E_mu'].append(E_mu.unsqueeze(0))
    if density_required:
        trace['density'].append(log_p_f.mean(0).unsqueeze(0)) # 1-by-B-length vector
#         trace['log_qf'].append(log_q_f)
#         trace['log_pf'].append(log_p_f)
#         trace['log_qb'].append(log_q_b)
#         trace['log_pb'].append(log_p_b)
#         trace['ll_b'].append(ll_b)
#         trace['log_prior_b'].append(log_prior_b)
        return log_w, mu, z, beta, trace
