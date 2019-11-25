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
def apg_objective(model, resample, apg_sweeps, ob, K, loss_required=True, ess_required=True, mode_required=False, density_required=False):
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

    S, B, N, D = ob.shape
    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = model
    w, mu, z, beta, trace = rws(enc_rws_mu=enc_rws_mu,
                                enc_rws_local=enc_apg_local,
                                dec=dec,
                                ob=ob,
                                K=K,
                                trace=trace,
                                loss_required=loss_required,
                                ess_required=ess_required,
                                mode_required=mode_required,
                                density_required=density_required)
    mu = resample(var=mu, weights=w, dim_expand=False)
    z = resample(var=z, weights=w, dim_expand=False)
    beta = resample(var=beta, weights=w, dim_expand=False)
    for m in range(apg_sweeps):
        w, mu, trace = apg_mu(enc_apg_mu=enc_apg_mu,
                              dec=dec,
                              ob=ob,
                              z=z,
                              beta=beta,
                              mu_old=mu,
                              K=K,
                              trace=trace,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required,
                              density_required=density_required)
        mu = resample(var=mu, weights= w, dim_expand=True)
        w, z, beta, trace = apg_local(enc_apg_local=enc_apg_local,
                                      dec=dec,
                                      ob=ob,
                                      mu=mu,
                                      z_old=z,
                                      beta_old=beta,
                                      K=K,
                                      trace=trace,
                                      loss_required=loss_required,
                                      ess_required=ess_required,
                                      mode_required=mode_required,
                                      density_required=density_required)
        z = resample(var=z, weights=w, dim_expand=True)
        beta = resample(var=beta, weights=w, dim_expand=True)

    if loss_required:
        trace['loss_phi'] = torch.cat(trace['loss_phi'], 0) # (1+apg_sweeps) * 1
        trace['loss_theta'] = torch.cat(trace['loss_theta'], 0) # (1+apg_sweeps) * 1
    if ess_required:
        trace['ess_mu'] = torch.cat(trace['ess_mu'], 0) # apg_sweeps * B
        trace['ess_local'] = torch.cat(trace['ess_local'], 0) # apg_sweeps * B
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
    w = F.softmax(log_p - log_q, 0).detach()
    if loss_required:
        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (- ll)).sum(0).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if ess_required:
        ess = (1. /(w**2).sum(0))
        trace['ess_rws'].append(ess)
    if mode_required:
        E_mu =  q_mu['means'].dist.loc.mean(0).detach()
        E_z = q_local['states'].dist.probs.mean(0).detach()
        E_recon = p['likelihood'].dist.loc.mean(0).detach()
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
        trace['E_recon'].append(E_recon.unsqueeze(0))
    if density_required:
        log_joint  = log_p.mean(0).detach()
        trace['density'].append(log_joint.unsqueeze(0))
    return w, mu, z, beta, trace

def apg_mu(enc_apg_mu, dec, ob, z, beta, mu_old, K, trace, loss_required, ess_required, mode_required, density_required):
    """
    Given local variable {z, beta}, update global variables mu
    """
    q_f = enc_apg_mu(ob=ob, z=z, beta=beta, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=True) ## forward kernel
    mu = q_f['means'].value
    log_q_f = q_f['means'].log_prob.sum(-1)
    p_f = dec(ob=ob, mu=mu, z=z, beta=beta)
    ll_f = p_f['likelihood'].log_prob.sum(-1)
    ll_f_collapsed = torch.cat([((z.argmax(-1)==k).float() * ll_f).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_priors_f = p_f['means'].log_prob.sum(-1)
    log_p_f = log_priors_f + ll_f_collapsed
    log_w_f =  log_p_f - log_q_f
    ## backward
    q_b = enc_apg_mu(ob=ob, z=z, beta=beta, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=False, mu_old=mu_old)
    log_q_b = q_b['means'].log_prob.sum(-1).detach()
    p_b = dec(ob=ob, mu=mu_old, z=z, beta=beta)
    ll_b = p_b['likelihood'].log_prob.sum(-1).detach()
    ll_b_collapsed = torch.cat([((z.argmax(-1)==k).float() * ll_b).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_p_b = p_b['means'].log_prob.sum(-1) + ll_b_collapsed
    log_w_b =  log_p_b - log_q_b
    w = F.softmax(log_w_f - log_w_b, 0).detach()
    if loss_required:
        loss_phi = (w * (- log_q_f)).sum(0).sum(-1).mean()
        loss_theta = (w * (- ll_f_collapsed)).sum(0).sum(-1).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0)).mean(-1)
        trace['ess_mu'].append(ess.unsqueeze(0)) # 1-by-B tensor
    if mode_required:
        E_mu =  q_f['means'].dist.loc.mean(0).detach()
        trace['E_mu'].append(E_mu.unsqueeze(0))
    if density_required:
        trace['density'].append(log_priors_f.sum(-1).mean(0).unsqueeze(0)) # 1-by-B-length vector
        return w, mu, trace


def apg_local(enc_apg_local, dec, ob, mu, z_old, beta_old, K, trace, loss_required, ess_required, mode_required, density_required):
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
    w = F.softmax(log_w_f - log_w_b, 0).detach()
    if loss_required:
        loss_phi = (w * (- log_q_f)).sum(0).sum(-1).mean()
        loss_theta = (w * (- ll_f)).sum(0).sum(-1).mean()
        trace['loss_phi'][-1] = trace['loss_phi'][-1] + loss_phi.unsqueeze(0)
        trace['loss_theta'][-1] = trace['loss_theta'][-1] + loss_theta.unsqueeze(0)
    if ess_required:
        ess = (1. / (w**2).sum(0)).mean(-1)
        trace['ess_local'].append(ess.unsqueeze(0))
    if mode_required:
        E_z = q_f['states'].dist.probs.mean(0).detach()
        E_recon = p_f['likelihood'].dist.loc.mean(0).detach()
        trace['E_z'].append(E_z.unsqueeze(0))
        trace['E_recon'].append(E_recon.unsqueeze(0))
    if density_required:
        trace['density'][-1] = trace['density'][-1] + log_p_f.sum(-1).mean(0).unsqueeze(0)
    return w, z, beta, trace
