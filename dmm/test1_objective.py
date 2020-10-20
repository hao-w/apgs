import torch
import torch.nn.functional as F
"""
Test 1: Use samples and importance weights from rws to train both rws and Gibbs kernels
==========
sampling scheme:
1. start with 'one-shot' predicting mu and {z, angle}, compute importance weights w
2. iteratively take the gradient of Gibbs kernels only using samples and weights above
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
"""
def test1_objective(model, resample, apg_sweeps, ob, K, loss_required=True, ess_required=True, mode_required=False, density_required=False):
    trace = dict()
    if loss_required:
        trace['loss_phi'] = []
        trace['loss_theta'] = []
    if ess_required:
        trace['ess_rws'] = []

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
        w, mu, z, beta, trace = conditional_update(enc_apg_mu=enc_apg_mu,
                                                   enc_apg_local=enc_apg_local,
                                                   dec=dec,
                                                   ob=ob,
                                                   w=w,
                                                   mu=mu,
                                                   z=z,
                                                   beta=beta,
                                                   K=K,
                                                   trace=trace,
                                                   loss_required=loss_required,
                                                   ess_required=ess_required,
                                                   mode_required=mode_required,
                                                   density_required=density_required)
        mu = resample(var=mu, weights= w, dim_expand=False)
        z = resample(var=z, weights=w, dim_expand=False)
        beta = resample(var=beta, weights=w, dim_expand=False)

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
    w = F.softmax(log_p - log_q, 0).detach()
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
        E_z = q_local['states'].dist.probs.mean(0).detach()
        E_recon = p['likelihood'].dist.loc.mean(0).detach()
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
        trace['E_recon'].append(E_recon.unsqueeze(0))
    if density_required:
        log_joint  = log_p.mean(0).detach()
        trace['density'].append(log_joint.unsqueeze(0))
    return w, mu, z, beta, trace

def conditional_update(enc_apg_mu, enc_apg_local, dec, ob, w, mu, z, beta, K, trace, loss_required, ess_required, mode_required, density_required):
    q_mu = enc_apg_mu(ob=ob, z=z, beta=beta, K=K, priors=(dec.prior_mu_mu, dec.prior_mu_sigma), sampled=False, mu_old=mu) ## forward kernel
    q_local = enc_apg_local(ob=ob, mu=mu, K=K, sampled=False, z_old=z, beta_old=beta)
    p = dec(ob=ob, mu=mu, z=z, beta=beta)
    log_q = q_mu['means'].log_prob.sum(-1).sum(-1) + q_local['states'].log_prob.sum(-1) + q_local['angles'].log_prob.sum(-1).sum(-1)
    ll = p['likelihood'].log_prob.sum(-1).sum(-1)
    log_p = ll + p['means'].log_prob.sum(-1).sum(-1) + p['states'].log_prob.sum(-1) + p['angles'].log_prob.sum(-1).sum(-1)
    if loss_required:
        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (- ll)).sum(0).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if ess_required:
        ess = (1. / (w**2).sum(0))
        trace['ess_rws'].append(ess.unsqueeze(0)) # 1-by-B tensor
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
