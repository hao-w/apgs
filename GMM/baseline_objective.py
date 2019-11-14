import torch
import torch.nn.functional as F
from resampling import resample
"""
Reweighed Wake-Sleep objective in GMM problem
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
def rws_objective(model, ob, architecture, loss_required=True, ess_required=True, mode_required=False, density_required=False):
    trace = dict() ## a dictionary that tracks variables needed during the sweeping
    if loss_required:
        trace['loss'] = []
    if ess_required:
        trace['ess_rws'] = []
    if mode_required:
        trace['E_tau'] = []
        trace['E_mu'] = []
        trace['E_z'] = []
    if density_required:
        trace['density'] = []

    (enc_rws_eta, enc_rws_z, generative) = model
    w, tau, mu, z, trace = rws(enc_rws_eta=enc_rws_eta,
                               enc_rws_z=enc_rws_z,
                               generative=generative,
                               ob=ob,
                               trace=trace,
                               architecture=architecture,
                               loss_required=loss_required,
                               ess_required=ess_required,
                               mode_required=mode_required,
                               density_required=density_required)
    if loss_required:
        trace['loss'] = torch.cat(trace['loss'], 0) # (1+apg_sweeps) * 1
    if mode_required:
        trace['E_tau'] = torch.cat(trace['E_tau'], 0) # (1 + apg_sweeps) * B * K * D
        trace['E_mu'] = torch.cat(trace['E_mu'], 0)  # (1 + apg_sweeps) * B * K * D
        trace['E_z'] = torch.cat(trace['E_z'], 0) # (1 + apg_sweeps) * B * N * K
    if density_required:
        trace['density'] = torch.cat(trace['density'], 0) # (1 + apg_sweeps) * B
    return trace

def rws(enc_rws_eta, enc_rws_z, generative, ob, trace, architecture, loss_required, ess_required, mode_required, density_required):
    assert isinstance(architecture, str)
    if architecture == 'mlp':
        q_eta = enc_rws_eta(ob=ob, prior_ng=generative.prior_ng, sampled=True)
    elif architecture == 'lstm':
        q_eta = enc_rws_eta(ob=ob, K=generative.K, sampled=True)
    else:
        print('ERROR! unexpected architecture name, only mlp or lstm are available now.')
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
        E_z = q_z['states'].dist.probs.mean(0).detach()
        trace['E_tau'].append(E_tau.unsqueeze(0))
        trace['E_mu'].append(E_mu.unsqueeze(0))
        trace['E_z'].append(E_z.unsqueeze(0))
    if density_required:
        log_joint = log_p.mean(0).detach()
        trace['density'].append(log_joint.unsqueeze(0)) # 1-by-B-length vector
    return w, tau, mu, z, trace
