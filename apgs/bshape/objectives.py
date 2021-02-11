import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import probtorch

def resample_variables(resampler, q, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    q_new = probtorch.Trace()
    for key, node in q.items():
        resampled_loc = resampler.resample_4dims(var=node.dist.loc, ancestral_index=ancestral_index)
        resampled_scale = resampler.resample_4dims(var=node.dist.scale, ancestral_index=ancestral_index)
        resampled_value = resampler.resample_4dims(var=node.value, ancestral_index=ancestral_index)
        q_new.normal(loc=resampled_loc, scale=resampled_scale, value=resampled_value, name=key)
    return q_new
        
def apg_objective(models, AT, frames, K, result_flags, num_sweeps, resampler, mean_shape):
    """
    Amortized Population Gibbs objective in Bouncing Shapes problem
    """
    metrics = {'loss_phi' : [], 'loss_theta' : [], 'ess' : [], 'E_where' : [], 'E_recon' : [], 'density' : []}
    log_w, q, metrics = oneshot(models, frames, mean_shape, metrics, result_flags)
    q = resample_variables(resampler, q, log_weights=log_w)
    T = frames.shape[2]
    for m in range(num_sweeps-1):
        for t in range(T):
            log_w, q, metrics = apg_where_t(models, frames, q, t, metrics, result_flags)
            q = resample_variables(resampler, q, log_weights=log_w)
        log_w, q, metrics = apg_what(models, frames, q, metrics, result_flags)
        q = resample_variables(resampler, q, log_weights=log_w)
        
    if result_flags['loss_required']:
        metrics['loss_phi'] = torch.cat(metrics['loss_phi'], 0) 
        metrics['loss_theta'] = torch.cat(metrics['loss_theta'], 0) 
    if result_flags['ess_required']:
        metrics['ess'] = torch.cat(metrics['ess'], 0) 
    if result_flags['mode_required']:
        metrics['E_where'] = torch.cat(metrics['E_where'], 0)  
        metrics['E_recon'] = torch.cat(metrics['E_recon'], 0)
    if result_flags['density_required']:
        metrics['density'] = torch.cat(metrics['density'], 0) 
    return metrics


def oneshot(models, frames, conv_kernel, metrics, result_flags):
    (enc_coor, enc_digit, decoder) = models
    T = frames.shape[2]
    S, B, K, DP, DP = conv_kernel.shape
    q = probtorch.Trace()
    for t in range(T):
        q = enc_coor(q, frames, t, conv_kernel, extend_dir='forward')
    q = enc_digit(q, frames, extend_dir='forward')  
    p = decoder(q, frames, recon_level='frames')
    log_q = q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (-log_p)).sum(0).mean()
        metrics['loss_phi'].append(loss_phi.unsqueeze(0))
        metrics['loss_theta'].append(loss_theta.unsqueeze(0))
    if result_flags['ess_required']:
        ess = (1. /(w**2).sum(0))
        metrics['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_where = []
        for t in range(T):
            E_where.append(q['z_where_%d' % (t+1)].dist.loc.unsqueeze(2))
        E_where = torch.cat(E_where, 2)
        metrics['E_where'].append(E_where.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * K * 2
        metrics['E_recon'].append(p['recon'].dist.probs.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * FP * FP
    if result_flags['density_required']:
        metrics['density'].append(log_p.detach().unsqueeze(0))
    return log_w, q, metrics

def apg_where_t(models, frames, q, timestep, metrics, result_flags):
    T = frames.shape[2]
    (enc_coor, enc_digit, decoder) = models
    conv_kernel = decoder(q, frames, recon_level='object')
    # forward
    q_f = enc_coor(q, frames, timestep, conv_kernel, extend_dir='forward')
    p_f = decoder(q_f, frames, recon_level='frames')
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_f = q_f['z_where_%d' % (timestep+1)].log_prob.sum(-1).sum(-1) ## equivanlent to call .log_joint, but not sure which one is computationally efficient
    log_w_f = log_p_f - log_q_f
    # backward
    q_b = enc_coor(q, frames, timestep, conv_kernel, extend_dir='backward')
    p_b = decoder(q_b, frames, recon_level='frames')
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_b = q_b['z_where_%d' % (timestep+1)].log_prob.sum(-1).sum(-1) ## equivanlent to call .log_joint, but not sure which one is computationally efficient
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()          
    if result_flags['loss_required']:
        metrics['loss_phi'].append((w * (- log_q_f)).sum(0).mean().unsqueeze(0))
        metrics['loss_theta'].append((w * (- log_p_f)).sum(0).mean().unsqueeze(0))
    return log_w, q_f, metrics


def apg_what(models, frames, q, metrics, result_flags):
    T = frames.shape[2]
    (enc_coor, enc_digit, decoder) = models
    q_f = enc_digit(q, frames, extend_dir='forward')  
    p_f = decoder(q_f, frames, recon_level='frames')
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_f = q_f['z_what'].log_prob.sum(-1).sum(-1)
    log_w_f = log_p_f - log_q_f
    q_b = enc_digit(q, frames, extend_dir='backward')  
    p_b = decoder(q_b, frames, recon_level='frames')
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_q_b = q_b['z_what'].log_prob.sum(-1).sum(-1)
    log_w_b = log_p_b - log_q_b
    
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    if result_flags['loss_required']:
        loss_phi = (w * (-log_q_f)).sum(0).mean()
        loss_theta = (w * (-log_p_f)).sum(0).mean()
        metrics['loss_phi'][-1] = metrics['loss_phi'][-1] + loss_phi.unsqueeze(0)
        metrics['loss_theta'][-1] = metrics['loss_theta'][-1] + loss_theta.unsqueeze(0)
    if result_flags['ess_required']:
        ess = (1. / (w**2).sum(0))
        metrics['ess'].append(ess.unsqueeze(0))
    if result_flags['mode_required']:
        E_where = []
        for t in range(T):
            E_where.append(q['z_where_%d' % (t+1)].dist.loc.unsqueeze(2))
        E_where = torch.cat(E_where, 2)
        metrics['E_where'].append(E_where.mean(0).unsqueeze(0).cpu().detach())
        metrics['E_recon'].append(p_f['recon'].dist.probs.mean(0).unsqueeze(0).detach().cpu())
    if result_flags['density_required']:
        metrics['density'].append(log_p_f.detach().unsqueeze(0))
    return log_w, q_f, metrics


def hmc_objective(models, AT, frames, result_flags, hmc_sampler, mean_shape):
    """
    HMC objective
    """
    metrics = {'density' : []} 
    S, B, T, FP, _ = frames.shape
    log_w, q, metrics = oneshot(models, frames, mean_shape, metrics, result_flags)
    z_where = []
    for t in range(frames.shape[2]):
        z_where.append(q['z_where_%d' % (t+1)].value.unsqueeze(2))
    z_where = torch.cat(z_where, 2)
    z_what = q['z_what'].value
    metrics = hmc_sampler.hmc_sampling(frames, z_where, z_what, metrics)
    metrics['density'] = torch.cat(metrics['density'], 0)
    return metrics

def bpg_objective(models, AT, frames, result_flags, num_sweeps, resampler, mnist_mean):
    """
    bpg objective
    """
    metrics = {'density' : []} ## a dictionary that tracks things needed during the sweeping
    S, B, T, FP, _ = frames.shape
    (enc_coor, dec_coor, enc_digit, dec_digit) = models
    log_w, z_where, z_what, metrics = oneshot(enc_coor, dec_coor, enc_digit, dec_digit, AT, frames, mnist_mean, metrics, result_flags)
    z_where, z_what = resample_variables(resampler, z_where, z_what, log_weights=log_w)
    for m in range(num_sweeps-1):
        z_where, metrics = apg_where(enc_coor, dec_coor, dec_digit, AT, resampler, frames, z_what, z_where, metrics, result_flags)
        log_w, z_what, metrics = bpg_what(dec_digit, AT, frames, z_where, z_what, metrics)
        z_where, z_what = resample_variables(resampler, z_where, z_what, log_weights=log_w)
    metrics['density'] = torch.cat(metrics['density'], 0) 
    return metrics

def bpg_what(dec_digit, AT, frames, z_where, z_what_old, metrics):
    S, B, T, K, _ = z_where.shape
    z_what_dim = z_what_old.shape[-1]
    cropped = AT.frame_to_digit(frames=frames, z_where=z_where)
    DP = cropped.shape[-1]
    q = Normal(dec_digit.prior_mu, dec_digit.prior_std)
    z_what = q.sample((S, B, K, ))
    cropped = cropped.view(S, B, T, K, int(DP*DP))
    log_p_f, ll_f, recon = dec_digit(frames=frames, z_what=z_what, z_where=z_where, AT=AT)
    log_prior = log_p_f.sum(-1)
    ## backward
    _, ll_b, _ = dec_digit(frames=frames, z_what=z_what_old, z_where=z_where, AT=AT)
    log_w = (ll_f.sum(-1) - ll_b.sum(-1)).detach()
    metrics['density'][-1] = metrics['density'][-1] + (ll_f.sum(-1) + log_prior).unsqueeze(0).detach()
    return log_w, z_what, metrics