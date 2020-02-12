import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
"""
Amortized Population Gibbs objective in Bouncing MNIST problem
==========
abbreviations:
K -- number of digits
T -- timesteps in one bmnist sequence
S -- sample size
B -- batch size
ZD -- z_what_dim (ZD=10 in the paper)
FP -- square root of frame pixels (FP=96 in the paper)
DP -- square root of mnist digit pixels (DP=28 by default)
AT -- affine transformer
==========
variables:
frames : S * B * T * FP * FP, sequences of frames in bmnist, as data points
frame_t : S * B * FP * FP, frame at timestep t
z_where : S * B * T * K * 2, latent representaions of the trajectory, as local variables
z_what : S * B * K * ZD, latent representaions of the digits, as global variables
digit :  S * B * K * DP * DP, mnist digit templates used in convolution
mnist_mean : DP * DP,  mean of all the mnist images
===========
conv2d usage https://pytorch.org/docs/1.3.0/nn.functional.html?highlight=conv2d#torch.nn.functional.conv2d
    images: 1 * (SB) * FP * FP, kernels: (SB) * 1 * DP * DP, groups=(SB)
    ===> convoved: 1 * (SB) * (FP-DP+1) * (FP-DP+1)
===========
"""

def apg_objective(model, AT, resampler, apg_sweeps, frames, mnist_mean, K, loss_required=True, ess_required=True, mode_required=False, density_required=False):
    """
    Start with the mnist_mean template,
    and iterate over z_where_t and z_what
    """
    trace = dict()
    if loss_required:
        trace['loss_phi'] = []
        trace['loss_theta'] = []
    if ess_required:
        trace['ess_rws'] = []
        trace['ess_what'] = []
        trace['ess_where'] = []
    if mode_required:
        trace['E_where'] = []
        trace['E_what'] = []
        trace['E_recon'] = []
    if density_required:
        trace['density'] = []

    S, B, T, FP, _ = frames.shape
    (enc_coor, dec_coor, enc_digit, dec_digit) = model
    # metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : [], 'log_joint' : []}
    log_w, z_where, z_what, trace = rws(enc_coor=enc_coor,
                                        dec_coor=dec_coor,
                                        enc_digit=enc_digit,
                                        dec_digit=dec_digit,
                                        AT=AT,
                                        frames=frames,
                                        digit=mnist_mean,
                                        trace=trace,
                                        loss_required=loss_required,
                                        ess_required=ess_required,
                                        mode_required=mode_required,
                                        density_required=density_required)
    ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
    z_where = resampler.resample_5dims(var=z_where, ancestral_index=ancestral_index)
    z_what = resampler.resample_4dims(var=z_what, ancestral_index=ancestral_index)
    for m in range(apg_sweeps):
        z_where, trace = apg_where(enc_coor=enc_coor,
                                   dec_coor=dec_coor,
                                   dec_digit=dec_digit,
                                   AT=AT,
                                   resampler=resampler,
                                   frames=frames,
                                   z_what=z_what,
                                   z_where_old=z_where,
                                   trace=trace,
                                   loss_required=loss_required,
                                   ess_required=ess_required,
                                   mode_required=mode_required,
                                   density_required=density_required)

        log_w, z_what, trace = apg_what(enc_digit=enc_digit,
                                        dec_digit=dec_digit,
                                        AT=AT,
                                        frames=frames,
                                        z_where=z_where,
                                        z_what_old=z_what,
                                        trace=trace,
                                        loss_required=loss_required,
                                        ess_required=ess_required,
                                        mode_required=mode_required,
                                        density_required=density_required)
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
        z_where = resampler.resample_5dims(var=z_where, ancestral_index=ancestral_index)
        z_what = resampler.resample_4dims(var=z_what, ancestral_index=ancestral_index)

    if loss_required:
        trace['loss_phi'] = torch.cat(trace['loss_phi'], 0) # (1+apg_sweeps) * 1
        trace['loss_theta'] = torch.cat(trace['loss_theta'], 0) # (1+apg_sweeps) * 1
    if ess_required:
        if trace['ess_what']:
            trace['ess_what'] = torch.cat(trace['ess_what'], 0) # apg_sweeps * B
        if trace['ess_where']:
            trace['ess_where'] = torch.cat(trace['ess_where'], 0) # apg_sweeps * B
    if mode_required:
        trace['E_where'] = torch.cat(trace['E_where'], 0)  # (1 + apg_sweeps) * B * K * D
        trace['E_what'] = torch.cat(trace['E_what'], 0) # (1 + apg_sweeps) * B * N * K
        trace['E_recon'] = torch.cat(trace['E_recon'], 0).cpu() # (1 + apg_sweeps) * B * T * FP * FP
    if density_required:
        trace['density'] = torch.cat(trace['density'], 0) # (1 + apg_sweeps) * B

    return trace

def propose_one_movement(enc_coor, dec_coor, AT, frame, template, z_where_t_1, z_where_old_t, z_where_old_t_1):
    FP = frame.shape[-1]
    S, B, K, DP, _ = template.shape
    z_where = []
    E_where = []
    log_q_f = []
    log_p_f = []
    frame_left = frame
    log_q_b = [] ## unused if z_where_old_t is None
    log_p_b = []
    for k in range(K):
        template_k = template[:,:,k,:,:]
        conved_k = F.conv2d(frame_left.view(S*B, FP, FP).unsqueeze(0), template_k.view(S*B, DP, DP).unsqueeze(1), groups=int(S*B))
        CP = conved_k.shape[-1] # convolved output pixels ##  S * B * CP * CP
        conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
        q_k_f = enc_coor.forward(conved=conved_k, sampled=True)
        z_where_k = q_k_f['z_where'].value
        z_where.append(z_where_k.unsqueeze(2)) ## expand to S * B * 1 * 2
        E_where.append(q_k_f['z_where'].dist.loc.unsqueeze(2).detach())
        log_q_f.append(q_k_f['z_where'].log_prob.sum(-1).unsqueeze(-1)) # S * B * 1 --> K after loop
        assert q_k_f['z_where'].log_prob.sum(-1).shape == (S, B), 'expected shape.'
        if z_where_t_1 is not None:
            log_p_f.append(dec_coor.forward(z_where_t=z_where_k, z_where_t_1=z_where_t_1[:,:,k,:]).unsqueeze(-1))
            assert dec_coor.forward(z_where_t=z_where_k, z_where_t_1=z_where_t_1[:,:,k,:]).shape ==(S,B), 'unexpected shape.'# S * B
        else:
            log_p_f.append(dec_coor.forward(z_where_t=z_where_k, z_where_t_1=None).unsqueeze(-1))  # S * B
            assert dec_coor.forward(z_where_t=z_where_k, z_where_t_1=None).shape ==(S,B), 'unexpected shape.'
        recon_k = AT.digit_to_frame(template_k.unsqueeze(2), z_where_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2) ## S * B * 64 * 64
        assert recon_k.shape ==(S,B,96,96), 'unexpected shape.'
        frame_left = frame_left - recon_k
        if z_where_old_t is not None:
            log_q_b_k = Normal(q_k_f['z_where'].dist.loc, q_k_f['z_where'].dist.scale).log_prob(z_where_old_t[:,:,k,:]).sum(-1).detach()
#             q_k_b = enc_coor(conved=conved_k, sampled=False, z_where_old=z_where_old_t[:,:,k,:])

            if z_where_old_t_1 is not None:
                log_p_b_k = dec_coor.forward(z_where_t=z_where_old_t[:,:,k,:], z_where_t_1=z_where_old_t_1[:,:,k,:]) # S * B
            else:
                log_p_b_k = dec_coor.forward(z_where_t=z_where_old_t[:,:,k,:], z_where_t_1=None) # S * B
            log_q_b.append(log_q_b_k.unsqueeze(-1)) # S * B * 1 --> K
            log_p_b.append(log_p_b_k.unsqueeze(-1))
    z_where = torch.cat(z_where, 2) # S * B * K * 2
    E_where = torch.cat(E_where, 2) # S * B * K * 2
    log_p_f = torch.cat(log_p_f, -1).sum(-1)
    log_q_f = torch.cat(log_q_f, -1).sum(-1)
    if z_where_old_t is not None:
        log_p_b = torch.cat(log_p_b, -1).sum(-1)
        log_q_b = torch.cat(log_q_b, -1).sum(-1)
        return log_p_f, log_q_f, log_p_b, log_q_b, z_where, E_where
    else:
        return log_p_f, log_q_f, z_where, E_where

def rws(enc_coor, dec_coor, enc_digit, dec_digit, AT, frames, digit, trace, loss_required, ess_required, mode_required, density_required):
    T = frames.shape[2]
    S, B, K, DP, DP = digit.shape
#     z_where_t_1 = None
#     log_q = []
#     log_p = []
    z_where = []
    E_where = []
    for t in range(T):
        if t == 0:
            log_p_where_t, log_q_where_t, z_where_t, E_where_t = propose_one_movement(enc_coor=enc_coor,
                                                                                      dec_coor=dec_coor,
                                                                                      AT=AT,
                                                                                      frame=frames[:,:,t, :,:],
                                                                                      template=digit,
                                                                                      z_where_t_1=None,
                                                                                      z_where_old_t=None,
                                                                                      z_where_old_t_1=None)
            log_p_where = log_p_where_t
            log_q_where = log_q_where_t
        else:
            log_p_where_t, log_q_where_t, z_where_t, E_where_t = propose_one_movement(enc_coor=enc_coor,
                                                                                      dec_coor=dec_coor,
                                                                                      AT=AT,
                                                                                      frame=frames[:,:,t, :,:],
                                                                                      template=digit,
                                                                                      z_where_t_1=z_where_t,
                                                                                      z_where_old_t=None,
                                                                                      z_where_old_t_1=None)
        log_q_where = log_q_where + log_q_where_t
        log_p_where = log_p_where + log_p_where_t
#         z_where_t_1 = z_where_t
        z_where.append(z_where_t.unsqueeze(2)) ## S * B * 1 * K * 2
        E_where.append(E_where_t.unsqueeze(2)) ## S * B * 1 * K * 2
    z_where = torch.cat(z_where, 2)
    E_where = torch.cat(E_where, 2)

    cropped = AT.frame_to_digit(frames=frames, z_where=z_where).view(S, B, T, K, DP*DP)
    q_what = enc_digit(cropped)
    z_what = q_what['z_what'].value # S * B * K * z_what_dim
    E_what = q_what['z_what'].dist.loc
    log_q_what = q_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
    log_p_what, ll, recon = dec_digit(frames=frames, z_what=z_what, z_where=z_where, AT=AT)
    assert log_q_what.shape == (S, B), "unexpected shape."
    assert log_p_what.shape == (S, B, K), 'unexpected shape.'
    assert ll.shape == (S, B, T), 'unexpected shape.'
    log_p = log_p_where + log_p_what.sum(-1) + ll.sum(-1)
    log_q = log_q_where + log_q_what
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    if loss_required:
        loss_phi = (w * (- log_q)).sum(0).mean()
        loss_theta = (w * (-ll.sum(-1))).sum(0).mean()
        trace['loss_phi'].append(loss_phi.unsqueeze(0))
        trace['loss_theta'].append(loss_theta.unsqueeze(0))
    if ess_required:
        ess = (1. /(w**2).sum(0))
        trace['ess_rws'].append(ess)
    if mode_required:
        trace['E_where'].append(E_where.mean(0).unsqueeze(0).detach()) # 1 * B * T * K * 2
        trace['E_what'].append(E_what.mean(0).unsqueeze(0).detach()) # 1 * B * K * z_what_dim
        trace['E_recon'].append(recon.mean(0).unsqueeze(0).detach()) # 1 * B * T * FP * FP
    if density_required:
        trace['density'].append(log_p.mean(0).unsqueeze(0).detach())
    return log_w, z_where, z_what, trace

def apg_where(enc_coor, dec_coor, dec_digit, AT, resampler, frames, z_what, z_where_old, trace, loss_required, ess_required, mode_required, density_required):
    T = frames.shape[2]
    template = dec_digit(frames=None, z_what=z_what, z_where=None, AT=None)
    S, B, K, DP, DP = template.shape

#     z_where_t_1 = None
#     z_where_old_t_1 = None
    # z_where = []
    E_where = []
    LOSS_phi = []
    LOSS_theta = []
    ESS = []
    log_prior = 0.0
    for t in range(T):

        frame_t = frames[:,:,t, :,:]
        if t == 0:
            log_p_f, log_q_f, log_p_b, log_q_b, z_where_t, E_where_t = propose_one_movement(enc_coor=enc_coor,
                                                                                            dec_coor=dec_coor,
                                                                                            AT=AT,
                                                                                            frame=frame_t,
                                                                                            template=template,
                                                                                            z_where_t_1=None,
                                                                                            z_where_old_t=z_where_old[:,:,t,:,:],
                                                                                            z_where_old_t_1=None)
        else:

            log_p_f, log_q_f, log_p_b, log_q_b, z_where_t, E_where_t = propose_one_movement(enc_coor=enc_coor,
                                                                                            dec_coor=dec_coor,
                                                                                            AT=AT,
                                                                                            frame=frame_t,
                                                                                            template=template,
                                                                                            z_where_t_1=z_where_t,
                                                                                            z_where_old_t=z_where_old[:,:,t,:,:],
                                                                                            z_where_old_t_1=z_where_old[:,:,t-1,:,:])
#         z_where_t_1 = z_where_t
#         z_where_old_t_1 = z_where_old_t
        log_w_f = log_p_f - log_q_f
        log_w_b = log_p_b - log_q_b
        if density_required:
            log_prior = log_prior + log_p_f
        if mode_required:
            E_where.append(E_where_t.unsqueeze(2)) ## S * B * 1 * K * 2
        _, ll_f, _ = dec_digit(frames=frame_t.unsqueeze(2), z_what=z_what, z_where=z_where_t.unsqueeze(2), AT=AT)
        _, ll_b, _ = dec_digit(frames=frame_t.unsqueeze(2), z_what=z_what, z_where=z_where_old[:,:,t,:,:].unsqueeze(2), AT=AT)
        assert ll_f.shape == (S, B, 1), "ERROR! unexpected likelihood shape"
        assert ll_b.shape == (S, B, 1), "ERROR! unexpected likelihood shape"
        log_w = (log_w_f - log_w_b  + ll_f.squeeze(-1) - ll_b.squeeze(-1)).detach()
        w = F.softmax(log_w, 0).detach()
        if t == 0:
            z_where = z_where_t.unsqueeze(2) ## S * B * 1 * K * 2
        else:
            z_where = torch.cat((z_where, z_where_t.unsqueeze(2)), 2) ## S * B * t * K * 2
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
        z_where = resampler.resample_5dims(var=z_where, ancestral_index=ancestral_index)
        z_what = resampler.resample_4dims(var=z_what, ancestral_index=ancestral_index)

        if loss_required:
            LOSS_phi.append((w * (- log_q_f)).sum(0).mean().unsqueeze(-1))
            LOSS_theta.append((w * (- ll_f.squeeze(-1))).sum(0).mean().unsqueeze(-1))
        if ess_required:
            ESS.append((1. / (w**2).sum(0)).unsqueeze(-1)) # B vector
    # z_where = torch.cat(z_where, 2)
    if mode_required:
        E_where = torch.cat(E_where, 2)

    if loss_required:

        trace['loss_phi'].append(torch.cat(LOSS_phi, -1).sum(-1).unsqueeze(0))
        trace['loss_theta'].append(torch.cat(LOSS_theta, -1).sum(-1).unsqueeze(0))
    if ess_required:
        ESS = torch.cat(ESS, -1).mean(-1)
        trace['ess_where'].append(ESS.unsqueeze(0))
    if mode_required:
        trace['E_where'].append(E_where.mean(0).unsqueeze(0).detach())
    if density_required:
        trace['density'].append(log_prior.mean(0).unsqueeze(0).detach())
    return z_where, trace


def apg_what(enc_digit, dec_digit, AT, frames, z_where, z_what_old, trace, loss_required, ess_required, mode_required, density_required):
    S, B, T, K, _ = z_where.shape
    cropped = AT.frame_to_digit(frames=frames, z_where=z_where)
    DP = cropped.shape[-1]
    cropped = cropped.view(S, B, T, K, int(DP*DP))
    q_f  = enc_digit(cropped, sampled=True)
    z_what = q_f['z_what'].value # S * B * K * z_what_dim
    log_q_f = q_f['z_what'].log_prob.sum(-1).sum(-1) # S * B
    log_p_f, ll_f, recon = dec_digit(frames=frames, z_what=z_what, z_where=z_where, AT=AT)
    ## backward
    q_b = enc_digit(cropped, sampled=False, z_what_old=z_what_old)
    log_q_b  = q_b['z_what'].log_prob.sum(-1).sum(-1) # S * B
    log_p_b, ll_b, _ = dec_digit(frames=frames, z_what=z_what_old, z_where=z_where, AT=AT)
    log_w = (ll_f.sum(-1) + log_p_f.sum(-1) - log_q_f - (ll_b.sum(-1) + log_p_b.sum(-1) - log_q_b)).detach()
    w = F.softmax(log_w, 0).detach()
    if loss_required:
        loss_phi = (w * (-log_q_f)).sum(0).mean()
        loss_theta = (w * (-ll_f.sum(-1))).sum(0).mean()
        trace['loss_phi'][-1] = trace['loss_phi'][-1] + loss_phi.unsqueeze(0)
        trace['loss_theta'][-1] = trace['loss_theta'][-1] + loss_theta.unsqueeze(0)
    if ess_required:
        ess = (1. / (w**2).sum(0))
        trace['ess_what'].append(ess.unsqueeze(0))
    if mode_required:
        E_what = q_f['z_what'].dist.loc
        trace['E_what'].append(E_what.mean(0).unsqueeze(0).detach())
        trace['E_recon'].append(recon.mean(0).unsqueeze(0).detach())
    if density_required:
        trace['density'][-1] = trace['density'][-1] + (ll_f.sum(-1) + log_p_f.sum(-1)).mean(0).unsqueeze(0).detach()
    return log_w, z_what, trace
