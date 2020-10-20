import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
"""
Tried to implemented DLGM by Matt Hoffman
"""

def dlgm_objective(model, HMC, AT, frames, mnist_mean, K, epsilon, hmc_num_steps, loss_required=True, ess_required=True, mode_required=False, density_required=False):
    """
    Start with the mnist_mean template,
    and iterate over z_where_t and z_what
    """
    trace = dict()
    if loss_required:
        trace['loss_phi'] = []
        trace['loss_theta'] = []
    if ess_required:
        trace['ess_vae'] = []
    if mode_required:
        trace['E_where'] = []
        trace['E_what'] = []
        trace['E_recon'] = []
    if density_required:
        trace['density'] = []

    S, B, T, FP, _ = frames.shape
    (enc_coor, dec_coor, enc_digit, dec_digit) = model
    # metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : [], 'log_joint' : []}
    z_where, z_what, trace, epsilon = vae(enc_coor=enc_coor,
                                             dec_coor=dec_coor,
                                             enc_digit=enc_digit,
                                             dec_digit=dec_digit,
                                             HMC=HMC,
                                             AT=AT,
                                             frames=frames,
                                             digit=mnist_mean,
                                             trace=trace,
                                             epsilon=epsilon,
                                             hmc_num_steps=hmc_num_steps,
                                             loss_required=loss_required,
                                             ess_required=ess_required,
                                             mode_required=mode_required,
                                             density_required=density_required)

    #
    # if loss_required:
    #     trace['loss'] = torch.cat(trace['loss'], 0) # (1+apg_sweeps) * 1
    if mode_required:
        trace['E_where'] = torch.cat(trace['E_where'], 0)  # (1 + apg_sweeps) * B * K * D
        trace['E_what'] = torch.cat(trace['E_what'], 0) # (1 + apg_sweeps) * B * N * K
        trace['E_recon'] = torch.cat(trace['E_recon'], 0) # (1 + apg_sweeps) * B * N * K
    if density_required:
        trace['density'] = torch.cat(trace['density'], 0) # (1 + apg_sweeps) * B

    return trace, epsilon

def propose_one_movement(enc_coor, dec_coor, AT, frame, template, z_where_t_1, z_where_old_t, z_where_old_t_1):
    FP = frame.shape[-1]
    S, B, K, DP, _ = template.shape
    z_where = []
    E_where = []
    Sigma_where = []
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

        Sigma_where.append(q_k_f['z_where'].dist.scale.unsqueeze(2).detach())
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
    Sigma_where = torch.cat(Sigma_where, 2) # S * B * K * 2
    log_p_f = torch.cat(log_p_f, -1).sum(-1)
    log_q_f = torch.cat(log_q_f, -1).sum(-1)
    if z_where_old_t is not None:
        log_p_b = torch.cat(log_p_b, -1).sum(-1)
        log_q_b = torch.cat(log_q_b, -1).sum(-1)
        return log_p_f, log_q_f, log_p_b, log_q_b, z_where, E_where
    else:
        return log_p_f, log_q_f, z_where, E_where, Sigma_where

def vae(enc_coor, dec_coor, enc_digit, dec_digit, HMC, AT, frames, digit, trace, epsilon, hmc_num_steps, loss_required, ess_required, mode_required, density_required):
    T = frames.shape[2]
    S, B, K, DP, DP = digit.shape
#     z_where_t_1 = None
#     log_q = []
#     log_p = []
    z_where = []
    E_where = []
    Sigma_where = []
    for t in range(T):
        if t == 0:
            log_p_where_t, log_q_where_t, z_where_t, E_where_t, Sigma_where_t = propose_one_movement(enc_coor=enc_coor,
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
            log_p_where_t, log_q_where_t, z_where_t, E_where_t, Sigma_where_t = propose_one_movement(enc_coor=enc_coor,
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
        Sigma_where.append(Sigma_where_t.unsqueeze(2)) ## S * B * 1 * K * 2
    z_where = torch.cat(z_where, 2)
    E_where = torch.cat(E_where, 2)
    Sigma_where = torch.cat(Sigma_where, 2)

    cropped = AT.frame_to_digit(frames=frames, z_where=z_where.detach()).view(S, B, T, K, DP*DP)
    q_what = enc_digit(cropped)
    z_what = q_what['z_what'].value # S * B * K * z_what_dim
    E_what = q_what['z_what'].dist.loc
    log_q_what = q_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
    log_p_what, ll, recon = dec_digit(frames=frames, z_what=z_what, z_where=z_where, AT=AT)
    log_p = log_p_where + log_p_what.sum(-1) + ll.sum(-1)
    log_q = log_q_where + log_q_what
    log_w = log_p - log_q
    w = F.softmax(log_w, 0).detach()
    Sigma_what = q_what['z_what'].dist.scale.detach()
    step_size_what = epsilon * Sigma_what
    step_size_where = epsilon * Sigma_where

    leapfrog_num_steps = int(1. / epsilon)
    trace_hmc = {'density' : []}
    z_where_hmc, z_what_hmc, _ = HMC.hmc_sampling(ob=frames,
                                                     z_where=z_where.detach(),
                                                     z_what=z_what.detach(),
                                                     trace=trace_hmc,
                                                     hmc_num_steps=hmc_num_steps,
                                                     step_size_what=step_size_what,
                                                     step_size_where=step_size_where,
                                                     leapfrog_num_steps=leapfrog_num_steps)
    if HMC.smallest_accept_ratio > 0.25:
        epsilon *= 1.005
    else:
        epsilon *= 0.995
    log_p_what_hmc, ll_hmc, _ = dec_digit(frames=frames, z_what=z_what_hmc.detach(), z_where=z_where_hmc.detach(), AT=AT)
    log_p_theta = (log_p_what.sum(-1) + ll.sum(-1)).mean()
    if loss_required:
        elbo = log_w.mean()
        trace['loss_phi'] = -elbo
        trace['loss_theta'] = - log_p_theta
    if ess_required:
        ess = (1. /(w**2).sum(0))
        trace['ess_vae'].append(ess)
    if mode_required:
        trace['E_where'].append(E_where.mean(0).unsqueeze(0).detach()) # 1 * B * T * K * 2
        trace['E_what'].append(E_what.mean(0).unsqueeze(0).detach()) # 1 * B * K * z_what_dim
        trace['E_recon'].append(recon.mean(0).unsqueeze(0).detach())
    if density_required:
        trace['density'].append(log_p.mean(0).unsqueeze(0).detach())
    return z_where, z_what, trace, epsilon
