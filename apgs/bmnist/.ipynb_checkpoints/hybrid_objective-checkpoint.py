import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat


def hybrid_objective(model, flags, AT, hmc, resampler, apg_sweeps, frames, mnist_mean, K):
    densities = dict()
    S, B, T, FP, _ = frames.shape
    (enc_coor, dec_coor, enc_digit, dec_digit) = model
    log_w_rws, z_where_rws, z_what_rws, log_joint_rws = rws(enc_coor=enc_coor,
                                                            dec_coor=dec_coor,
                                                            enc_digit=enc_digit,
                                                            dec_digit=dec_digit,
                                                            AT=AT,
                                                            frames=frames,
                                                            digit=mnist_mean)

    if flags['apg']:
        print('Running APG updates..')
        trace_apg = {'density' : []}
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_rws)
        z_where_apg = resampler.resample_5dims(var=z_where_rws, ancestral_index=ancestral_index)
        z_what_apg = resampler.resample_4dims(var=z_what_rws, ancestral_index=ancestral_index)
        for m in range(apg_sweeps):
            z_where_apg, trace_apg = apg_where(enc_coor=enc_coor,
                                               dec_coor=dec_coor,
                                               dec_digit=dec_digit,
                                               AT=AT,
                                               resampler=resampler,
                                               frames=frames,
                                               z_what=z_what_apg,
                                               z_where_old=z_where_apg,
                                               trace=trace_apg)
            log_w_apg, z_what_apg, trace_apg = apg_what(enc_digit=enc_digit,
                                                        dec_digit=dec_digit,
                                                        AT=AT,
                                                        frames=frames,
                                                        z_where=z_where_apg,
                                                        z_what_old=z_what_apg,
                                                        trace=trace_apg)
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_apg)
            z_where_apg = resampler.resample_5dims(var=z_where_apg, ancestral_index=ancestral_index)
            z_what_apg = resampler.resample_4dims(var=z_what_apg, ancestral_index=ancestral_index)
        densities['apg'] = torch.cat([log_joint_rws.unsqueeze(0)] + trace_apg['density'], 0)

    if flags['hmc']:
        print('Running HMC-RWS updates..')
        _, _, density_list = hmc.hmc_sampling(ob=frames,
                                              z_where=z_where_rws,
                                              z_what=z_what_rws)
        densities['hmc'] =  torch.cat([log_joint_rws.unsqueeze(0)]+density_list, 0)

    if flags['bpg']:
        print('Running Bootstrapped Population Gibbs updates..')
        trace_bpg = {'density' : []}
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_rws)
        z_where_bpg = resampler.resample_5dims(var=z_where_rws, ancestral_index=ancestral_index)
        z_what_bpg = resampler.resample_4dims(var=z_what_rws, ancestral_index=ancestral_index)
        for m in range(apg_sweeps):
            z_where_bpg, trace_bpg = apg_where(enc_coor=enc_coor,
                                               dec_coor=dec_coor,
                                               dec_digit=dec_digit,
                                               AT=AT,
                                               resampler=resampler,
                                               frames=frames,
                                               z_what=z_what_bpg,
                                               z_where_old=z_where_bpg,
                                               trace=trace_bpg)

            log_w_bpg, z_what_bpg, trace_bpg = bpg_what(dec_digit=dec_digit,
                                                        AT=AT,
                                                        frames=frames,
                                                        z_where=z_where_bpg,
                                                        z_what_old=z_what_bpg,
                                                        trace=trace_bpg)
            ancestral_index = resampler.sample_ancestral_index(log_weights=log_w_bpg)
            z_where_bpg = resampler.resample_5dims(var=z_where_bpg, ancestral_index=ancestral_index)
            z_what_bpg = resampler.resample_4dims(var=z_what_bpg, ancestral_index=ancestral_index)
        densities['bpg'] = torch.cat([log_joint_rws.unsqueeze(0)]+ trace_bpg['density'], 0)
    return densities

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

def rws(enc_coor, dec_coor, enc_digit, dec_digit, AT, frames, digit):
    T = frames.shape[2]
    S, B, K, DP, DP = digit.shape
#     z_where_t_1 = None
#     log_q = []
    z_where = []
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
    z_where = torch.cat(z_where, 2)
    cropped = AT.frame_to_digit(frames=frames, z_where=z_where).view(S, B, T, K, DP*DP)
    q_what = enc_digit(cropped)
    z_what = q_what['z_what'].value # S * B * K * z_what_dim
    log_q_what = q_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
    log_p_what, ll, recon = dec_digit(frames=frames, z_what=z_what, z_where=z_where, AT=AT)
    assert log_q_what.shape == (S, B), "unexpected shape."
    assert log_p_what.shape == (S, B, K), 'unexpected shape.'
    assert ll.shape == (S, B, T), 'unexpected shape.'
    log_p = log_p_where + log_p_what.sum(-1) + ll.sum(-1)
    log_q = log_q_where + log_q_what
    log_w = (log_p - log_q).detach()
    return log_w, z_where, z_what, log_p.detach()

def apg_where(enc_coor, dec_coor, dec_digit, AT, resampler, frames, z_what, z_where_old, trace):
    T = frames.shape[2]
    template = dec_digit(frames=None, z_what=z_what, z_where=None, AT=None)
    S, B, K, DP, DP = template.shape
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
        log_prior = log_prior + log_p_f
        _, ll_f, _ = dec_digit(frames=frame_t.unsqueeze(2), z_what=z_what, z_where=z_where_t.unsqueeze(2), AT=AT)
        _, ll_b, _ = dec_digit(frames=frame_t.unsqueeze(2), z_what=z_what, z_where=z_where_old[:,:,t,:,:].unsqueeze(2), AT=AT)
        assert ll_f.shape == (S, B, 1), "ERROR! unexpected likelihood shape"
        assert ll_b.shape == (S, B, 1), "ERROR! unexpected likelihood shape"
        log_w = (log_w_f - log_w_b  + ll_f.squeeze(-1) - ll_b.squeeze(-1)).detach()
        if t == 0:
            z_where = z_where_t.unsqueeze(2) ## S * B * 1 * K * 2
        else:
            z_where = torch.cat((z_where, z_where_t.unsqueeze(2)), 2) ## S * B * t * K * 2
        ancestral_index = resampler.sample_ancestral_index(log_weights=log_w)
        z_where = resampler.resample_5dims(var=z_where, ancestral_index=ancestral_index)
        z_what = resampler.resample_4dims(var=z_what, ancestral_index=ancestral_index)


    trace['density'].append(log_prior.unsqueeze(0).detach())
    return z_where, trace


def apg_what(enc_digit, dec_digit, AT, frames, z_where, z_what_old, trace):
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
    trace['density'][-1] = trace['density'][-1] + (ll_f.sum(-1) + log_p_f.sum(-1)).unsqueeze(0).detach()
    return log_w, z_what, trace


def bpg_what(dec_digit, AT, frames, z_where, z_what_old, trace):
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
    trace['density'][-1] = trace['density'][-1] + (ll_f.sum(-1) + log_prior).unsqueeze(0).detach()
    return log_w, z_what, trace