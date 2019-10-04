import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
import probtorch
"""
Update z_what^k
========================

frames  : S * B * T * 64 * 64
z_where : S * B * T * K * D
"""

def Init_where_what(enc_coor, dec_coor, enc_digit, dec_digit, AT, frames, mnist_mean, training=True):
    S, B, T, _, _ = frames.shape
    K = 2
    z_where  = []
    E_where = []
    Cropped = []
    for t in range(T):
        frame_t = frames[:,:,t, :,:]
        if t == 0:
            log_w_t, z_where_t, E_where_t = Sample_where_t(enc_coor, dec_coor, AT, frame_t=frame_t, digit=mnist_mean, z_where_t_1=None)
            log_w = log_w_t
        else:
            log_w_t, z_where_t, E_where_t = Sample_where_t(enc_coor, dec_coor, AT, frame_t=frame_t, digit=mnist_mean, z_where_t_1=z_where_t)
            log_w = log_w + log_w_t

        z_where.append(z_where_t.unsqueeze(2))
        E_where.append(E_where_t.unsqueeze(2))
        Cropped.append(AT.frame_to_digit(frame_t, z_where_t).view(S, B, K, 28*28).unsqueeze(2))
    z_where = torch.cat(z_where, 2)
    E_where = torch.cat(E_where, 2)
    Cropped = torch.cat(Cropped, 2) ## S * B * T * K * 784
    q_f_what, p_f_what = enc_digit(Cropped)
    z_what = q_f_what['z_what'].value
    log_q_f = q_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
    log_p_f = p_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B

    ll_f = []
    for t in range(T):
        _, ll_f_t = dec_digit(frames[:,:,t, :,:], z_what, z_where=z_where[:,:,t,:,:])
        ll_f.append(ll_f_t.unsqueeze(0))
    ll_f = torch.cat(ll_f, 0).sum(0)
    log_w = log_w + ll_f.detach() + log_p_f - log_q_f
    w = F.softmax(log_w, 0).detach()
    ess = (1. / (w**2).sum(0)).mean()

    if training:
        phi_loss = (w * log_w).sum(0).mean()
        theta_loss = (w * (-ll_f)).sum(0).mean()
        return phi_loss, theta_loss, ess, w, z_where, z_what
    else:
        E_what = q_f_what['z_what'].dist.loc.mean(0).cpu()
        return E_what, E_where, ess , w, z_where, z_what

def Sample_where_t(enc_coor, dec_coor, AT, frame_t, digit, z_where_t_1=None):
    S, B, K, _, _ = digit.shape
    frame_left = frame_t
    log_w = []
    z_where = []
    E_where_t = []
    for k in range(K):
        digit_k = digit[:,:,k,:,:]
        conved_k = F.conv2d(frame_left.view(S*B, 64, 64).unsqueeze(0), digit_k.view(S*B, 28, 28).unsqueeze(1), groups=int(S*B))
        CP = conved_k.shape[-1] # convolved output pixels ## T * S * B * CP * CP
        conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
        q_k = enc_coor.forward(conved_k)
        z_where_k = q_k['z_where'].value
        log_q_f_k = q_k['z_where'].log_prob.sum(-1)
        E_where_t.append(q_k['z_where'].dist.loc.mean(0).cpu().unsqueeze(2))
        if z_where_t_1 is not None:
            log_p_f_k = dec_coor.forward(z_where_k, z_where_t_1=z_where_t_1[:,:,k,:])
        else:
            log_p_f_k = dec_coor.forward(z_where_k)
        log_w.append((log_p_f_k - log_q_f_k).unsqueeze(0))
        z_where.append(z_where_k.unsqueeze(2))
        recon_frame_t_k = AT.digit_to_frame(digit_k.unsqueeze(2), z_where_k.unsqueeze(2)).squeeze(2) ## S * B * 64 * 64
        frame_left = frame_left - recon_frame_t_k
    return torch.cat(log_w, 0).sum(0), torch.cat(z_where, 2), torch.cat(E_where_t, 2)


def Sample_where_t2(enc_coor, dec_coor, AT, frame_t, digit, z_where_old_t, z_where_old_t_1=None, z_where_t_1=None):
    S, B, K, _, _ = digit.shape
    frame_left = frame_t
    log_w = []
    log_w_b = []
    z_where = []
    E_where_t = []
    for k in range(K):

        digit_k = digit[:,:,k,:,:]
        conved_k = F.conv2d(frame_left.view(S*B, 64, 64).unsqueeze(0), digit_k.view(S*B, 28, 28).unsqueeze(1), groups=int(S*B))
        CP = conved_k.shape[-1] # convolved output pixels ## T * S * B * CP * CP
        conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
        q_k = enc_coor.forward(conved_k)
        z_where_k = q_k['z_where'].value
        log_q_f_k = q_k['z_where'].log_prob.sum(-1)
        E_where_t.append(q_k['z_where'].dist.loc.mean(0).cpu().unsqueeze(2))
        if z_where_t_1 is not None:
            log_p_f_k = dec_coor.forward(z_where_k, z_where_t_1=z_where_t_1[:,:,k,:])
        else:
            log_p_f_k = dec_coor.forward(z_where_k)
        ## backward
        log_q_b_k = Normal(q_k['z_where'].dist.loc, q_k['z_where'].dist.scale).log_prob(z_where_old_t[:,:,k,:]).sum(-1).detach()
        if z_where_old_t_1 is not None:
            log_p_b_k = dec_coor.forward(z_where_old_t[:,:,k,:], z_where_t_1=z_where_old_t_1[:,:,k,:])
        else:
            log_p_b_k = dec_coor.forward(z_where_old_t[:,:,k,:])

        log_w.append((log_p_f_k - log_q_f_k).unsqueeze(0))
        log_w_b.append((log_p_b_k - log_q_b_k).unsqueeze(0))
        z_where.append(z_where_k.unsqueeze(2))
        recon_frame_t_k = AT.digit_to_frame(digit_k.unsqueeze(2), z_where_k.unsqueeze(2)).squeeze(2) ## S * B * 64 * 64
        frame_left = frame_left - recon_frame_t_k
    return torch.cat(log_w, 0).sum(0), torch.cat(log_w_b, 0).sum(0), torch.cat(z_where, 2), torch.cat(E_where_t, 2)




def APG_where(enc_coor, dec_coor, dec_digit, AT, frames, z_what, z_where_old, training=True):
    S, B, T, _, _ = frames.shape
    K = 2
    z_where = []
    E_where = []
    Phi_loss = []
    Theta_loss = []
    ESS = []
    for t in range(T):
        frame_t = frames[:,:,t, :,:]
        digit = dec_digit(frame_t, z_what)
        if t == 0:
            log_w_t, log_w_b_t, z_where_t, E_where_t = Sample_where_t2(enc_coor, dec_coor, AT, frame_t=frame_t, digit=digit, z_where_old_t=z_where_old[:,:,t, :, :], z_where_old_t_1=None, z_where_t_1=None)
        else:
            log_w_t, log_w_b_t, z_where_t, E_where_t = Sample_where_t2(enc_coor, dec_coor, AT, frame_t=frame_t, digit=digit, z_where_old_t=z_where_old[:,:,t, :, :],  z_where_old_t_1=z_where_old[:,:,t-1, :,:], z_where_t_1=z_where_t)

        recon_t, ll_f_t = dec_digit(frame_t, z_what, z_where=z_where_t)
        _, ll_b_t = dec_digit(frame_t, z_what, z_where=z_where_old[:,:,t,:,:])
        log_w_f_t = ll_f_t.detach() + log_w_t.detach()
        log_w_b_t = ll_b_t.detach() + log_w_b_t.detach()
        w = F.softmax(log_w_f_t - log_w_b_t, 0).detach()
        ess = (1. / (w**2).sum(0)).mean()
        ESS.append(ess.unsqueeze(0))
        z_where_t = Resample_where(z_where_t, w)
        z_where.append(z_where_t.unsqueeze(2))
        if training:
            Phi_loss.append((w * log_w_t).sum(0).mean().unsqueeze(0))
            Theta_loss.append((w * (-ll_f_t)).sum(0).mean().unsqueeze(0))
        E_where.append(E_where_t.unsqueeze(2))

    if training:
        return torch.cat(Phi_loss, 0).sum(0), torch.cat(Theta_loss,0).sum(0), torch.cat(ESS, 0).mean(), w, torch.cat(z_where, 2)
    else:
        return E_where, torch.cat(ESS, 0).mean(), w,  torch.cat(E_where, 2)


def APG_what(enc_digit, dec_digit, AT, frames, z_where, z_what_old=None, training=True):
    K = 2
    S, B, T, _, _ = frames.shape
    croppd = AT.frame_to_digit_vectorized(frames, z_where).view(S, B, T, K, 28*28)
    q_f_what, p_f_what = enc_digit(croppd)
    z_what = q_f_what['z_what'].value
    log_q_f = q_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
    log_p_f = p_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
    recon, ll_f = dec_digit.forward_vectorized(frames, z_what, z_where=z_where)
    log_w_f = ll_f + log_p_f - log_q_f

    q_b_what, p_b_what = enc_digit(croppd, sampled=False, z_what_old=z_what_old)
    log_p_b = p_b_what['z_what'].log_prob.sum(-1).sum(-1).detach()
    log_q_b  = q_b_what['z_what'].log_prob.sum(-1).sum(-1).detach()
    _, ll_b = dec_digit.forward_vectorized(frames, z_what_old, z_where=z_where)
    log_w_b = ll_b.detach() + log_p_b - log_q_b
    if training:
        w = F.softmax(log_w_f - log_w_b, 0).detach()
        phi_loss = (w * (- log_q_f)).sum(0).mean()
        theta_loss =(w * (- ll_f)).sum(0).mean()
        ess = (1. / (w**2).sum(0)).mean()
        return phi_loss, theta_loss, ess, w, z_what
    else:
        w = F.softmax(log_w_f - log_w_b, 0).detach()
        ess = (1. / (w**2).sum(0)).mean().detach()
        E_what = q_f_what['z_what'].dist.loc.mean(0).detach()
        return  E_what, recon.detach(), ess, w, z_what



def Resample_what(z_what, weights):
    S, B, K, dim4 = z_what.shape
    ancesters = Categorical(weights.transpose(0, 1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, dim4)
    return torch.gather(z_what, 0, ancesters)

def Resample_where(z_where, weights):
    S, B, K, dim4 = z_where.shape
    ancesters = Categorical(weights.transpose(0, 1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, dim4) ## S * B * T * K * 2
    return torch.gather(z_where, 0, ancesters)
