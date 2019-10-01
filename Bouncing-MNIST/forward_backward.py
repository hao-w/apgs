import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
from torch import logsumexp

def Init_step(enc_coor, enc_digit, dec_digit, frames, tjs, tj_std, crop, mnist_mean, training=True):
    q_where = enc_coor(frames=frames, digit=mnist_mean)
    z_where = q_where['z_where'].value
    log_q_where = q_where['z_where'].log_prob.sum(-1).sum(-1) # S * B
    ##
    log_p_where = Normal(tjs, tj_std).log_prob(z_where).sum(-1).sum(-1)
    ##
    q_what, p_what = enc_digit(frames, z_where, crop)
    z_what = q_what['z_what'].value
    log_q_what = q_what['z_what'].log_prob.sum(-1) # S * B
    log_p_what = p_what['z_what'].log_prob.sum(-1) # S * B
    recon, ll = dec_digit(frames, z_what, crop, z_where=z_where)
    w = F.softmax(ll.sum(-1) + log_p_what + log_p_where - log_q_what - log_q_where, 0).detach()
    if training:
        phi_loss = (w * (- log_q_what - log_q_where)).sum(0).mean()
        theta_loss = (w * (- ll.sum(-1))).sum(0).mean()
        ess = (1. / (w ** 2).sum(0)).mean()
        return phi_loss, theta_loss, ess, w, z_where, z_what
    else:
        ess = (1. / (w ** 2).sum(0)).mean()
        E_what =  q_what['z_what'].dist.loc.mean(0).cpu().data.numpy()
        E_where = q_where['z_where'].dist.loc.mean(0).cpu().data.numpy()
        return E_where, E_what, ess, w, z_where, z_what

def Update_where(enc_coor, dec_digit, frames, tjs, tj_std, crop, z_what, z_where_old, training=True):
    """
    update the z_where given the frames and digit images
    z_what  S * B * H* W
    """
    digit = dec_digit(frames, z_what, crop, z_where=None, intermediate=True).detach()
    q_f_where = enc_coor(frames=frames, digit=digit)
    z_where = q_f_where['z_where'].value
    log_q_f = q_f_where['z_where'].log_prob.sum(-1).sum(-1) # S * B
    ##
    log_p_f = Normal(tjs, tj_std).log_prob(z_where).sum(-1).sum(-1)

    ##
    recon, ll_f = dec_digit(frames, z_what, crop, z_where=z_where)
    log_w_f = ll_f.sum(-1) + log_p_f - log_q_f

    q_b_where = enc_coor(frames=frames, digit=digit, sampled=False, z_where_old=z_where_old)
    _, ll_b = dec_digit(frames, z_what, crop, z_where=z_where_old)
    log_p_b = Normal(tjs, tj_std).log_prob(z_where_old).sum(-1).sum(-1)
    log_w_b = ll_b.sum(-1).detach() + log_p_b - q_b_where['z_where'].log_prob.sum(-1).sum(-1).detach()
    # log_w_b = ll_b.sum(-1).detach() + p_b_where['z_where'].log_prob.sum(-1).sum(-1).detach() - q_b_where['z_where'].log_prob.sum(-1).sum(-1).detach()
    if training:
        phi_loss, theta_loss, w = Compose_IW(log_w_f, log_q_f, log_w_b, ll_f.sum(-1))
        ess = (1. / (w**2).sum(0)).mean()
        return phi_loss, theta_loss, ess, w, z_where
    else:
        w = F.softmax(log_f_w - log_b_w, 0).detach()
        ess = (1. / (w**2).sum(0)).mean()
        E_where = q_f_where['z_where'].dist.loc.mean(0).cpu().data.numpy()
        return recon, E_where, ess, w, z_where

def Update_what(enc_digit, dec_digit, frames, crop, z_where, z_what_old, training=True):
    """
    one-shot predicts z_what
    frames : B * T * 64 * 64
    S : sample_size
    DP : digit height/width
    FP : frame height/width
    """
    q_f_what, p_f_what = enc_digit(frames, z_where, crop)
    z_what = q_f_what['z_what'].value
    log_q_f = q_f_what['z_what'].log_prob.sum(-1) # S * B
    log_p_f = p_f_what['z_what'].log_prob.sum(-1) # S * B
    recon, ll_f = dec_digit(frames, z_what, crop, z_where=z_where)
    log_w_f = ll_f.sum(-1) + log_p_f - log_q_f
    ## backward
    q_b_what, p_b_what = enc_digit(frames, z_where, crop, sampled=False, z_what_old=z_what_old)
    _, ll_b = dec_digit(frames, z_what_old, crop, z_where=z_where)
    log_w_b = ll_b.sum(-1).detach() + p_b_what['z_what'].log_prob.sum(-1).detach() - q_b_what['z_what'].log_prob.sum(-1).detach()
    if training:
        phi_loss, theta_loss, w = Compose_IW(log_w_f, log_q_f, log_w_b, ll_f.sum(-1))
        ess = (1. / (w**2).sum(0)).mean()
        return phi_loss, theta_loss, ess, w, z_what
    else:
        w = F.softmax(log_f_w - log_b_w, 0).detach()
        ess = (1. / (w**2).sum(0)).mean()
        E_what = q_f_what['z_what'].dist.loc.mean(0).cpu().data.numpy()
        return recon, E_what, ess, w, z_what


def Compose_IW(log_w_f, log_q_f, log_w_b, log_p_x):
    """
    log_w_f : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_w_b : log \frac {p(x, z)} {q_\f (z | z', x)}

    self-normalized importance weights w := softmax(log_f_w = log_b_w).detach()
    phi_loss := w * (-log_q_f) + (- log_q_b)
    theta_loss := w * (log_p_x)
    """
    w = F.softmax(log_w_f - log_w_b, 0).detach()
    phi_loss = (w * ( - log_q_f)).sum(0).mean()
    theta_loss = (w * (- log_p_x)).sum(0).mean()
    return phi_loss, theta_loss, w
