import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
def Init_mu(os_mu, f_state, f_angle, dec_x, ob, K, training=True):
    """
    One-shot predicts mu, like a normal VAE
    """
    ## mu
    q_mu, p_mu = os_mu(ob, K)
    log_p_mu = p_mu['means'].log_prob.sum(-1).sum(-1)
    log_q_mu = q_mu['means'].log_prob.sum(-1).sum(-1)
    mu = q_mu['means'].value
    ## z
    q_state, p_state = f_state.forward(ob, mu, K)
    log_p_state = p_state['states'].log_prob.sum(-1)
    log_q_state = q_state['states'].log_prob.sum(-1)
    state = q_state['states'].value ## S * B * N * K
    ## angle
    q_angle, p_angle = f_angle(ob, state, mu)
    log_q_angle = q_angle['angles'].log_prob.sum(-1).sum(-1)
    log_p_angle = p_angle['angles'].log_prob.sum(-1).sum(-1)
    angle = q_angle['angles'].value * 2 * math.pi
    ## decoder
    p_recon = dec_x.forward(ob, state, angle, mu)
    log_p_x = p_recon['likelihood'].log_prob.sum(-1).sum(-1)

    log_w = log_p_x.detach() + log_p_mu + log_p_state + log_p_angle - log_q_mu - log_q_state - log_q_angle
    w = F.softmax(log_w, 0).detach()

    phi_loss = (w * log_w).sum(0).mean()  ## weights S * B
    theta_loss = (w * (- log_p_x)).sum(0).mean()
    ess = (1. / (w ** 2).sum(0)).mean()
    if training:
        return phi_loss, theta_loss, ess, w, mu, state, angle
    else:
        E_mu = q_mu['means'].dist.loc.mean(0)[0].cpu().data.numpy()
        E_z = q_state['states'].dist.probs.mean(0)[0].cpu().data.numpy()
        return  E_mu, E_z, ess, w, mu, state, angle


def Update_mu(f_mu, b_mu, dec_x, ob, state, angle, mu_old, K, training=True):
    """
    Given the current samples of local variable (state and angle),
    update global variable (mu).
    """
    q_f_mu, p_f_mu = f_mu(ob, state, angle)
    mu = q_f_mu['means'].value
    log_p_f_mu = p_f_mu['means'].log_prob.sum(-1)
    log_q_f_mu = q_f_mu['means'].log_prob.sum(-1)
    p_recon_f = dec_x.forward(ob, state, angle, mu)
    log_p_f_x = torch.cat([((state.argmax(-1)==k).float() * (p_recon_f['likelihood'].log_prob.sum(-1))).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_f_w = log_p_f_x.detach() + log_p_f_mu - log_q_f_mu

    q_b_mu, p_b_mu = b_mu(ob, state, angle, sampled=False, mu_old=mu_old)     ## backward
    log_p_b_mu = p_b_mu['means'].log_prob.sum(-1)
    log_q_b_mu = q_b_mu['means'].log_prob.sum(-1)
    p_recon_b = dec_x.forward(ob, state, angle, mu_old)
    log_p_b_x = torch.cat([((state.argmax(-1)==k).float() * (p_recon_b['likelihood'].log_prob.sum(-1).detach())).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_b_w = log_p_b_x.detach() + log_p_b_mu - log_q_b_mu
    phi_loss, theta_loss, w = Compose_IW(log_f_w, log_q_f_mu, log_b_w, log_q_b_mu, log_p_f_x)
    ess = (1. / (w**2).sum(0)).mean()
    if training:
        return phi_loss, theta_loss, ess, w, mu
    else:
        E_mu = q_f_mu['means'].dist.loc.mean(0)[0].cpu().data.numpy()
        return E_mu, ess, w, mu

def Update_state_angle(f_state, f_angle, b_state, b_angle, dec_x, ob, state_old, angle_old, mu, K, training=True):
    """
    Given the current samples of global variable (mu),
    update local variables (state and angle)
    """
    q_f_state, p_f_state = f_state.forward(ob, mu, K)
    state = q_f_state['states'].value ## S * B * N * K
    q_f_angle, p_f_angle = f_angle(ob, state, mu)
    log_q_f = q_f_state['states'].log_prob + q_f_angle['angles'].log_prob.sum(-1)
    log_p_f = p_f_state['states'].log_prob + p_f_angle['angles'].log_prob.sum(-1)
    angle = q_f_angle['angles'].value * 2 * math.pi
    p_recon_f = dec_x.forward(ob, state, angle, mu)
    log_p_f_x = p_recon_f['likelihood'].log_prob.sum(-1)
    log_f_w = log_p_f_x.detach() + log_p_f - log_q_f

    beta_old = angle_old / (2*math.pi)
    q_b_state, p_b_state = b_state.forward(ob, mu, K, sampled=False, state_old=state_old) ## backward
    q_b_angle, p_b_angle = b_angle(ob, state_old, mu, sampled=False, beta_old=beta_old)
    log_q_b = q_b_state['states'].log_prob + q_b_angle['angles'].log_prob.sum(-1)
    log_p_b = p_b_state['states'].log_prob + p_b_angle['angles'].log_prob.sum(-1)
    p_recon_b = dec_x.forward(ob, state_old, angle_old, mu)
    log_p_b_x = p_recon_b['likelihood'].log_prob.sum(-1).detach()
    log_b_w = log_p_b_x.detach() + log_p_b - log_q_b
    phi_loss, theta_loss, w = Compose_IW(log_f_w, log_q_f, log_b_w, log_q_b, log_p_f_x)
    ess = (1. / (w**2).sum(0)).mean()
    if training:
        return phi_loss, theta_loss, ess, w, state, angle
    else:
        E_z = q_f_state['states'].dist.probs.mean(0)[0].cpu().data.numpy()
        return E_z, ess, w, state, angle

def Compose_IW(log_f_w, log_q_f, log_b_w, log_q_b, log_p_x):
    """
    log_w_f : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_w_b : log \frac {p(x, z)} {q_\f (z | z', x)}

    self-normalized importance weights w := softmax(log_f_w = log_b_w).detach()
    phi_loss := w * (-log_q_f) + (- log_q_b)
    theta_loss := w * (log_p_x)
    """
    w = F.softmax(log_f_w - log_b_w, 0).detach()
    phi_loss = (w * ( - log_q_f)).sum(0).sum(-1).mean() - log_q_b.sum(-1).mean()
    theta_loss = (w * (- log_p_x)).sum(0).sum(-1).mean()
    return phi_loss, theta_loss, w
