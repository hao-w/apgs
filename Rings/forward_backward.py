import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
def Init_mu(os_mu, f_state, f_angle, dec_x, ob, K):
    """
    initialize mu by one-shot encoder
    """
    ## mu
    q_mu, p_mu = os_mu(ob, K)
    log_p_mu = p_mu['means'].log_prob.sum(-1).sum(-1)
    log_q_mu = q_mu['means'].log_prob.sum(-1).sum(-1)
    mu = q_mu['means'].value
    ## z
    q_state, p_state = f_state.forward(ob, mu, K)
    log_p_state = p_state['zs'].log_prob.sum(-1)
    log_q_state = q_state['zs'].log_prob.sum(-1)
    state = q_state['zs'].value ## S * B * N * K
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
    return phi_loss, theta_loss, ess, w, mu, state, angle

def Update_mu(f_mu, dec_x, ob, state, angle, mu_old, K):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu ).
    """
    q_mu, p_mu = f_mu(ob, state, angle)
    mu = q_mu['means'].value
    log_p_f = p_mu['means'].log_prob.sum(-1)
    log_q_f = q_mu['means'].log_prob.sum(-1)
    p_recon_f = dec_x.forward(ob, state, angle, mu)
    log_p_x_f_naive = p_recon_f['likelihood'].log_prob.sum(-1)
    log_p_x_f = torch.cat([((state.argmax(-1)==k).float() * log_p_x_f_naive).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_f_w = log_p_x_f.detach() + log_p_f - log_q_f
    ## backward
    log_p_b = Normal(p_mu['means'].dist.loc, p_mu['means'].dist.scale).log_prob(mu_old).sum(-1)
    log_q_b = Normal(q_mu['means'].dist.loc, q_mu['means'].dist.scale).log_prob(mu_old).sum(-1)
    p_recon_b = dec_x.forward(ob, state, angle, mu_old)
    log_p_x_b_naive = p_recon_b['likelihood'].log_prob.sum(-1)
    log_p_x_b = torch.cat([((state.argmax(-1)==k).float() * log_p_x_b_naive).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_b_w = log_p_x_b.detach() + log_p_b - log_q_b
    phi_loss, theta_loss, w = Compose_IW(log_f_w, log_b_w, log_p_x_f)
    ess = (1. / (w**2).sum(0)).mean()
    return phi_loss, theta_loss, ess, w, mu

def Update_state_angle(f_state, f_angle, dec_x, ob, state_old, angle_old, mu, K):
    q_state, p_state = f_state.forward(ob, mu, K)
    state = q_state['zs'].value ## S * B * N * K
    q_angle, p_angle = f_angle(ob, state, mu)
    log_q_f = q_state['zs'].log_prob + q_angle['angles'].log_prob.sum(-1)
    log_p_f = p_state['zs'].log_prob + p_angle['angles'].log_prob.sum(-1)
    angle = q_angle['angles'].value * 2 * math.pi
    p_recon_f = dec_x.forward(ob, state, angle, mu)
    log_p_x_f = p_recon_f['likelihood'].log_prob.sum(-1)
    log_f_w = log_p_x_f.detach() + log_p_f - log_q_f
    ## backward
    beta_old = angle_old / (2*math.pi)
    q_angle_old, p_angle_old = f_angle(ob, state_old, mu)
    log_p_b = cat(probs=p_state['zs'].dist.probs).log_prob(state_old) + Beta(p_angle_old['angles'].dist.concentration1, p_angle_old['angles'].dist.concentration0).log_prob(beta_old).sum(-1)
    log_q_b = cat(probs=q_state['zs'].dist.probs).log_prob(state_old) + Beta(q_angle_old['angles'].dist.concentration1, q_angle_old['angles'].dist.concentration0).log_prob(beta_old).sum(-1)
    p_recon_b = dec_x.forward(ob, state_old, angle_old, mu)
    log_p_x_b = p_recon_b['likelihood'].log_prob.sum(-1)
    log_b_w = log_p_x_b.detach() + log_p_b - log_q_b
    phi_loss, theta_loss, w = Compose_IW(log_f_w, log_b_w, log_p_x_f)
    ess = (1. / (w**2).sum(0)).mean()
    return phi_loss, theta_loss, ess, w, state, angle

def Compose_IW(log_f_w, log_b_w, log_p_x):
    """
    log_w_f : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_w_b : log \frac {p(x, z)} {q_\f (z | z', x)}
    """
    log_w = log_f_w - log_b_w
    w = F.softmax(log_w, 0).detach()
    phi_loss = (w * log_f_w).sum(0).sum(-1).mean()
    theta_loss = (w * (- log_p_x)).sum(0).sum(-1).mean()
    return phi_loss, theta_loss, w
