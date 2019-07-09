import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
def oneshot(oneshot_mu, gibbs_state, dec_x, ob, K):
    """
    initialize mu by one-shot encoder
    """
    q_mu, p_mu = oneshot_mu(ob, K)
    log_p_os = p_mu['means'].log_prob.sum(-1).sum(-1) + p_mu['angles'].log_prob.sum(-1).sum(-1)
    log_q_os = q_mu['means'].log_prob.sum(-1).sum(-1) + q_mu['angles'].log_prob.sum(-1).sum(-1)
    mu = q_mu['means'].value
    angle = q_mu['angles'].value * 2 * math.pi
    ## z
    q_state, p_state = gibbs_state.forward(dec_x, ob, angle, mu, K)
    log_p_state = p_state['zs'].log_prob.sum(-1)
    log_q_state = q_state['zs'].log_prob.sum(-1)
    state = q_state['zs'].value ## S * B * N * K
    log_recon = dec_x.forward(ob, state, angle, mu, idw_flag=False).sum(-1)
    log_w = log_recon + log_p_state - log_q_state + log_p_os - log_q_os
    w = F.softmax(log_w, 0).detach()
    eubo = (w * log_w).sum(0).mean()  ## weights S * B
    elbo = log_w.mean()
    return state, angle, mu, w, eubo, elbo, q_mu, p_mu

def Update_mu(enc_mu, dec_x, ob, state, angle, mu_old, K):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu ).
    """
    q_mu, p_mu = enc_mu(ob, state, angle)
    log_p_mu = p_mu['means'].log_prob.sum(-1)
    log_q_mu = q_mu['means'].log_prob.sum(-1)
    mu = q_mu['means'].value
    log_recon = dec_x.forward(ob, state, angle, mu, idw_flag=True)
    log_w_f = log_recon + log_p_mu - log_q_mu
    ## backward
    log_p_mu_old = Normal(p_mu['means'].dist.loc, p_mu['means'].dist.scale).log_prob(mu_old).sum(-1)
    log_q_mu_old = Normal(q_mu['means'].dist.loc, q_mu['means'].dist.scale).log_prob(mu_old).sum(-1)
    log_recon_old = dec_x.forward(ob, state, angle, mu_old, idw_flag=True)
    log_w_b = log_recon_old + log_p_mu_old - log_q_mu_old
    w, eubo, elbo = Grad_phi(log_w_f, log_w_b)
    return mu, w, eubo, elbo, q_mu, p_mu

def Update_angle(enc_angle, dec_x, ob, state, mu, angle_old):
    betas = angle_old / (2*math.pi)
    q_angle, p_angle = enc_angle(ob, state, mu)
    log_q_angle = q_angle['angles'].log_prob.sum(-1)
    log_p_angle = p_angle['angles'].log_prob.sum(-1)
    angle = q_angle['angles'].value * 2 * math.pi
    log_recon = dec_x.forward(ob, state, angle, mu, idw_flag=False)
    log_w_f = log_recon + log_p_angle - log_q_angle
    log_p_angle_old = Beta(p_angle['angles'].dist.concentration1, p_angle['angles'].dist.concentration0).log_prob(betas).sum(-1)
    log_q_angle_old = Beta(q_angle['angles'].dist.concentration1, q_angle['angles'].dist.concentration0).log_prob(betas).sum(-1)
    log_recon_old = dec_x.forward(ob, state, angle_old, mu, idw_flag=False)
    log_w_b = log_recon_old + log_p_angle_old - log_q_angle_old
    w, eubo, elbo = Grad_phi(log_w_f, log_w_b)
    return angle, w, eubo, elbo, q_angle, p_angle

def Update_state(gibbs_state, dec_x, ob, angle, mu, state_old):
    K = state_old.shape[-1]
    q_state, p_state = gibbs_state.forward(dec_x, ob, angle, mu, K)
    log_p_state = p_state['zs'].log_prob
    log_q_state = q_state['zs'].log_prob
    state = q_state['zs'].value
    log_recon = dec_x.forward(ob, state, angle, mu, idw_flag=False)
    log_w_f = log_recon + log_p_state - log_q_state
    ## backward
    log_p_state_old = cat(probs=p_state['zs'].dist.probs).log_prob(state_old)
    log_q_state_old = cat(probs=q_state['zs'].dist.probs).log_prob(state_old)
    log_recon_old = dec_x.forward(ob, state_old, angle, mu, idw_flag=False)
    log_w_b = log_recon_old + log_p_state_old - log_q_state_old
    w, eubo, elbo = Grad_phi(log_w_f, log_w_b)
    return state, w, eubo, elbo, q_state, p_state

def Grad_phi(log_w_f, log_w_b):
    """
    log_w_f : log \frac {p(x, z')} {q_\f (z' | z, x)}
    log_w_b : log \frac {p(x, z)} {q_\f (z | z', x)}
    """
    log_w = log_w_f - log_w_b
    w = F.softmax(log_w, 0).detach()
    # kl_f_b = - log_w.sum(-1).mean()
    # kl_b_f = (w * log_w).sum(0).sum(-1).mean()
    # symkl_db = kl_f_b + kl_b_f
    eubo = (w * log_w_f).sum(0).sum(-1).mean()
    elbo = log_w_f.sum(-1).mean()

    return w, eubo, elbo
