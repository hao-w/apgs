import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
def oneshot(oneshot_mu, oneshot_state, enc_angle, dec_x, ob, K):
    """
    initialize mu by one-shot encoder
    """
    ## mu
    q_mu, p_mu = oneshot_mu(ob, K)
    log_p_mu = p_mu['means'].log_prob.sum(-1).sum(-1)
    log_q_mu = q_mu['means'].log_prob.sum(-1).sum(-1)
    mu = q_mu['means'].value
    ## z
    q_state, p_state = oneshot_state.forward(ob, mu, K)
    log_p_state = p_state['zs'].log_prob.sum(-1)
    log_q_state = q_state['zs'].log_prob.sum(-1)
    state = q_state['zs'].value ## S * B * N * K
    ## angle
    q_angle, p_angle = enc_angle(ob, state, mu)
    log_q_angle = q_angle['angles'].log_prob.sum(-1).sum(-1)
    log_p_angle = p_angle['angles'].log_prob.sum(-1).sum(-1)
    angle = q_angle['angles'].value * 2 * math.pi
    ## decoder
    p_recon = dec_x.forward(ob, state, angle, mu)
    log_recon = p_recon['likelihood'].log_prob.sum(-1).sum(-1)

    log_w = log_recon.detach() + log_p_mu + log_p_state + log_p_angle - log_q_mu - log_q_state - log_q_angle
    w = F.softmax(log_w, 0).detach()
    eubo = (w * log_w).sum(0).mean()  ## weights S * B
    theta_loss = (w * (- log_recon)).sum(0).mean()
    elbo = log_w.mean().detach()
    return state, angle, mu, w, eubo, elbo, theta_loss, q_mu, q_state, p_recon

def Update_mu(enc_mu, dec_x, ob, state_old, angle_old, mu_old):
    """
    Given the current samples for local variable (state),
    sample new global variable (eta = mu ).
    """
    K = state_old.shape[-1]
    q_mu, p_mu = enc_mu(ob, state_old, angle_old)
    mu = q_mu['means'].value
    log_p_f = p_mu['means'].log_prob.sum(-1)
    log_q_f = q_mu['means'].log_prob.sum(-1)
    p_recon_f = dec_x.forward(ob, state_old, angle_old, mu)
    ll = p_recon_f['likelihood'].log_prob.sum(-1)
    log_recon_f = torch.cat([((state_old.argmax(-1)==k).float() * ll).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K

    log_w_f = log_recon_f.detach() + log_p_f - log_q_f
    ## backward
    beta_old = angle_old / (2*math.pi)
    log_p_b= Normal(p_mu['means'].dist.loc, p_mu['means'].dist.scale).log_prob(mu_old).sum(-1)
    log_q_b = Normal(q_mu['means'].dist.loc, q_mu['means'].dist.scale).log_prob(mu_old).sum(-1)
    p_recon_b = dec_x.forward(ob, state_old, angle_old, mu_old)
    ll_b = p_recon_b['likelihood'].log_prob.sum(-1)
    log_recon_b = torch.cat([((state_old.argmax(-1)==k).float() * ll_b).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
    log_w_b = log_recon_b.detach() + log_p_b - log_q_b
    w, eubo, elbo, theta_loss = Grad_phi(log_w_f, log_w_b, log_recon_f)
    return mu, w, eubo, elbo, theta_loss, q_mu

def Update_state_angle(oneshot_state, enc_angle, dec_x, ob, state_old, angle_old, mu_old):
    K = mu_old.shape[-2]
    q_state, p_state = oneshot_state.forward(ob, mu_old, K)
    state = q_state['zs'].value ## S * B * N * K
    q_angle, p_angle = enc_angle(ob, state, mu_old)
    log_q_f = q_state['zs'].log_prob + q_angle['angles'].log_prob.sum(-1)
    log_p_f = p_state['zs'].log_prob + p_angle['angles'].log_prob.sum(-1)
    angle = q_angle['angles'].value * 2 * math.pi
    p_recon_f = dec_x.forward(ob, state, angle, mu_old)
    log_recon_f = p_recon_f['likelihood'].log_prob.sum(-1)
    log_w_f = log_recon_f.detach() + log_p_f - log_q_f
    ## backward
    beta_old = angle_old / (2*math.pi)
    q_angle_old, p_angle_old = enc_angle(ob, state_old, mu_old)
    log_p_b = cat(probs=p_state['zs'].dist.probs).log_prob(state_old) + Beta(p_angle_old['angles'].dist.concentration1, p_angle_old['angles'].dist.concentration0).log_prob(beta_old).sum(-1)
    log_q_b = cat(probs=q_state['zs'].dist.probs).log_prob(state_old) + Beta(q_angle_old['angles'].dist.concentration1, q_angle_old['angles'].dist.concentration0).log_prob(beta_old).sum(-1)
    p_recon_b = dec_x.forward(ob, state_old, angle_old, mu_old)
    log_recon_b = p_recon_b['likelihood'].log_prob.sum(-1)
    log_w_b = log_recon_b.detach() + log_p_b - log_q_b
    w, eubo, elbo, theta_loss = Grad_phi(log_w_f, log_w_b, log_recon_f)
    return state, angle, w, eubo, elbo, theta_loss, q_state, q_angle, p_recon_f

def Grad_phi(log_w_f, log_w_b, log_recon_f):
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
    theta_loss = (w * (- log_recon_f)).sum(0).sum(-1).mean()
    elbo = log_w_f.sum(-1).mean().detach()

    return w, eubo, elbo, theta_loss