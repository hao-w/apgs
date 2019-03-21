import time
import torch
from torch import logsumexp
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from utils import *


def mc(q_mu, q_sigma, p_mu, p_sigma, num_samples, alpha=None):
    q = Normal(q_mu, q_sigma)
    xs = q.rsample((num_samples,)) ## reparam sampler
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    log_weights = log_p - log_q
    weights = F.softmax(log_weights, 0).detach()
    ess = 1. / (weights ** 2).sum()
    eubo = (weights * log_weights).sum()
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean()
    loss = -elbo
    return loss, eubo, elbo, iwelbo, ess

def iwae(q_mu, q_sigma, p_mu, p_sigma, num_samples, alpha=None):
    q = Normal(q_mu, q_sigma)
    xs = q.rsample((num_samples,)) ## reparam sampler
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    log_weights = log_p - log_q
    weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
    ess = 1. / (weights ** 2).sum()
    eubo = torch.mul(weights, log_weights).sum()
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean()
    estor = torch.mul(weights, log_weights).sum()
    loss = - estor
    return loss, eubo, elbo, iwelbo, ess

def dreg(q_mu, q_sigma, p_mu, p_sigma, num_samples, alpha):
    """
    alpha = 1 : RWS - DReG
    alpha = 0 : IWAE - DReG
    alpha = 0.5 : STL
    """
    q = Normal(q_mu, q_sigma)
    dq = Normal(q_mu.detach(), q_sigma.detach())
    xs = q.rsample((num_samples,))
    log_p = (-1.0 / ((p_sigma**2) *2.0)) * ((xs - p_mu) ** 2)
    log_q = dq.log_prob(xs)
    log_weights = log_p - log_q
    weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
    estor_w = alpha * weights + (1 - 2 * alpha) * (weights ** 2)
    ess = 1. / (weights ** 2).sum()
    estor = torch.mul(estor_w, log_weights).sum()
    eubo = torch.mul(weights, log_weights).sum()
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean()
    loss = - estor
    return loss, eubo, elbo, iwelbo, ess

def rws(q_mu, q_sigma, p_mu, p_sigma, num_samples, alpha=None):
    q = Normal(q_mu, q_sigma)
    xs = q.sample((num_samples,)) ## nonreparam sampler
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    log_weights = log_p - log_q
    weights = F.softmax(log_weights, 0).detach()
    ess = 1. / (weights ** 2).sum()
    eubo = (weights * log_weights).sum()
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean()
    loss = eubo
    return loss, eubo, elbo, iwelbo, ess
