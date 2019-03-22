import time
import torch
from torch import logsumexp
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from utils import *


def mc(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample=True, num_batches=None):
    q = Normal(q_mu, q_sigma)
    if num_batches == None:
        xs = q.rsample((num_samples,)) ## reparam sampler
    else:
        xs = q.rsample((num_samples, num_batches))
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    if joint_sample:
        log_p_joint = log_p.sum(-1)
        log_joint = log_q.sum(-1)
    log_weights = log_p_joint - log_q_joint
    weights = F.softmax(log_weights, 0).detach()
    ess = 1. / (weights ** 2).sum(0)
    eubo = (weights * log_weights).sum(0)
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean(0)
    loss = - elbo
    return loss, eubo, elbo, iwelbo, ess

def iwae(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample=True, num_batches=None):
    q = Normal(q_mu, q_sigma)
    if num_batches == None:
        xs = q.rsample((num_samples,)) ## reparam sampler
    else:
        xs = q.rsample((num_samples, num_batches))
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    if joint_sample:
        log_p_joint = log_p.sum(-1)
        log_joint = log_q.sum(-1)
    log_weights = log_p_joint - log_q_joint
    log_weights = log_p - log_q
    weights = F.softmax(log_weights, 0).detach()
    ess = 1. / (weights ** 2).sum(0)
    eubo = (weights * log_weights).sum(0)
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean(0)

    loss = - iwelbo
    return loss, eubo, elbo, iwelbo, ess

def driwae(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample=True, num_batches=None):

    q = Normal(q_mu, q_sigma)
    dq = Normal(q_mu.detach(), q_sigma.detach())
    if num_batches == None:
        xs = q.rsample((num_samples,))
    else:
        xs = q.rsample((num_samples, num_batches))
    log_p = (-1.0 / ((p_sigma**2) *2.0)) * ((xs - p_mu) ** 2)
    log_q = dq.log_prob(xs)
    log_weights = log_p - log_q
    weights = F.softmax(log_weights, 0).detach()
    ess = 1. / (weights ** 2).sum(0)
    estor = ((weights ** 2) * log_weights).sum(0)
    eubo = (weights * log_weights).sum(0)
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean(0)
    loss = - estor
    return loss, eubo, elbo, iwelbo, ess

def drrws(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample=True, num_batches=None):
    q = Normal(q_mu, q_sigma)
    dq = Normal(q_mu.detach(), q_sigma.detach())
    if num_batches == None:
        xs = q.rsample((num_samples,))
    else:
        xs = q.rsample((num_samples, num_batches))
    log_p = (-1.0 / ((p_sigma**2) *2.0)) * ((xs - p_mu) ** 2)
    log_q = dq.log_prob(xs)
    log_weights = log_p - log_q
    weights = F.softmax(log_weights, 0).detach()
    estor_w = weights - weights ** 2
    ess = 1. / (weights ** 2).sum(0)
    estor = (estor_w * log_weights).sum(0)
    eubo = (weights * log_weights).sum(0)
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean(0)
    loss = - estor
    return loss, eubo, elbo, iwelbo, ess

def rws(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample=True, num_batches=None):
    q = Normal(q_mu, q_sigma)
    if num_batches == None:
        xs = q.sample((num_samples,)) ## nonreparam sampler
    else:
        xs = q.sample((num_samples, num_batches)) ## nonreparam sampler
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    log_weights = log_p - log_q
    weights = F.softmax(log_weights, 0).detach()
    ess = 1. / (weights ** 2).sum(0)
    eubo = (weights * log_weights).sum(0)
    iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
    elbo = log_weights.mean(0)
    loss = eubo
    return loss, eubo, elbo, iwelbo, ess
