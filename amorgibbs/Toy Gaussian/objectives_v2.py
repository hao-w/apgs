import time
import torch
from torch import logsumexp
import torch.nn.functional as F
from torch.distributions.normal import Normal

def mc(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample):
    q = Normal(q_mu, q_sigma)
    xs = q.rsample((num_samples,)) ## reparam sampler
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    if joint_sample:
        log_p_joint = log_p.sum(-1)
        log_q_joint = log_q.sum(-1)
        log_weights = log_p_joint - log_q_joint
        weights = F.softmax(log_weights, 0).detach()
        ess = 1. / (weights ** 2).sum(0)
        elbo = log_weights.mean(0)
        loss = - elbo
    else:
        log_weights = log_p - log_q ## S * B
        weights = F.softmax(log_weights, 0).detach()
        ess = (1. / (weights ** 2).sum(0)).mean()
        elbo = log_weights.mean(0)
        loss = - elbo.sum()
    return loss, ess

def iwae(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample):
    q = Normal(q_mu, q_sigma)
    xs = q.rsample((num_samples,)) ## reparam sampler
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    if joint_sample:
        log_p_joint = log_p.sum(-1)
        log_q_joint = log_q.sum(-1)
        log_weights = log_p_joint - log_q_joint
        weights = F.softmax(log_weights, 0).detach()
        ess = 1. / (weights ** 2).sum(0)
        iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
        loss = - iwelbo
    else:
        log_weights = log_p - log_q
        weights = F.softmax(log_weights, 0).detach()
        ess = (1. / (weights ** 2).sum(0)).mean()
        iwelbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
        loss = - iwelbo.sum()
    return loss, ess

def driwae(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample):
    q = Normal(q_mu, q_sigma)
    dq = Normal(q_mu.detach(), q_sigma.detach())
    xs = q.rsample((num_samples,))
    log_p = (-1.0 / ((p_sigma**2) *2.0)) * ((xs - p_mu) ** 2)
    log_q = dq.log_prob(xs)
    if joint_sample:
        log_p_joint = log_p.sum(-1)
        log_q_joint = log_q.sum(-1)
        log_weights = log_p_joint - log_q_joint
        weights = F.softmax(log_weights, 0).detach()
        ess = 1. / (weights ** 2).sum(0)
        estor = ((weights ** 2) * log_weights).sum(0)
        loss = - estor
    else:
        log_weights = log_p - log_q
        weights = F.softmax(log_weights, 0).detach()
        ess = (1. / (weights ** 2).sum(0)).mean()
        estor = ((weights**2) * log_weights).sum(0)
        loss = - estor.sum()
    return loss, ess

def drrws(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample):
    q = Normal(q_mu, q_sigma)
    dq = Normal(q_mu.detach(), q_sigma.detach())
    xs = q.rsample((num_samples,))
    log_p = (-1.0 / ((p_sigma**2) *2.0)) * ((xs - p_mu) ** 2)
    log_q = dq.log_prob(xs)
    if joint_sample:
        log_p_joint = log_p.sum(-1)
        log_q_joint = log_q.sum(-1)
        log_weights = log_p_joint - log_q_joint
        weights = F.softmax(log_weights, 0).detach()
        ess = 1. / (weights ** 2).sum(0)
        estor_w = weights - weights ** 2
        estor = (estor_w * log_weights).sum(0)
        loss = - estor
    else:
        log_weights = log_p - log_q
        weights = F.softmax(log_weights, 0).detach()
        ess = (1. / (weights ** 2).sum(0)).mean()
        estor_w = weights - weights ** 2
        estor = (estor_w * log_weights).sum(0)
        loss = - estor.sum()
    return loss, ess

def rws(q_mu, q_sigma, p_mu, p_sigma, num_samples, joint_sample):
    q = Normal(q_mu, q_sigma)
    xs = q.sample((num_samples,)) ## nonreparam sampler
    log_p = (-1.0 / ((p_sigma**2) * 2.0)) * ((xs - p_mu) ** 2)
    log_q = q.log_prob(xs)
    if joint_sample:
        log_p_joint = log_p.sum(-1)
        log_q_joint = log_q.sum(-1)
        log_weights = log_p_joint - log_q_joint
        weights = F.softmax(log_weights, 0).detach()
        ess = 1. / (weights ** 2).sum(0)
        eubo = (weights * log_weights).sum(0)
        loss = eubo
    else:
        log_weights = log_p - log_q
        weights = F.softmax(log_weights, 0).detach()
        ess = (1. / (weights ** 2).sum(0)).mean()
        eubo = (weights * log_weights).sum(0)
        loss = eubo.sum()
    return loss, ess
