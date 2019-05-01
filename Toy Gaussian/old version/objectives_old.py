import time
import torch
from torch import logsumexp
from torch.distributions.normal import Normal
import numpy as np
from utils import *

def mc(num_samples, q_mu, q_sigma, p_mu, p_sigma2, iterations, optimizer):
    ELBO = []
    Mu = []
    Sigma = []
    Grad_mu = []
    Grad_sigma = []
    ESS = []
    time_start = time.time()
    for i in range(iterations):
        optimizer.zero_grad()
        q = Normal(q_mu, q_sigma)
        xs = q.rsample((num_samples,))
        log_p = (-1.0 / (p_sigma2 *2.0)) * ((xs - p_mu) ** 2)
        log_q = q.log_prob(xs)
        log_weights = log_p - log_q
        weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
        ess = 1. / (weights ** 2).sum()
        estor = log_weights.mean()
        elbo = estor
        loss = - estor
        loss.backward()
        optimizer.step()
        ELBO.append(elbo.item())
        Mu.append(q_mu.item())
        Sigma.append(q_sigma.item())
        Grad_mu.append((- q_mu.grad).item())
        Grad_sigma.append((- q_sigma.grad).item())
        ESS.append(ess.item())
        if i % 1000 == 0:
            time_end = time.time()
            print('iteration:%d, ELBO:%.3f, ESS:%.3f (%ds)' % (i, elbo, ess, (time_end - time_start)))
            time_start = time.time()
    return ELBO, Mu, Sigma, Grad_mu, Grad_sigma, ESS

def iwae(num_samples, q_mu, q_sigma, p_mu, p_sigma2, iterations, optimizer):
    ELBO = []
    Mu = []
    Sigma = []
    Grad_mu = []
    Grad_sigma = []
    ESS = []
    time_start = time.time()
    for i in range(iterations):
        optimizer.zero_grad()
        q = Normal(q_mu, q_sigma)
        xs = q.rsample((num_samples,))
        log_p = (-1.0 / (p_sigma2 *2.0)) * ((xs - p_mu) ** 2)
        log_q = q.log_prob(xs)
        log_weights = log_p - log_q
        weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
        ess = 1. / (weights ** 2).sum()
        elbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
        estor = torch.mul(weights, log_weights).sum()
        loss = - estor
        loss.backward()
        optimizer.step()
        ELBO.append(elbo.item())
        Mu.append(q_mu.item())
        Sigma.append(q_sigma.item())
        Grad_mu.append((- q_mu.grad).item())
        Grad_sigma.append((- q_sigma.grad).item())
        ESS.append(ess.item())
        if i % 1000 == 0:
            time_end = time.time()
            print('iteration:%d, ELBO:%.3f, ESS:%.3f (%ds)' % (i, elbo, ess, (time_end - time_start)))
            time_start = time.time()
    return ELBO, Mu, Sigma, Grad_mu, Grad_sigma, ESS

def dreg(num_samples, q_mu, q_sigma, p_mu, p_sigma2, iterations, optimizer, alpha):
    """
    alpha = 1 : RWS - DReG
    alpha = 0 : IWAE - DReG
    alpha = 0.5 : STL
    """
    EUBO = []
    ELBO = []
    Mu = []
    Sigma = []
    Grad_mu = []
    Grad_sigma = []
    ESS = []
    time_start = time.time()
    for i in range(iterations):
        optimizer.zero_grad()
        q = Normal(q_mu, q_sigma)
        dq = Normal(q_mu.detach(), q_sigma.detach())
        xs = q.rsample((num_samples,))
        log_p = (-1.0 / (p_sigma2 *2.0)) * ((xs - p_mu) ** 2)
        log_dq = dq.log_prob(xs)
        log_weights = log_p - log_dq
        weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
        estor_w = alpha * weights + (1 - 2 * alpha) * (weights ** 2)
        ess = 1. / (weights ** 2).sum()
        estor = (estor_w * log_weights).sum()
        eubo = (weights * log_weights).sum()
        elbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
        loss = - estor
        loss.backward()
        optimizer.step()

        EUBO.append(eubo.item())
        ELBO.append(elbo.item())
        Mu.append(q_mu.item())
        Sigma.append(q_sigma.item())
        Grad_mu.append((- q_mu.grad).item())
        Grad_sigma.append((- q_sigma.grad).item())
        ESS.append(ess.item())
        if i % 10000 == 0:
            time_end = time.time()
            print('iteration:%d, EUBO:%.3f, ELBO:%.3f, ESS:%.3f (%ds)' % (i, eubo, elbo, ess, (time_end - time_start)))
            time_start = time.time()
    return EUBO, ELBO, Mu, Sigma, Grad_mu, Grad_sigma, ESS

def rws(num_samples, q_mu, q_sigma, p_mu, p_sigma2, iterations, optimizer):

    EUBO = []
    ELBO = []
    Mu = []
    Sigma = []
    Grad_mu = []
    Grad_sigma = []
    ESS = []
    time_start = time.time()
    for i in range(iterations):
        optimizer.zero_grad()
        q = Normal(q_mu, q_sigma)
        dq = Normal(q_mu.detach(), q_sigma.detach())
        xs = q.sample((num_samples,))
        log_p = (-1.0 / (p_sigma2 *2.0)) * ((xs - p_mu) ** 2)
        log_q = dq.log_prob(xs)
        log_weights = log_p - log_q
        weights = torch.exp(log_weights - logsumexp(log_weights, dim=0)).detach()
        ess = 1. / (weights ** 2).sum()
        eubo = torch.mul(weights, log_weights).sum()
        elbo = logsumexp(log_weights, 0) - torch.log(torch.FloatTensor([num_samples]))
        eubo.backward()
        optimizer.step()

        EUBO.append(eubo.item())
        ELBO.append(elbo.item())
        Mu.append(q_mu.item())
        Sigma.append(q_sigma.item())
        Grad_mu.append((- q_mu.grad).item())
        Grad_sigma.append((- q_sigma.grad).item())
        ESS.append(ess.item())
        if i % 10000 == 0:
            time_end = time.time()
            print('iteration:%d, EUBO:%.3f, ELBO:%.3f, ESS:%.3f (%ds)' % (i, eubo, elbo, ess, (time_end - time_start)))
            time_start = time.time()
    return EUBO, ELBO, Mu, Sigma, Grad_mu, Grad_sigma, ESS
