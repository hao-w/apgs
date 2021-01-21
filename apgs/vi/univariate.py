import torch
from torch.distributions.normal import Normal
import pandas as pd
import time
import torch.nn.functional as F
from models import init_kernels


def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
        
        
def train(grad_steps, objective, target_mean, lr, num_samples, num_dims, num_hiddens, CUDA, device, reparameterized=False):
    time_start = time.time()
    output = pd.DataFrame(index=range(grad_steps), columns=['ess_f', 'ex_kl', 'in_kl'])
    target, proposal = init_densities(target_mean, num_dims, CUDA, device, optimized=False)
    fk, bk = init_kernels(num_hiddens, CUDA, device, reparameterized=reparameterized)
    optimizer = torch.optim.Adam(list(fk.parameters())+list(bk.parameters()), lr=lr)
    for n in range(grad_steps):
        optimizer.zero_grad()
        loss, ess_f, ex_kl, in_kl = objective(target, proposal, fk, bk, num_samples)
        loss.backward()
        optimizer.step()
        output['ess_f'][n] = ess_f.cpu().data.numpy()
        output['ex_kl'][n] = ex_kl.cpu().data.numpy()
        output['in_kl'][n] = in_kl.cpu().data.numpy()  
    time_end = time.time()
    print('grad steps=%d, samples=%d (%ds)' % (grad_steps, num_samples, time_end - time_start))
    return output


def init_densities(target_mean, num_dims, CUDA, device, optimized=False):
    """
    initialize a 1D initial density and 1D target density
    """
    target = {'loc': torch.zeros(num_dims), 
             'log_scale': torch.zeros(num_dims)}
    proposal = {'loc': target_mean * torch.ones(num_dims),
           'log_scale': torch.log(2.0 * torch.ones(num_dims))}    
    if CUDA:
        with torch.cuda.device(device):
            for key in target.keys():
                target[key] = target[key].cuda()
            for key in proposal.keys():
                proposal[key] = proposal[key].cuda()         
    if optimized:
        for key in target.keys():
            target[key] = torch.nn.Parameter(target[key])
        for key in proposal.keys():
            proposal[key] = torch.nn.Parameter(proposal[key])
            
    return target, proposal

def loss_rhs(target, proposal, fk, bk, num_samples):
    """
    this loss function employs the following gradient estimators:
    For phi
    - \nabla_\phi KL (\pi_2(x_2) r_\theta(x_1 | x2) || \pi_1(x1) q_\phi(x_2 | x_1))
    For theta
    - \nabla_\theta KL (\pi_1(x1) q_\phi(x_2 | x_1) || \pi_2(x_2) r_\theta(x_1 | x2))
    so that both of them do not require neither reparameterization nor score function
    """
    p = Normal(target['loc'], target['log_scale'].exp())
    q = Normal(proposal['loc'], proposal['log_scale'].exp())
    x1 = q.sample((num_samples,))
    log_q = q.log_prob(x1)
    x2, log_f, mu_x2, sigma_x2 = fk.forward(x=x1, sampled=True)
    log_p = p.log_prob(x2)
    log_b, mu_x1, sigma_x1 = bk.forward(x=x2, sampled=False, samples_old=x1)
    log_w = (log_p + log_b - log_q - log_f).sum(-1).detach()
    w = F.softmax(log_w, 0)
    ess_f = 1.0 / (w**2).sum(0)
    loss_f = (w * ( - log_f.sum(-1))).sum(0)
    loss_b = - log_b.mean()
    ex_kl = - log_w.mean()
    in_kl = (w * log_w).sum()
    return loss_f + loss_b, ess_f, ex_kl, in_kl

def loss_inc(target, proposal, fk, bk, num_samples):
    """
    this loss function only targets the inclusive KL divergence:
    - \nabla_{\phi,\theta} KL (\pi_2(x_2) r_\theta(x_1 | x2) || \pi_1(x1) q_\phi(x_2 | x_1))
    """
    p = Normal(target['loc'], target['log_scale'].exp())
    q = Normal(proposal['loc'], proposal['log_scale'].exp())
    x1 = q.sample((num_samples,))
    log_q = q.log_prob(x1)
    x2, log_f, mu_x2, sigma_x2 = fk.forward(x=x1, sampled=True)
    log_p = p.log_prob(x2)
    log_b, mu_x1, sigma_x1 = bk.forward(x=x2, sampled=False, samples_old=x1)
    log_w = (log_p + log_b - log_q - log_f).sum(-1).detach()
    w = F.softmax(log_w, 0)
    ess_f = 1.0 / (w**2).sum(0)
    loss_f = (w * ( - log_f.sum(-1))).sum(0)
    assert log_b.shape == (num_samples, 1), "ERROR"
    loss_b = (w * ((log_w+1) * log_b.squeeze(-1))).sum(0)
    ex_kl = - log_w.mean()
    in_kl = (w * log_w).sum()
    return loss_f + loss_b, ess_f, ex_kl, in_kl

def loss_exc(target, proposal, fk, bk, num_samples):
    """
    this loss function only targets the exclusive KL divergence:
    - \nabla_{\phi,\theta} KL (\pi_1(x1) q_\phi(x_2 | x_1) || \pi_2(x_2) r_\theta(x_1 | x2) )
    """
    p = Normal(target['loc'], target['log_scale'].exp())
    q = Normal(proposal['loc'], proposal['log_scale'].exp())
    x1 = q.sample((num_samples,))
    log_q = q.log_prob(x1)
    x2, log_f, mu_x2, sigma_x2 = fk.forward(x=x1, sampled=True)
    log_p = p.log_prob(x2)
    log_b, mu_x1, sigma_x1 = bk.forward(x=x2, sampled=False, samples_old=x1)
    log_w = (log_p + log_b - log_q - log_f).squeeze(-1).detach()
    w = F.softmax(log_w, 0)
    ess_f = 1.0 / (w**2).sum(0)
    loss_b = (- log_b).mean()
    loss_f = ((- log_w + 1) * log_f.squeeze(-1)).mean()
    ex_kl = - log_w.mean()
    in_kl = (w * log_w).sum()
    return loss_f+loss_b, ess_f, ex_kl, in_kl

def loss_exc_reparam(target, proposal, fk, bk, num_samples):
    """
    this loss function only targets the exclusive KL divergence:
    - \nabla_{\phi,\theta} KL (\pi_1(x1) q_\phi(x_2 | x_1) || \pi_2(x_2) r_\theta(x_1 | x2) )
    with q_\phi reparameterized
    """
    p = Normal(target['loc'], target['log_scale'].exp())
    q = Normal(proposal['loc'], proposal['log_scale'].exp())
    x1 = q.sample((num_samples,))
    log_q = q.log_prob(x1)
    x2, log_f, mu_x2, sigma_x2 = fk.forward(x=x1, sampled=True)
    log_p = p.log_prob(x2)
    log_b, mu_x1, sigma_x1 = bk.forward(x=x2, sampled=False, samples_old=x1)
    log_w = (log_p + log_b - log_q - log_f).squeeze(-1)
    w = F.softmax(log_w, 0).detach()
    ess_f = 1.0 / (w**2).sum(0)
    loss = (- log_w).mean()
    ex_kl = - log_w.mean().detach()
    in_kl = (w * log_w).sum().detach()
    return loss, ess_f, ex_kl, in_kl

import numpy as np
import matplotlib.pyplot as plt
def plot_convergence(output, iters, num_samples):
    fs = 4
    fig = plt.figure(figsize=(fs*2.5,fs))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(output['ex_kl'], color='#009988', alpha=0.5, label=r'$\mathrm{KL} (\pi_1(x_1) \: q_\phi (x_2 | x_1) \: || \: \pi_2(x_2) \: r_\theta(x_1 | x_2))$')
    ax1.plot(output['in_kl'], color='#AA3377', alpha=0.5, label=r'$\mathrm{KL} (\pi_2(x_2) \: r_\theta(x_1 | x_2) \: || \: \pi_1(x_1) \: q_\phi (x_2 | x_1))$')
    ax1.plot(np.zeros(iters), label='zero constant')
    ax1.set_ylim(-10, 10)
    ax1.set_xlabel('Gradient Steps')
    ax1.legend(fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(output['ess_f'], label='ESS with %d samples' % num_samples)
    ax2.legend(fontsize=10)
    ax2.set_xlabel('Gradient Steps')
#     plt.savefig('results-loss%d-%dsamples-%dsteps.png' % (obj, num_samples, grad_steps))