import torch
from torch.distributions.normal import Normal
from utils  import kls_normal_normal, resample
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import init_kernels, Bivariate_Gaussian
import time
import numpy as np

def train(objective, num_hiddens, iters, num_samples, num_sweeps, lr, mu1, mu2, sigma1, sigma2, rho, CUDA, device, yscale):
    """
    objectives: inckl_2k, inckl_4k, exckl_4k_reparam
    """
    bivg = Bivariate_Gaussian(mu1, mu2, sigma1, sigma2, rho, CUDA=CUDA, device=device)
    if objective == 'inckl_2k':
        q1, q2 = init_kernels(num_hiddens, CUDA, device)
        optimizer = torch.optim.SGD(list(q1.parameters())+list(q2.parameters()), lr=lr)
        logging = {}
        time_start = time.time()
        for i in range(iters):
            optimizer.zero_grad()
            trace = vgibbs(q1, q2, bivg, num_sweeps, num_samples)
            trace['loss'].backward()
            optimizer.step()
            for key in trace.keys():
                if key != 'loss':
                    if key in logging:
                        logging[key].append(trace[key][-1])
                    else:
                        logging[key] = [trace[key][-1]]
            if (i+1) % 500 == 0:
                time_end = time.time()
                print('step=%d, in_kl1=%.2f, in_kl2=%.2f, ex_kl1=%.2f, ex_kl2=%.2f, ess=%.2f (%ds)' % (i+1, trace['in_kl1'][-1], trace['in_kl2'][-1], trace['ex_kl1'][-1], trace['ex_kl2'][-1], trace['ess'][-1], time_end - time_start))
                time_start = time.time()
    elif objective == 'inckl_4k':
        loss_func = vgibbs2
        q1f, q2f = init_kernels(num_hiddens, CUDA, device)
        q1b, q2b = init_kernels(num_hiddens, CUDA, device)
        optimizer = torch.optim.SGD(list(q1f.parameters())+list(q2f.parameters())+list(q1b.parameters())+list(q2b.parameters()), lr=lr)
        logging = {}
        time_start = time.time()
        for i in range(iters):
            optimizer.zero_grad()
            trace = vgibbs2(q1f, q2f, q1b, q2b, bivg, num_sweeps, num_samples)
            trace['loss'].backward()
            optimizer.step()
            for key in trace.keys():
                if key != 'loss':
                    if key in logging:
                        logging[key].append(trace[key][-1])
                    else:
                        logging[key] = [trace[key][-1]]
            if (i+1) % 500 == 0:
                time_end = time.time()
                print('step=%d, in_kl1=%.2f, in_kl2=%.2f, ex_kl1=%.2f, ex_kl2=%.2f, ess=%.2f (%ds)' % (i+1, trace['in_kl1'][-1], trace['in_kl2'][-1], trace['ex_kl1'][-1], trace['ex_kl2'][-1], trace['ess'][-1], time_end - time_start))
                time_start = time.time()
    elif objective == 'exckl_4k_reparam':
        loss_func = vgibbs3
        q1f, q2f = init_kernels(num_hiddens, CUDA, device, reparameterized=True)
        q1b, q2b = init_kernels(num_hiddens, CUDA, device)
        optimizer = torch.optim.SGD(list(q1f.parameters())+list(q2f.parameters())+list(q1b.parameters())+list(q2b.parameters()), lr=lr)
        logging = {}
        time_start = time.time()
        for i in range(iters):
            optimizer.zero_grad()
            trace = vgibbs3(q1f, q2f, q1b, q2b, bivg, num_sweeps, num_samples)
            trace['loss'].backward()
            optimizer.step()
            for key in trace.keys():
                if key != 'loss':
                    if key in logging:
                        logging[key].append(trace[key][-1])
                    else:
                        logging[key] = [trace[key][-1]]
            if (i+1) % 500 == 0:
                time_end = time.time()
                print('step=%d, in_kl1=%.2f, in_kl2=%.2f, ex_kl1=%.2f, ex_kl2=%.2f, ess=%.2f (%ds)' % (i+1, trace['in_kl1'][-1], trace['in_kl2'][-1], trace['ex_kl1'][-1], trace['ex_kl2'][-1], trace['ess'][-1], time_end - time_start))
                time_start = time.time()
    else:
        raise NameError
    plot_convergence(logging, yscale=yscale)
    
    

        
def vgibbs(q1, q2, bivg, num_sweeps, num_samples):
    """
    variational gibbs sampler with 2 kernels
    """
    trace = {'ess' : [], 'loss' : 0.0, 'in_kl1': [], 'in_kl2' : [], 'ex_kl1': [], 'ex_kl2' : [], 'db1' : [], 'db2' : [], 'density': []}
    x, log_q0 = bivg.sample_prior(num_samples)
    log_p = bivg.dist.log_prob(x)
    w0 = F.softmax(log_p - log_q0, 0).detach()
    x = resample(var=x, weights=w0)
    for k in range(num_sweeps):
#         print('sweep=%d' % (k+1))
        x1_old, x2_old = x[:, 0].unsqueeze(-1), x[:, 1].unsqueeze(-1)
        x1_sampled, log_q1_f, q1_mu, q1_sigma = q1.forward(x2_old)
        log_p1_f = bivg.dist.log_prob(torch.cat((x1_sampled, x2_old), -1))
        log_p1_b = bivg.dist.log_prob(torch.cat((x1_old, x2_old), -1))
        log_q1_b, _, _ = q1.forward(x2_old, sampled=False, samples_old=x1_old)
        log_w1 = log_p1_f.squeeze() - log_p1_b.squeeze() + log_q1_b.squeeze() - log_q1_f.squeeze()
        w1 = F.softmax(log_w1, 0).detach()
        trace['loss'] += (w1 * (- log_q1_f.squeeze())).sum(0)
        true_mu1, true_sigma1 = bivg.conditional_params(x2_old, cond='x2')
        in_kl1, ex_kl1 = kls_normal_normal(true_mu1, true_sigma1, q1_mu, q1_sigma)
        trace['in_kl1'].append(in_kl1)
        trace['ex_kl1'].append(ex_kl1)
        trace['db1'].append((log_q1_f - log_q1_b).mean().detach())
        x = resample(var=torch.cat((x1_sampled, x2_old), -1), weights=w1)
        
        x1_old, x2_old = x[:, 0].unsqueeze(-1), x[:, 1].unsqueeze(-1)
        x2_sampled, log_q2_f, q2_mu, q2_sigma = q2.forward(x1_old)
        log_p2_f = bivg.dist.log_prob(torch.cat((x1_old, x2_sampled), -1))
        log_p2_b = bivg.dist.log_prob(torch.cat((x1_old, x2_old), -1))
        log_q2_b, _, _ = q2.forward(x1_old, sampled=False, samples_old=x2_old)
        log_w2 = log_p2_f.squeeze() - log_p2_b.squeeze() + log_q2_b.squeeze() - log_q2_f.squeeze()
        w2 = F.softmax(log_w2, 0).detach()
        trace['loss'] += (w2 * (- log_q2_f.squeeze())).sum(0)
        trace['ess'].append(1. / (w2**2).sum(0))
        true_mu2, true_sigma2 = bivg.conditional_params(x1_old, cond='x1')
        in_kl2, ex_kl2 = kls_normal_normal(true_mu2, true_sigma2, q2_mu, q2_sigma)
        trace['in_kl2'].append(in_kl2)
        trace['ex_kl2'].append(ex_kl2)
        trace['db2'].append((log_q2_f - log_q2_b).mean().detach())
        trace['density'].append(log_p2_f.mean().detach())
        x = resample(var=torch.cat((x1_old, x2_sampled), -1), weights=w2)
    return trace

def vgibbs2(q1f, q2f, q1b, q2b, bivg, num_sweeps, num_samples):
    """
    variational gibbs sampler with 4 kernels
    """
    trace = {'ess' : [], 'loss' : 0.0, 'in_kl1': [], 'in_kl2' : [], 'ex_kl1': [], 'ex_kl2' : [], 'db1' : [], 'db2' : [], 'density' : []}
    x, log_q0 = bivg.sample_prior(num_samples)
    log_p = bivg.dist.log_prob(x)
    w0 = F.softmax(log_p - log_q0, 0).detach()
    x = resample(var=x, weights=w0)
    for k in range(num_sweeps):
#         print('sweep=%d' % (k+1))
        x1_old, x2_old = x[:, 0].unsqueeze(-1), x[:, 1].unsqueeze(-1)
        x1_sampled, log_q1_f, q1_mu, q1_sigma = q1f.forward(x2_old)
        log_p1_f = bivg.dist.log_prob(torch.cat((x1_sampled, x2_old), -1))
        log_p1_b = bivg.dist.log_prob(torch.cat((x1_old, x2_old), -1))
        log_q1_b, _, _ = q1b.forward(x2_old, sampled=False, samples_old=x1_old)
        log_w1 = (log_p1_f.squeeze() - log_p1_b.squeeze() + log_q1_b.squeeze() - log_q1_f.squeeze()).detach()
        w1 = F.softmax(log_w1, 0)
        trace['loss'] += (w1 * (- log_q1_f.squeeze())).sum(0) + (w1 * ((log_w1 + 1) * log_q1_b.squeeze())).sum(0)
        true_mu1, true_sigma1 = bivg.conditional_params(x2_old, cond='x2')
        in_kl1, ex_kl1 = kls_normal_normal(true_mu1, true_sigma1, q1_mu, q1_sigma)
        trace['in_kl1'].append(in_kl1)
        trace['ex_kl1'].append(ex_kl1)
        trace['db1'].append((log_q1_f - log_q1_b).mean().detach())
        x = resample(var=torch.cat((x1_sampled, x2_old), -1), weights=w1)
        
        x1_old, x2_old = x[:, 0].unsqueeze(-1), x[:, 1].unsqueeze(-1)
        x2_sampled, log_q2_f, q2_mu, q2_sigma = q2f.forward(x1_old)
        log_p2_f = bivg.dist.log_prob(torch.cat((x1_old, x2_sampled), -1))
        log_p2_b = bivg.dist.log_prob(torch.cat((x1_old, x2_old), -1))
        log_q2_b, _, _ = q2b.forward(x1_old, sampled=False, samples_old=x2_old)
        log_w2 = (log_p2_f.squeeze() - log_p2_b.squeeze() + log_q2_b.squeeze() - log_q2_f.squeeze()).detach()
        w2 = F.softmax(log_w2, 0)
        trace['loss'] += (w2 * (- log_q2_f.squeeze())).sum(0) + (w2 * ((log_w2 + 1) * log_q2_b.squeeze())).sum(0)
        trace['ess'].append(1. / (w2**2).sum(0))
        true_mu2, true_sigma2 = bivg.conditional_params(x1_old, cond='x1')
        in_kl2, ex_kl2 = kls_normal_normal(true_mu2, true_sigma2, q2_mu, q2_sigma)
        trace['in_kl2'].append(in_kl2)
        trace['ex_kl2'].append(ex_kl2)
        trace['db2'].append((log_q2_f - log_q2_b).mean().detach())
        trace['density'].append(log_p2_f.mean().detach())
        x = resample(var=torch.cat((x1_old, x2_sampled), -1), weights=w2)
    return trace

def vgibbs3(q1f, q2f, q1b, q2b, bivg, num_sweeps, num_samples):
    """
    variational gibbs sampler with 4 kernels with reparameterization
    """
    trace = {'ess' : [], 'loss' : 0.0, 'in_kl1': [], 'in_kl2' : [], 'ex_kl1': [], 'ex_kl2' : [], 'db1' : [], 'db2' : [], 'density' : []}
    x, log_q0 = bivg.sample_prior(num_samples)
    log_p = bivg.dist.log_prob(x)
    w0 = F.softmax(log_p - log_q0, 0).detach()
    x = resample(var=x, weights=w0)
    for k in range(num_sweeps):
#         print('sweep=%d' % (k+1))
        x1_old, x2_old = x[:, 0].unsqueeze(-1).detach(), x[:, 1].unsqueeze(-1).detach()
        x1_sampled, log_q1_f, q1_mu, q1_sigma = q1f.forward(x2_old)
        log_p1_f = bivg.dist.log_prob(torch.cat((x1_sampled, x2_old), -1))
        log_p1_b = bivg.dist.log_prob(torch.cat((x1_old, x2_old), -1))
        log_q1_b, _, _ = q1b.forward(x2_old, sampled=False, samples_old=x1_old)
        log_w1 = log_p1_f.squeeze() - log_p1_b.squeeze() + log_q1_b.squeeze() - log_q1_f.squeeze()
        w1 = F.softmax(log_w1, 0).detach()
        trace['loss'] += (- log_w1).mean()
        true_mu1, true_sigma1 = bivg.conditional_params(x2_old, cond='x2')
        in_kl1, ex_kl1 = kls_normal_normal(true_mu1, true_sigma1, q1_mu, q1_sigma)
        trace['in_kl1'].append(in_kl1)
        trace['ex_kl1'].append(ex_kl1)
        trace['db1'].append((log_q1_f - log_q1_b).mean().detach())
        x = resample(var=torch.cat((x1_sampled, x2_old), -1), weights=w1)
        
        x1_old, x2_old = x[:, 0].unsqueeze(-1).detach(), x[:, 1].unsqueeze(-1).detach()
        x2_sampled, log_q2_f, q2_mu, q2_sigma = q2f.forward(x1_old)
        log_p2_f = bivg.dist.log_prob(torch.cat((x1_old, x2_sampled), -1))
        log_p2_b = bivg.dist.log_prob(torch.cat((x1_old, x2_old), -1))
        log_q2_b, _, _ = q2b.forward(x1_old, sampled=False, samples_old=x2_old)
        log_w2 = log_p2_f.squeeze() - log_p2_b.squeeze() + log_q2_b.squeeze() - log_q2_f.squeeze()
        w2 = F.softmax(log_w2, 0).detach()
        trace['loss'] += (- log_w2).mean()
        trace['ess'].append(1. / (w2**2).sum(0))
        true_mu2, true_sigma2 = bivg.conditional_params(x1_old, cond='x1')
        in_kl2, ex_kl2 = kls_normal_normal(true_mu2, true_sigma2, q2_mu, q2_sigma)
        trace['in_kl2'].append(in_kl2)
        trace['ex_kl2'].append(ex_kl2)
        trace['db2'].append((log_q2_f - log_q2_b).mean().detach())
        trace['density'].append(log_p2_f.mean().detach())
        x = resample(var=torch.cat((x1_old, x2_sampled), -1), weights=w2)
    return trace

def Gibbs(bivg, steps, sampling=True):
    """
    perform closed-form Gibbs sampling
    return the new corrdinate after each sweep
    """
    ## randomly start from a point in 2D space
    init = Normal(torch.ones(2) * -5, torch.ones(2) * 2).sample().unsqueeze(0)
#     Mu, Sigma = bg.Joint()
#     INIT = Normal(Mu, Sigma).sample().unsqueeze(0)
    x2 = init[:, 1]
    x1 = init[:, 0]
    updates = []
    updates.append(init)
    for i in range(steps):
        ## update x1
        cond_x1_mu, cond_x1_sigma = bivg.conditional_params(x2, cond='x2')
        kernel_x1 = Normal(cond_x1_mu, cond_x1_sigma)
        if sampling:
            x1 = kernel_x1.sample()
        else:
            x1 = kernel_x1.mean
    #     updates.append(torch.cat((x1, x2), 0).unsqueeze(0))
        ## update x2
        cond_x2_mu, cond_x2_sigma = bivg.conditional_params(x1, cond='x1')
        kernel_x2 = Normal(cond_x2_mu, cond_x2_sigma)
        if sampling:
            x2 = kernel_x2.sample()
        else:
            x2 = kernel_x2.mean
        updates.append(torch.cat((x1, x2), 0).unsqueeze(0))
    return torch.cat(updates, 0)

def plot_convergence(trace, yscale):
    fig = plt.figure(figsize=(24,6))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot((np.array(trace['in_kl1']) + np.array(trace['ex_kl1'])) / 2, label=r'$SymKL(p(x_1 | x_2) || q_\phi(x_1 | x_2))$')
    ax1.plot((np.array(trace['in_kl2']) + np.array(trace['ex_kl2'])) / 2, label=r'$SymKL(p(x_2 | x_1) || r_\theta(x_2 | x_1))$')
    ax1.set_yscale(yscale)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(trace['db1'], label=r'$\mathbb{E}[\frac{p(x_2) q_\phi (x_1 | x_2)} {p(x_2) r_\theta (x_1 | x_2)}]$')
    ax2.plot(trace['db2'], label=r'$\mathbb{E}[\frac{p(x_1) q_\phi (x_2 | x_1)} {p(x_1) r_\theta (x_2 | x_1)}]$')
    ax2.set_ylim([-9, 10.0])
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(trace['ess'], label='ESS')
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(trace['density'], label=r'$\log p(x_1, x_2)$')
    ax1.legend(fontsize=12)
    ax2.legend(fontsize=16)
    ax3.legend(fontsize=12)
    ax4.legend(fontsize=14)
    
#     def fb(self, q, x_cond, x_cond_name, x_old, sampled=True, obj='ag'):
#         """
#         """
#         x_new, log_q_f, _, _ = q.forward(x_cond, sampled=sampled) ## sample from approximate conditional
#         true_post_mu, true_post_sigma = self.conditional(x_cond, cond=x_cond_name) ## true conditional kernels
#         log_p_f = self.log_pdf_gamma(x_new, true_post_mu, true_post_sigma) 
#         log_q_b = q.backward(x_cond, x_old)
#         log_p_b = self.log_pdf_gamma(x_old, true_post_mu, true_post_sigma)
#         log_w_f = log_p_f.sum(-1) - log_q_f.sum(-1)
#         log_w_b = log_p_b.sum(-1) - log_q_b.sum(-1)

#         if obj =='ag':
#             log_w = log_w_f - log_w_b
#             w = F.softmax(log_w, 0).detach()
#             loss = (w * log_w_f).sum(0).mean().unsqueeze(0)
#         elif obj =='kl':
#             log_w = log_w_f - log_w_b
#             w = F.softmax(log_w, 0).detach()
#             loss = (w * ((log_w.detach() + c + 1) * log_q_b - log_q_f)).sum(0).mean().unsqueeze(0)
#         ess = (1. / (w ** 2).sum(0)).mean().unsqueeze(0)
#         return loss, ess, x_new, log_w, w


#     def train(self, q_x1, q_x2, mcmc_steps, num_samples, sampled=True, obj='ag'):
#         """
#         obj = 'ag' : the objective proposed in our paper
#         obj= 'kl' : the KL(p(z_t) q(z_t-1 | z_t) || p(z_t-1) q(z_t | z_t-1))
#         """
#         losss = []
#         esss = []
#         ## start with sampling x1 from the prior
#         x1, log_p_x1_f = q_x1.sample_prior(num_samples)
#         x2, log_q_x2, _, _ = q_x2.forward(x1, sampled=sampled)
#         r_mu, r_sigma = self.conditional(x1, cond='x1')
#         log_p_x2_f = self.log_pdf_gamma(x2, r_mu, r_sigma)
#         log_w = log_p_x2_f.sum(-1).detach() - log_q_x2.sum(-1)
#         w_x2 = F.softmax(log_w, 0).detach()
#         losss.append((w_x2 * log_w).sum(0).mean().unsqueeze(0))
#         esss.append((1. / (w_x2** 2).sum(0)).mean().unsqueeze(0))
#         for m in range(mcmc_steps):
#             x2 = resample(x2, w_x2) ## resample x2
#             # x1_old = x1
#             loss_x1, ess_x1, x1, log_w_x1, w_x1 = self.fb(q_x1, x2, 'x2', x1, sampled=sampled, obj=obj) ## update x1
#             x1 = resample(x1, w_x1) ## resample x1
#             # x2_old = x2
#             loss_x2, ess_x2, x2, log_w_x2, w_x2 = self.fb(q_x2, x1, 'x1', x2, sampled=sampled, obj=obj) ## update x2
#             losss.append(loss_x1+loss_x2)
#             esss.append((ess_x1 + ess_x2) / 2)
#         return torch.cat(losss, 0).sum(), torch.cat(esss, 0).mean()

#     def fb_test(self, q, x_cond, x_cond_name, x_old, sampled=True):
#         x_new, log_q_f, q_mu, q_sigma = q.forward(x_cond, sampled=sampled)
#         true_post_mu, true_post_sigma = self.conditional(x_cond, cond=x_cond_name)
#         kls = kl_normal_normal(true_post_mu, true_post_sigma, q_mu, q_sigma).mean()

#         log_p_f = self.log_pdf_gamma(x_new, true_post_mu, true_post_sigma)
#         log_q_b = q.backward(x_cond, x_old)
#         log_p_b = self.log_pdf_gamma(x_old, true_post_mu, true_post_sigma)
#         log_w_f = log_p_f.sum(-1) - log_q_f.sum(-1)
#         log_w_b = log_p_b.sum(-1) - log_q_b.sum(-1)
#         log_w = log_w_f - log_w_b
#         w = F.softmax(log_w, 0).detach()
#         return x_new , (w * log_w).sum(0).mean().unsqueeze(0), kls, log_w

#     def test(self, q_x1, q_x2, mcmc_steps, num_samples, sampled=True, init=None):
#         updates = []
#         DBs_q1 = []
#         DBs_q2 = []
#         KLs_q1 = []
#         KLs_q2 = []

#         ELBOs = []
#         ## start with sampling x1 from the prior
#         if init is not None:
#             x1 = init
#         else:
#             x1, log_p_x1_f = q_x1.sample_prior(num_samples)

#         x2, log_q_x2, q_mu, q_sigma = q_x2.forward(x1, sampled=sampled)
#         r_mu, r_sigma = self.conditional(x1, cond='x1')
#         log_p_x2_f = self.log_pdf_gamma(x2, r_mu, r_sigma)
#         log_w = log_p_x2_f.sum(-1).detach() - log_q_x2.sum(-1)
#         r_mu, r_sigma = self.conditional(x1, cond='x1')
#         # kls = kl_normal_normal(r_mu, r_sigma, q_mu, q_sigma).mean()
#         # KLs.append(kls.unsqueeze(0))
#         updates.append(torch.cat((x1, x2), -1))
#         ELBOs.append(log_w.mean().unsqueeze(0))
#         # print('Init: x1=%.4f, x2=%.4f' % (x1, x2))
#         for m in range(mcmc_steps):
#             x1_old = x1
#             x1, db_x1, kls_x1, log_w_x1  = self.fb_test(q_x1, x2, 'x2', x1_old, sampled=sampled) ## update x1
#             DBs_q1.append(db_x1)
#     #         x1 = post_mu_x1
#             x2_old = x2
#             x2, db_x2, kls_x2, log_w_x2 = self.fb_test(q_x2, x1, 'x1', x2_old, sampled=sampled) ## update x2
#             DBs_q2.append(db_x2)
#             KLs_q1.append(kls_x1.unsqueeze(0))
#             KLs_q2.append(kls_x2.unsqueeze(0))
#             updates.append(torch.cat((x1, x2), -1))
#     #         print('step=%d: x1=%.4f, x2=%.4f' % (i, x1, x2))
#             ELBOs.append(ELBOs[-1] + (log_w_x1 + log_w_x2).mean().unsqueeze(0))
#         return torch.cat(updates, 0), [torch.cat(DBs_q1, 0), torch.cat(DBs_q2,0)], [torch.cat(KLs_q1, 0), torch.cat(KLs_q2, 0)], torch.cat(ELBOs, 0)
