import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.beta import Beta
import math


class HMC():
    def __init__(self, enc_local, dec, S, B, N, K, D, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, CUDA, device):
        self.S, self.B, self.N, self.K, self.D = S, B, N, K, D        
        self.Sigma = torch.ones(1)
        self.mu = torch.zeros(1)
        self.accept_count = 0.0
        self.smallest_accept_ratio = 0.0
        self.hmc_num_steps = hmc_num_steps
        self.lf_step_size = leapfrog_step_size
        self.lf_num_steps = leapfrog_num_steps
        self.enc_local = enc_local
        self.dec = dec
        if CUDA:
            with torch.cuda.device(device):
                self.Sigma = self.Sigma.cuda()
                self.mu = self.mu.cuda()
                self.uniformer = Uniform(torch.Tensor([0.0]).cuda(), torch.Tensor([1.0]).cuda())
        else:
            self.uniformer = Uniform(torch.Tensor([0.0]), torch.Tensor([1.0]))

        self.gauss_dist = Normal(self.mu, self.Sigma)


    def init_sample(self):
        """
        initialize auxiliary variables from univariate Gaussian
        return r_mu, r_beta
        """
        return self.gauss_dist.sample((self.S, self.B, self.K, self.D, )).squeeze(-1)

    def beta_to_y(self, beta):
        return (beta / (1 - beta)).log()

    def y_to_beta(self, y):
        return 1. / (1 + (- y).exp())
    
    def hmc_sampling(self, x, mu, z, beta, trace):
        """
        change of variables beta  = 1 / (1 + exp(-y))
        y = log (beta / (1 - beta))
        """
        for m in range(self.hmc_num_steps):
            mu = self.metrioplis(x, mu=mu.detach(), z=z.detach(), beta=beta.detach())
            q = self.enc_local(x, mu=mu, K=self.K)
            z = q['states'].value
            beta = q['angles'].value
            log_joint = self.log_joint(x, mu=mu, z=z, beta=beta)
            trace['density'].append(log_joint.unsqueeze(0))

        self.smallest_accept_ratio = (self.accept_count  / self.hmc_num_steps).min().item()
        if self.smallest_accept_ratio > 0.25: # adaptive leapfrog step size
            self.smallest_accept_ratio *= 1.005
        else:
            self.smallest_accept_ratio *= 0.995
        return trace

    def log_joint(self, x, mu, z, beta):
        p = self.dec.forward(x, mu=mu, z=z, beta=beta)
        ll = p['likelihood'].log_prob.sum(-1).sum(-1)
        log_prior_mu = Normal(self.dec.prior_mu_mu, self.dec.prior_mu_sigma).log_prob(mu).sum(-1).sum(-1)
        log_prior_z = cat(probs=self.dec.prior_pi).log_prob(z).sum(-1)
        log_prior_beta = Beta(self.dec.prior_con1, self.dec.prior_con0).log_prob(beta).sum(-1).sum(-1)
        return ll + log_prior_mu + log_prior_z + log_prior_beta

    def log_conditional(self, x, mu, z, beta):
        p = self.dec.forward(x, mu=mu, z=z, beta=beta)
        ll = p['likelihood'].log_prob.sum(-1).sum(-1)
        log_prior_mu = Normal(self.dec.prior_mu_mu, self.dec.prior_mu_sigma).log_prob(mu).sum(-1).sum(-1)
        return ll + log_prior_mu

    def metrioplis(self, x, mu, z, beta):
        r_mu = self.init_sample()
        ## compute hamiltonian given original position and momentum
        H_orig = self.hamiltonian(x, mu=mu, z=z, beta=beta, r_mu=r_mu)
        new_mu, new_r_mu = self.leapfrog(x, mu, z, beta, r_mu)
        ## compute hamiltonian given new proposals
        H_new = self.hamiltonian(x, mu=new_mu, z=z, beta=beta, r_mu=new_r_mu)
        accept_ratio = (H_new - H_orig).exp()
        u_samples = self.uniformer.sample((self.S, self.B, )).squeeze(-1)
        accept_index = (u_samples < accept_ratio)
        accept_index_expand1 = accept_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.K, self.D)
        filtered_mu = new_mu * accept_index_expand1.float() + mu * (~accept_index_expand1).float()
        self.accept_count = self.accept_count + accept_index.float()
        return filtered_mu.detach()

    def leapfrog(self, x, mu, z, beta, r_mu):
        for step in range(self.lf_num_steps):
            mu.requires_grad = True
            log_p = self.log_conditional(x, mu, z, beta)
            log_p.sum().backward(retain_graph=False)
            r_mu = (r_mu + 0.5 * self.lf_step_size * mu.grad).detach()
            mu.requires_grad = True
            log_p = self.log_conditional(x, mu, z, beta)
            log_p.sum().backward(retain_graph=False)
            r_mu = (r_mu + 0.5 * self.lf_step_size * mu.grad).detach()
            mu = mu.detach()
        return mu, r_mu

    def hamiltonian(self, x, mu, z, beta, r_mu):
        """
        compute the Hamiltonian given the position and momntum
        """
        Kp = self.kinetic_energy(r_mu=r_mu)
        Uq = self.log_conditional(x, mu=mu, z=z, beta=beta)
        assert Kp.shape == (self.S, self.B), "ERROR! Kp has unexpected shape."
        assert Uq.shape ==  (self.S, self.B), 'ERROR! Uq has unexpected shape.'
        return Kp + Uq

    def kinetic_energy(self, r_mu):
        """
        r_mu : S * B * K * D
        return - 1/2 * ||r_mu+r_beta||^2
        """
        return - ((r_mu ** 2).sum(-1).sum(-1)) * 0.5