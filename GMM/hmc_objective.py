import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from kls_gmm import posterior_z
import time
"""
HMC + RWS by marginalization over discrete variables in GMM problem
"""
################ implementation of HMC
class HMC():
    def __init__(self, generative, S, B, N, K, D, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, CUDA, DEVICE):
        self.S = S
        self.B = B
        self.N = N
        self.K = K
        self.D = D
        self.generative = generative
        self.Sigma = torch.ones(self.D)
        self.mu = torch.zeros(self.D)
        self.accept_count = 0.0
        self.smallest_accept_ratio = 0.0
        self.hmc_num_steps = hmc_num_steps
        self.lf_step_size = leapfrog_step_size
        self.lf_num_steps = leapfrog_num_steps
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.Sigma = self.Sigma.cuda()
                self.mu = self.mu.cuda()
                self.uniformer = Uniform(torch.Tensor([0.0]).cuda(), torch.Tensor([1.0]).cuda())
        else:
            self.uniformer = Uniform(torch.Tensor([0.0]), torch.Tensor([1.0]))

        self.gauss_dist = Normal(self.mu, self.Sigma)

    def init_sample(self):
        """
        initialize auxiliary variables from univariate Gaussian
        return r_tau, r_mu
        """
        return self.gauss_dist.sample((self.S, self.B, self.K, )), self.gauss_dist.sample((self.S, self.B, self.K, ))

    def hmc_sampling(self, ob, log_tau, mu):
        self.accept_count = 0.0
        density_list = []
        for m in range(self.hmc_num_steps):
            log_tau, mu = self.metrioplis(ob=ob, log_tau=log_tau.detach(), mu=mu.detach())
            posterior_logits = posterior_z(ob=ob,
                                           tau=log_tau.exp(),
                                           mu=mu,
                                           prior_pi=self.generative.prior_pi)
            E_z = posterior_logits.exp().mean(0)
            z = cat(logits=posterior_logits).sample()
            log_joint = self.log_joint(ob=ob, z=z, tau=log_tau.exp(), mu=mu)
            density_list.append(log_joint.unsqueeze(0))

        self.smallest_accept_ratio = (self.accept_count / self.hmc_num_steps).min().item()
        if self.smallest_accept_ratio > 0.25: # adaptive leapfrog step size
            self.smallest_accept_ratio *= 1.005
        else:
            self.smallest_accept_ratio *= 0.995
        return log_tau, mu, density_list

    def log_joint(self, ob, z, tau, mu):
        ll = self.generative.log_prob(ob=ob, z=z, tau=tau, mu=mu, aggregate=True)
        log_prior_tau = Gamma(self.generative.prior_alpha, self.generative.prior_beta).log_prob(tau).sum(-1).sum(-1)
        log_prior_mu = Normal(self.generative.prior_mu, 1. / (self.generative.prior_nu * tau).sqrt()).log_prob(mu).sum(-1).sum(-1)
        log_prior_z = cat(probs=self.generative.prior_pi).log_prob(z).sum(-1)
        return (ll + log_prior_tau + log_prior_mu + log_prior_z)



    def metrioplis(self, ob, log_tau, mu):
        r_tau, r_mu = self.init_sample()
        ## compute hamiltonian given original position and momentum
        H_orig = self.hamiltonian(ob=ob, log_tau=log_tau, mu=mu, r_tau=r_tau, r_mu=r_mu)
        new_log_tau, new_mu, new_r_tau, new_r_mu = self.leapfrog(ob, log_tau, mu, r_tau, r_mu)
        ## compute hamiltonian given new proposals
        H_new = self.hamiltonian(ob=ob, log_tau=new_log_tau, mu=new_mu, r_tau=new_r_tau, r_mu=new_r_mu)
        accept_ratio = (H_new - H_orig).exp()
        u_samples = self.uniformer.sample((self.S, self.B, )).squeeze(-1)
        accept_index = (u_samples < accept_ratio)
        # assert accept_index.shape == (self.S, self.B), "ERROR! index has unexpected shape."
        accept_index_expand = accept_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.K, self.D)
        assert accept_index_expand.shape == (self.S, self.B, self.K, self.D), "ERROR! index has unexpected shape."
        filtered_log_tau = new_log_tau * accept_index_expand.float() + log_tau * (~accept_index_expand).float()
        filtered_mu = new_mu * accept_index_expand.float() + mu * (~accept_index_expand).float()
        self.accept_count = self.accept_count + accept_index_expand.float()
        return filtered_log_tau.detach(), filtered_mu.detach()


    def leapfrog(self, ob, log_tau, mu, r_tau, r_mu):
        for step in range(self.lf_num_steps):
            log_tau.requires_grad = True
            mu.requires_grad = True
            log_p = self.log_marginal(ob, log_tau, mu)
            log_p.sum().backward(retain_graph=False)

            r_tau = (r_tau + 0.5 * self.lf_step_size * log_tau.grad).detach()
            r_mu = (r_mu + 0.5 * self.lf_step_size * mu.grad).detach()
            log_tau = (log_tau + self.lf_step_size * r_tau).detach()
            mu = (mu + self.lf_step_size * r_mu).detach()
            log_tau.requires_grad = True
            mu.requires_grad = True
            log_p = self.log_marginal(ob, log_tau, mu)
            log_p.sum().backward(retain_graph=False)
            r_tau = (r_tau + 0.5 * self.lf_step_size * log_tau.grad).detach()
            r_mu = (r_mu + 0.5 * self.lf_step_size * mu.grad).detach()
            log_tau = log_tau.detach()
            mu = mu.detach()
        return log_tau, mu, r_tau, r_mu



    def hamiltonian(self, ob, log_tau, mu, r_tau, r_mu):
        """
        compute the Hamiltonian given the position and momntum
        """
        Kp = self.kinetic_energy(r_tau=r_tau, r_mu=r_mu)
        Uq = self.log_marginal(ob=ob, log_tau=log_tau, mu=mu)
        assert Kp.shape == (self.S, self.B), "ERROR! Kp has unexpected shape."
        assert Uq.shape ==  (self.S, self.B), 'ERROR! Uq has unexpected shape.'
        return Kp + Uq


    def kinetic_energy(self, r_tau, r_mu):
        """
        r_tau, r_mu : S * B * K * D
        return - 1/2 * ||(r_tau, r_mu)||^2
        """
        return - ((r_tau ** 2).sum(-1).sum(-1) + (r_mu ** 2).sum(-1).sum(-1)) * 0.5

    def log_marginal(self, ob, log_tau, mu):
        """
        compute log density log p(x_1:N, mu_1:N, tau_1:N)
        by marginalizing discrete varaibles :
                                   loss_required=loss_required,
                                   ess_required=ess_required,
                                   mode_required=mode_required,
                                   density_required=density_required
        log p(x_1:N, mu_1:N, tau_1:N)
        = \sum_{n=1}^N [log(\sum_{k=1}^K N(x_n; \mu_k, \Sigma_k)) - log(K)]
          + \sum_{k=1}^K [log p(\mu_k) + log p(\Sigma_k)]

        S * B tensor
        """
        tau = log_tau.exp()
        sigma = 1. / tau.sqrt()
        logprior_tau =(Gamma(self.generative.prior_alpha, self.generative.prior_beta).log_prob(tau) + log_tau).sum(-1).sum(-1)  # S * B
        logprior_mu = Normal(self.generative.prior_mu, 1. / (self.generative.prior_nu * tau).sqrt()).log_prob(mu).sum(-1).sum(-1)  # S * B
        mu_expand = mu.unsqueeze(2).repeat(1, 1, self.N, 1, 1).permute(3, 0, 1, 2, 4)
        sigma_expand = sigma.unsqueeze(2).repeat(1, 1, self.N, 1, 1).permute(3, 0, 1, 2, 4) #  K * S * B * N * D
        ll = Normal(mu_expand, sigma_expand).log_prob(ob).sum(-1).permute(1, 2, 3, 0) # S * B * N * K
        log_density = torch.logsumexp(self.generative.prior_pi.log() + ll, dim=-1).sum(-1)
        return log_density + logprior_mu + logprior_tau
