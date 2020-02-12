import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.beta import Beta
import math

class HMC():
    def __init__(self, enc_local, dec, burn_in, S, B, N, K, D, CUDA, DEVICE):
        self.S = S
        self.B = B
        self.N = N
        self.K = K
        self.D = D

        self.dec = dec
        self.enc_local = enc_local
        self.Sigma = torch.ones(1)
        self.mu = torch.zeros(1)
        self.accept_count = 0.0
        self.smallest_accept_ratio = 0.0
        self.burn_in = burn_in
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.Sigma = self.Sigma.cuda()
                self.mu = self.mu.cuda()
                self.uniformer = Uniform(torch.Tensor([0.0]).cuda(), torch.Tensor([1.0]).cuda())
        else:
            self.uniformer = Uniform(torch.Tensor([0.0]), torch.Tensor([1.0]))

        self.gauss_dist = Normal(self.mu, self.Sigma)

        self.prior_mu_mu = dec.prior_mu_mu
        self.prior_mu_sigma = dec.prior_mu_sigma

    def init_sample(self):
        """
        initialize auxiliary variables from univariate Gaussian
        return r_mu, r_beta
        """
        return self.gauss_dist.sample((self.S, self.B, self.K, self.D, )).squeeze(-1)
        # self.gauss_dist.sample((self.S, self.B, self.N, ))

    def beta_to_y(self, beta):
        return (beta / (1 - beta)).log()

    def y_to_beta(self, y):
        return 1. / (1 + (- y).exp())
    def hmc_sampling(self, ob, mu, z, beta, trace, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps):

        """
        change of variables beta  = 1 / (1 + exp(-y))
        y = log (beta / (1 - beta))
        """
        y = self.beta_to_y(beta)
        self.accept_count = 0.0
        for m in range(hmc_num_steps):
            
            mu = self.metrioplis(ob=ob, mu=mu.detach(), z=beta, beta=beta, step_size=leapfrog_step_size, num_steps=leapfrog_num_steps)
            # print(y)
            q = self.enc_local(ob=ob, mu=mu, K=self.K)
            z = q['states'].value
            beta = q['angles'].value
            # beta = self.y_to_beta(y)
            # log_marginal = self.log_marginal(ob, mu, y)
            log_joint = self.log_joint(ob=ob, mu=mu, z=z, beta=beta)
            trace['density'].append(log_joint.unsqueeze(0))
            # trace['marginal'].append(log_marginal.unsqueeze(0))
            # if m % 10 == 0:
            #     print('hmc-step=%d' % (m+1))
                # print(self.accept_count)
        self.smallest_accept_ratio = (self.accept_count  / hmc_num_steps).min()
        return mu, trace

    def log_joint(self, ob, mu, z, beta):
        p = self.dec.forward(ob=ob, mu=mu, z=z, beta=beta)
        ll = p['likelihood'].log_prob.sum(-1).sum(-1)
        log_prior_mu = Normal(self.dec.prior_mu_mu, self.dec.prior_mu_sigma).log_prob(mu).sum(-1).sum(-1)
        log_prior_z = cat(probs=self.dec.prior_pi).log_prob(z).sum(-1)
        log_prior_beta = Beta(self.dec.prior_con1, self.dec.prior_con0).log_prob(beta).sum(-1).sum(-1)
        return ll + log_prior_mu + log_prior_z + log_prior_beta

    def log_conditional(self, ob, mu, z, beta):
        p = self.dec.forward(ob=ob, mu=mu, z=z, beta=beta)
        ll = p['likelihood'].log_prob.sum(-1).sum(-1)
        log_prior_mu = Normal(self.dec.prior_mu_mu, self.dec.prior_mu_sigma).log_prob(mu).sum(-1).sum(-1)
        # log_prior_z = cat(probs=self.dec.prior_pi).log_prob(z).sum(-1)
        # log_prior_beta = Beta(self.dec.prior_con1, self.dec.prior_con0).log_prob(beta).sum(-1).sum(-1)
        return ll + log_prior_mu

    def metrioplis(self, ob, mu, z, beta, step_size, num_steps):
        r_mu = self.init_sample()
        ## compute hamiltonian given original position and momentum
        H_orig = self.hamiltonian(ob=ob, mu=mu, z=z, beta=beta, r_mu=r_mu)
        new_mu, new_r_mu = self.leapfrog(ob, mu, z, beta, r_mu, step_size, num_steps)
        # print(log_tau)
        ## compute hamiltonian given new proposals
        H_new = self.hamiltonian(ob=ob, mu=new_mu, z=z, beta=beta, r_mu=new_r_mu)
        accept_ratio = (H_new - H_orig).exp()
        u_samples = self.uniformer.sample((self.S, self.B, )).squeeze(-1)
        accept_index = (u_samples < accept_ratio)

        # assert accept_index.shape == (self.S, self.B), "ERROR! index has unexpected shape."
        accept_index_expand1 = accept_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.K, self.D)
        # accept_index_expand2 = accept_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.N, 1)

        # assert accept_index_expand.shape == (self.S, self.B, self.K, self.D), "ERROR! index has unexpected shape."
        filtered_mu = new_mu * accept_index_expand1.float() + mu * (~accept_index_expand1).float()
        # filtered_y = new_y * accept_index_expand2.float() + y * (~accept_index_expand2).float()
        self.accept_count = self.accept_count + accept_index.float()
        return filtered_mu.detach()


    def leapfrog(self, ob, mu, z, beta, r_mu, step_size, num_steps):
        for step in range(num_steps):
            mu.requires_grad = True
            # y.requires_grad = True
            log_p = self.log_conditional(ob, mu, z, beta)
            log_p.sum().backward(retain_graph=False)
            # print(log_p.sum())
            r_mu = (r_mu + 0.5 * step_size * mu.grad).detach()
            # r_beta = (r_beta + 0.5 * step_size * y.grad).detach()
            #
            # mu = (mu + step_size * r_mu).detach()
            # y = (y + step_size * r_beta).detach()
            # beta = beta % (1.0)
            mu.requires_grad = True
            # y.requires_grad = True
            log_p = self.log_conditional(ob, mu, z, beta)
            log_p.sum().backward(retain_graph=False)
            r_mu = (r_mu + 0.5 * step_size * mu.grad).detach()
            # r_beta = (r_beta + 0.5 * step_size * y.grad).detach()
            mu = mu.detach()
            # y = y.detach()
        return mu, r_mu



    def hamiltonian(self, ob, mu, z, beta, r_mu):
        """
        compute the Hamiltonian given the position and momntum

        """
        Kp = self.kinetic_energy(r_mu=r_mu)
        Uq = self.log_conditional(ob=ob, mu=mu, z=z, beta=beta)
        assert Kp.shape == (self.S, self.B), "ERROR! Kp has unexpected shape."
        assert Uq.shape ==  (self.S, self.B), 'ERROR! Uq has unexpected shape.'
        return Kp + Uq


    def kinetic_energy(self, r_mu):
        """
        r_mu : S * B * K * D
        return - 1/2 * ||r_mu+r_beta||^2
        """
        return - ((r_mu ** 2).sum(-1).sum(-1)) * 0.5

    def log_marginal(self, ob, mu, y):
        """
        compute log density log p(x_1:N, mu_1:K, tau_1:N)
        by marginalizing discrete varaibles :                                   loss_required=loss_required,
                                   ess_required=ess_required,
                                   mode_required=mode_required,
                                   density_required=density_required
        log p(x_1:N, mu_1:N, tau_1:N)
        = \sum_{n=1}^N [log(\sum_{k=1}^K N(x_n; \mu_k, \Sigma_k)) - log(K)]
          + \sum_{k=1}^K [log p(\mu_k) + log p(\Sigma_k)]


        S * B tensor
        """
        beta = self.y_to_beta(y)
        mu_expand = mu.unsqueeze(2).repeat(1, 1, self.N, 1, 1).permute(3, 0, 1, 2, 4)
        ob_expand = ob.unsqueeze(0).repeat(self.K, 1, 1, 1, 1) # K S B N D
        beta_expand = beta.unsqueeze(0).repeat(self.K, 1, 1, 1, 1)
        # p = self.dec.forward(ob=ob_expand, mu=mu_expand, z=, beta=beta)
        # ll = p['likelihood'].log_prob.sum(-1).sum(-1)

        hidden = self.dec.recon_mu(beta * 2 * math.pi)
        hidden2 = hidden / (hidden**2).sum(-1).unsqueeze(-1).sqrt()
        # mu_expand = torch.gather(mu, -2, z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, D))
        recon_mu = hidden2 * self.dec.radi + mu_expand
        ll = Normal(recon_mu, self.dec.recon_sigma).log_prob(ob_expand).sum(-1).permute(1, 2, 3, 0)

        log_prior_mu = Normal(self.dec.prior_mu_mu, self.dec.prior_mu_sigma).log_prob(mu).sum(-1).sum(-1)
        log_prior_y = (Beta(self.dec.prior_con1, self.dec.prior_con0).log_prob(beta_expand) + (beta_expand * (1 - beta_expand)).log() ).sum(-1).permute(1, 2, 3, 0) # S B N K

        log_density = torch.logsumexp(self.dec.prior_pi.log() + ll + log_prior_y, dim=-1).sum(-1) # S B
        return log_density + log_prior_mu
