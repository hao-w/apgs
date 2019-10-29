import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Kernel(nn.Module):
    def __init__(self, num_hiddens, prior_mu, prior_sigma, CUDA, device):
        super(self.__class__, self).__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))

        self.q_log_sigma = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))

        self.p_mu = prior_mu
        self.p_sigma = prior_sigma

        if CUDA:
             with torch.cuda.device(device):
                self.p_mu = self.p_mu.cuda()
                self.p_sigma = self.p_sigma.cuda()

    def forward(self, cond_var, sampled):
        ## input size S * 1
        mu = self.q_mu(cond_var)
        sigma = self.q_log_sigma(cond_var).exp()
        q = Normal(mu, sigma)
        if sampled:
            value = q.sample()
        else:
            value = mu.detach()
        log_pdf = q.log_prob(value)
        return value, log_pdf, mu, sigma

    def backward(self, cond_var, value):
        mu = self.q_mu(cond_var)
        sigma = self.q_log_sigma(cond_var).exp()
        q = Normal(mu, sigma)
        return q.log_prob(value)

    def sample_prior(self, num_samples):
        p = Normal(self.p_mu, self.p_sigma)
        sampled_var = p.sample((num_samples,))
        log_pdf = p.log_prob(sampled_var)
        return sampled_var, log_pdf
