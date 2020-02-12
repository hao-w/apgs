import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Kernel(nn.Module):
    def __init__(self, num_hiddens, prior_mu, prior_sigma, CUDA, DEVICE):
        super(self.__class__, self).__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))

        self.q_log_sigma = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))

        self.p_mu = torch.zeros(1)
        self.p_sigma = torch.ones(1) * 10

        if CUDA:
             with torch.cuda.device(DEVICE):
                self.p_mu = self.p_mu.cuda()
                self.p_sigma = self.p_sigma.cuda()

    def forward(self, cond_var, sampled=True, value_old=None):
        ## input size S * 1
        mu = self.q_mu(cond_var)
        sigma = self.q_log_sigma(cond_var).exp()
        q = Normal(mu, sigma)
        if sampled:
            value = q.sample()
        else:
            value = value_old
        log_pdf = q.log_prob(value)
        return value, log_pdf, mu, sigma
