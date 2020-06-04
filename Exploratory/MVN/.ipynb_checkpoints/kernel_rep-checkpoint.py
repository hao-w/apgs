import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Kernel(nn.Module):
    def __init__(self, num_hiddens):
        super(self.__class__, self).__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))

        self.q_log_sigma = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))

    def forward(self, x, sampled=True, samples_old=None):
        ## input size S * 1
        mu = self.q_mu(x)
        sigma = self.q_log_sigma(x).exp()
        q = Normal(mu, sigma)
        if sampled:
            samples = q.rsample()
            log_pdf = q.log_prob(samples)
            return samples, log_pdf, mu, sigma
        else:
            log_pdf = q.log_prob(samples_old)
        return log_pdf, mu, sigma
