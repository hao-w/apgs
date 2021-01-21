import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as mvn

def init_kernels(num_hiddens, CUDA, device, reparameterized=False):
    """
    intiailize forward and backward kernels
    """
    fk = Kernel(num_hiddens=num_hiddens, reparameterized=reparameterized)
    bk = Kernel(num_hiddens=num_hiddens, reparameterized=reparameterized)
    if CUDA:
        with torch.cuda.device(device):
            fk.cuda()
            bk.cuda()
    return fk, bk

class Kernel(nn.Module):
    """
    A 1D proposal
    """
    def __init__(self, num_hiddens, reparameterized=False):
        super(self.__class__, self).__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))

        self.q_log_sigma = nn.Sequential(
            nn.Linear(1, num_hiddens),
            # nn.Tanh(),
            nn.Linear(num_hiddens, 1))
        self.reparameterized = reparameterized 
        
    def forward(self, x, sampled=True, samples_old=None):
        ## input size S * 1
        mu = self.q_mu(x)
        sigma = self.q_log_sigma(x).exp()
        q = Normal(mu, sigma)
        if sampled:
            if self.reparameterized:
                samples = q.rsample()
            else:
                samples = q.sample()
            log_pdf = q.log_prob(samples)
            return samples, log_pdf, mu, sigma
        else:
            log_pdf = q.log_prob(samples_old)
        return log_pdf, mu, sigma

class Bivariate_Gaussian():
    """
    A bivariate guassian
    """
    def __init__(self, mu1, mu2, sigma1, sigma2, rho, CUDA, device):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        
        self.prior_mu, self.prior_sigma = torch.zeros(2), torch.ones(2)
        if CUDA:
            with torch.cuda.device(device):
                self.mu1 = self.mu1.cuda()
                self.mu2 = self.mu2.cuda()
                self.sigma1 = self.sigma1.cuda()
                self.sigma2 = self.sigma2.cuda()
                self.rho = self.rho.cuda()
                self.prior_mu = self.prior_mu.cuda()
                self.prior_sigma = self.prior_sigma.cuda()
                
        self.Mu = torch.cat((self.mu1.unsqueeze(-1), self.mu2.unsqueeze(-1)), -1)
        cov = self.rho * self.sigma1 * self.sigma2
        part1 = torch.cat((self.sigma1.unsqueeze(-1)**2, cov.unsqueeze(-1)), -1)
        part2 = torch.cat((cov.unsqueeze(-1), self.sigma2.unsqueeze(-1)**2), -1)
        self.Sigma = torch.cat((part1.unsqueeze(-1), part2.unsqueeze(-1)), -1)
        self.dist = mvn(self.Mu, self.Sigma)
        self.prior = Normal(self.prior_mu, self.prior_sigma)
        
    def conditional_params(self, x, cond):
        """
        return parameters of conditional distribution X1 | X2 or X2 | X1
        based on the cond flag
        """
        if cond == 'x2': ## X1 | X2
            cond_mu = self.mu1 + (self.sigma1 / self.sigma2) * (x - self.mu2) * self.rho
            cond_sigma = ((1 - self.rho ** 2) * (self.sigma1**2)).sqrt()
        else: ## X2 | X1
            cond_mu = self.mu2 + (self.sigma2 / self.sigma1) * (x - self.mu1) * self.rho
            cond_sigma = ((1 - self.rho ** 2) * (self.sigma2**2)).sqrt()
        return cond_mu, cond_sigma

    def log_pdf_gamma(self, x, mu, sigma):
        """
        return log unormalized density of a univariate Gaussian
        """
        return Normal(mu, sigma).log_prob(x)
        # return ((x - mu)**2) / (-2 * sigma**2)

    def marginal(self, x, name):
        """
        return the log prob of marginal
        """
        if name == 'x1':
            return Normal(self.mu1, self.sigma1).log_prob(x)
        else:
            return Normal(self.mu2, self.sigma2).log_prob(x)
        
    def sample_prior(self, num_samples):
        x = self.prior.sample((num_samples, ))
        return x, self.prior.log_prob(x).sum(-1)