import torch
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch

class Generative():
    def __init__(self, K, D, CUDA, DEVICE):
        super().__init__()

        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.3
        self.prior_alpha = torch.ones((K, D)) * 4
        self.prior_beta = torch.ones((K, D)) * 4
        self.prior_pi = torch.ones(K) * (1./ K)

        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_nu = self.prior_nu.cuda()
                self.prior_alpha = self.prior_alpha.cuda()
                self.prior_beta = self.prior_beta.cuda()
                self.prior_pi = self.prior_pi.cuda()

    def prior_eta(self, q):
        p = probtorch.Trace()
        ## prior distributions
        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=q['precisions'],
                name='precisions')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * p['precisions'].value).sqrt(),
                 value=q['means'],
                 name='means')
        return p

    def prior_z(self, q):
        _ = p.variable(cat, probs=self.prior_pi, value=q['states'], name='states')
        return p

    def log_prob(self, ob, z , tau, mu, cluster_flag=False):
        """
        cluster_flag = False : return S * B * N
        cluster_flag = True : return S * B * K
        """
        sigma = 1. / tau.sqrt()
        labels = z.argmax(-1)
        labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, ob.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(sigma, 2, labels_flat)
        ll = Normal(mu_expand, sigma_expand).log_prob(ob).sum(-1) # S * B * N
        if cluster_flag:
            ll = torch.cat([((labels==k).float() * ll).sum(-1).unsqueeze(-1) for k in range(z.shape[-1])], -1) # S * B * K
        return ll
