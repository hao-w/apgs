import torch
from torch.distributions.normal import Normal
from utils  import kl_normal_normal, resample
import torch.nn.functional as F

class Bi_Gaussian():
    def __init__(self, mu1, mu2, sigma1, sigma2, rho, CUDA, device):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        if CUDA:
            with torch.cuda.device(device):
                self.mu1 = self.mu1.cuda()
                self.mu2 = self.mu2.cuda()
                self.sigma1 = self.sigma1.cuda()
                self.sigma2 = self.sigma2.cuda()
                self.rho = self.rho.cuda()

    def conditional(self, x, cond):
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

    def Joint(self):
        """
        return the paramters for the bivariate Gaussian given the indiviual parameters
        """
        Mu = torch.cat((self.mu1.unsqueeze(-1), self.mu2.unsqueeze(-1)), -1)
        cov = self.rho * self.sigma1 * self.sigma2
        part1 = torch.cat((self.sigma1.unsqueeze(-1)**2, cov.unsqueeze(-1)), -1)
        part2 = torch.cat((cov.unsqueeze(-1), self.sigma2.unsqueeze(-1)**2), -1)
        Sigma = torch.cat((part1.unsqueeze(-1), part2.unsqueeze(-1)), -1)
        return Mu, Sigma

    def Marginal(self, x, name):
        if name == 'x1':
            return Normal(self.mu1, self.sigma1).log_prob(x)
        else:
            return Normal(self.mu2, self.sigma2).log_prob(x)

    def fb(self, q, x_cond, x_cond_name, x_old, sampled=True, obj='ag'):
        x_new, log_q_f, _, _ = q.forward(x_cond, sampled=sampled)
        true_post_mu, true_post_sigma = self.conditional(x_cond, cond=x_cond_name)
        log_p_f = self.log_pdf_gamma(x_new, true_post_mu, true_post_sigma)
        log_q_b = q.backward(x_cond, x_old)
        log_p_b = self.log_pdf_gamma(x_old, true_post_mu, true_post_sigma)
        log_w_f = log_p_f.sum(-1) - log_q_f.sum(-1)
        log_w_b = log_p_b.sum(-1) - log_q_b.sum(-1)

        if obj =='ag':
            log_w = log_w_f - log_w_b
            w = F.softmax(log_w, 0).detach()
            loss = (w * log_w_f).sum(0).mean().unsqueeze(0)
        elif obj =='kl':
            log_w = log_w_f - log_w_b
            w = F.softmax(log_w, 0).detach()
            loss = (w * ((log_w.detach() + c + 1) * log_q_b - log_q_f)).sum(0).mean().unsqueeze(0)
        ess = (1. / (w ** 2).sum(0)).mean().unsqueeze(0)
        return loss, ess, x_new, log_w, w


    def train(self, q_x1, q_x2, mcmc_steps, num_samples, sampled=True, obj='ag'):
        """
        obj = 'ag' : the objective proposed in our paper
        obj= 'kl' : the KL(p(z_t) q(z_t-1 | z_t) || p(z_t-1) q(z_t | z_t-1))
        """
        losss = []
        esss = []
        ## start with sampling x1 from the prior
        x1, log_p_x1_f = q_x1.sample_prior(num_samples)
        x2, log_q_x2, _, _ = q_x2.forward(x1, sampled=sampled)
        r_mu, r_sigma = self.conditional(x1, cond='x1')
        log_p_x2_f = self.log_pdf_gamma(x2, r_mu, r_sigma)
        log_w = log_p_x2_f.sum(-1).detach() - log_q_x2.sum(-1)
        w_x2 = F.softmax(log_w, 0).detach()
        losss.append((w_x2 * log_w).sum(0).mean().unsqueeze(0))
        esss.append((1. / (w_x2** 2).sum(0)).mean().unsqueeze(0))
        for m in range(mcmc_steps):
            x2 = resample(x2, w_x2) ## resample x2
            x1_old = x1
            loss_x1, ess_x1, x1, log_w_x1, w_x1 = self.fb(q_x1, x2, 'x2', x1_old, sampled=sampled, obj=obj) ## update x1
            x1 = resample(x1, w_x1) ## resample x1
            x2_old = x2
            loss_x2, ess_x2, x2, log_w_x2, w_x2 = self.fb(q_x2, x1, 'x1', x2_old, sampled=sampled, obj=obj) ## update x2
            losss.append(loss_x1+loss_x2)
            esss.append((ess_x1 + ess_x2) / 2)
        return torch.cat(losss, 0).sum(), torch.cat(esss, 0).mean()

    def fb_test(self, q, x_cond, x_cond_name, x_old, sampled=True):
        x_new, log_q_f, q_mu, q_sigma = q.forward(x_cond, sampled=sampled)
        true_post_mu, true_post_sigma = self.conditional(x_cond, cond=x_cond_name)
        kls = kl_normal_normal(true_post_mu, true_post_sigma, q_mu, q_sigma).mean()

        log_p_f = self.log_pdf_gamma(x_new, true_post_mu, true_post_sigma)
        log_q_b = q.backward(x_cond, x_old)
        log_p_b = self.log_pdf_gamma(x_old, true_post_mu, true_post_sigma)
        log_w_f = log_p_f.sum(-1) - log_q_f.sum(-1)
        log_w_b = log_p_b.sum(-1) - log_q_b.sum(-1)
        log_w = log_w_f - log_w_b
        w = F.softmax(log_w, 0).detach()
        return x_new , (w * log_w).sum(0).mean().unsqueeze(0), kls, log_w

    def test(self, q_x1, q_x2, mcmc_steps, num_samples, sampled=True, init=None):
        updates = []
        DBs_q1 = []
        DBs_q2 = []
        KLs_q1 = []
        KLs_q2 = []

        ELBOs = []
        ## start with sampling x1 from the prior
        if init is not None:
            x1 = init
        else:
            x1, log_p_x1_f = q_x1.sample_prior(num_samples)

        x2, log_q_x2, q_mu, q_sigma = q_x2.forward(x1, sampled=sampled)
        r_mu, r_sigma = self.conditional(x1, cond='x1')
        log_p_x2_f = self.log_pdf_gamma(x2, r_mu, r_sigma)
        log_w = log_p_x2_f.sum(-1).detach() - log_q_x2.sum(-1)
        r_mu, r_sigma = self.conditional(x1, cond='x1')
        # kls = kl_normal_normal(r_mu, r_sigma, q_mu, q_sigma).mean()
        # KLs.append(kls.unsqueeze(0))
        updates.append(torch.cat((x1, x2), -1))
        ELBOs.append(log_w.mean().unsqueeze(0))
        # print('Init: x1=%.4f, x2=%.4f' % (x1, x2))
        for m in range(mcmc_steps):
            x1_old = x1
            x1, db_x1, kls_x1, log_w_x1  = self.fb_test(q_x1, x2, 'x2', x1_old, sampled=sampled) ## update x1
            DBs_q1.append(db_x1)
    #         x1 = post_mu_x1
            x2_old = x2
            x2, db_x2, kls_x2, log_w_x2 = self.fb_test(q_x2, x1, 'x1', x2_old, sampled=sampled) ## update x2
            DBs_q2.append(db_x2)
            KLs_q1.append(kls_x1.unsqueeze(0))
            KLs_q2.append(kls_x2.unsqueeze(0))
            updates.append(torch.cat((x1, x2), -1))
    #         print('step=%d: x1=%.4f, x2=%.4f' % (i, x1, x2))
            ELBOs.append(ELBOs[-1] + (log_w_x1 + log_w_x2).mean().unsqueeze(0))
        return torch.cat(updates, 0), [torch.cat(DBs_q1, 0), torch.cat(DBs_q2,0)], [torch.cat(KLs_q1, 0), torch.cat(KLs_q2, 0)], torch.cat(ELBOs, 0)
