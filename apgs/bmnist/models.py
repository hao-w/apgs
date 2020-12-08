import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import probtorch
import math

class Enc_coor(nn.Module):
    """
    encoder of the digit positions
    """
    def __init__(self, num_pixels, num_hidden, z_where_dim, reparameterized=False):
        super(self.__class__, self).__init__()
        self.enc_coor_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.where_mean = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim),
                            nn.Tanh())

        self.where_log_std = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim))
        self.reparameterized = reparameterized

    def forward(self, conved, sampled=True, z_where_old=None):
        q = probtorch.Trace()
        hidden = self.enc_coor_hidden(conved)
        q_mean = self.where_mean(hidden)
        q_std = self.where_log_std(hidden).exp()
        if sampled:
            if self.reparameterized:
                z_where = Normal(q_mean, q_std).rsample()
            else:
                z_where = Normal(q_mean, q_std).sample()
            q.normal(loc=q_mean, scale=q_std, value=z_where, name='z_where')
        else:
            q.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where')
        return q

    
class Enc_digit(nn.Module):
    """
    encoder of digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, reparameterized=False):
        super(self.__class__, self).__init__()
        self.enc_digit_hidden = nn.Sequential(
                        nn.Linear(num_pixels, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, int(0.5*num_hidden)),
                        nn.ReLU())
        self.enc_digit_mean = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))
        self.enc_digit_log_std = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))

        self.reparameterized = reparameterized
        
    def forward(self, cropped, sampled=True, z_what_old=None):
        q = probtorch.Trace()
        hidden = self.enc_digit_hidden(cropped).mean(2)
        q_mu = self.enc_digit_mean(hidden) ## because T is on the 3rd dim in cropped
        q_std = self.enc_digit_log_std(hidden).exp()
        if sampled:
            if self.reparameterized:
                z_what = Normal(q_mu, q_std).rsample()
            else:
                z_what = Normal(q_mu, q_std).sample() ## S * B * K * z_what_dim
            q.normal(loc=q_mu,
                     scale=q_std,
                     value=z_what,
                     name='z_what')
        else:
            q.normal(loc=q_mu,
                     scale=q_std,
                     value=z_what_old,
                     name='z_what')

        return q

    
class Dec_coor():
    """
    generative model of digit positions
    Real generative model for time dynamics
    z_1 ~ N (0, Sigma_0) : S * B * D
    z_t | z_t-1 ~ N (A z_t-1, Sigma_t)
    where A is the transformation matrix
    """
    def __init__(self, z_where_dim, CUDA, device):
        super(self.__class__, self)

        self.prior_mu0 = torch.zeros(z_where_dim)
        self.prior_Sigma0 = torch.ones(z_where_dim) * 1.0
        self.prior_Sigmat = torch.ones(z_where_dim) * 0.2
        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu0  = self.prior_mu0.cuda()
                self.prior_Sigma0 = self.prior_Sigma0.cuda()
                self.prior_Sigmat = self.prior_Sigmat.cuda()
        # self.prior_Sigmat = nn.Parameter(self.prior_Sigmat)
    def forward(self, z_where_t, z_where_t_1=None, disp=None):
        S, B, D = z_where_t.shape
        if z_where_t_1 is None:
            p0 = Normal(self.prior_mu0, self.prior_Sigma0)
            return p0.log_prob(z_where_t).sum(-1)# S * B
        else:
            p0 = Normal(z_where_t_1, self.prior_Sigmat)
            return p0.log_prob(z_where_t).sum(-1) # S * B

    def log_prior(self, z_where_t, z_where_t_1=None, disp=None):
        S, B, K, D = z_where_t.shape
        if z_where_t_1 is None:
            p0 = Normal(self.prior_mu0, self.prior_Sigma0)
            return p0.log_prob(z_where_t).sum(-1).sum(-1)#)# S * B
        else:
            p0 = Normal(z_where_t_1, self.prior_Sigmat)
            return p0.log_prob(z_where_t).sum(-1).sum(-1)# # S * B
        
        
class Dec_digit(nn.Module):
    """
    decoder of the digit features
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, CUDA, device):
        super(self.__class__, self).__init__()
        self.dec_digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

        self.prior_mu = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_std = self.prior_std.cuda()

    def forward(self, frames, z_what, z_where=None, AT=None):
        digit_mean = self.dec_digit_mean(z_what)  # S * B * K * (28*28)
        S, B, K, DP2 = digit_mean.shape
        DP = int(math.sqrt(DP2))
        digit_mean = digit_mean.view(S, B, K, DP, DP)
        if z_where is None: ## return the recnostruction of mnist image
            return digit_mean.detach()
        else: # return the reconstruction of the frame
            assert AT is not None, "ERROR! NoneType variable AT found."
            assert frames is not None, "ERROR! NoneType variable frames found."
            _, _, T, FP, _ = frames.shape
            recon_frames = torch.clamp(AT.digit_to_frame(digit=digit_mean, z_where=z_where).sum(-3), min=0.0, max=1.0) # S * B * T * FP * FP
            assert recon_frames.shape == (S, B, T, FP, FP), "ERROR! unexpected reconstruction shape"
            log_prior = Normal(loc=self.prior_mu, scale=self.prior_std).log_prob(z_what).sum(-1) # S * B * K
            assert log_prior.shape == (S, B, K), "ERROR! unexpected prior shape"
            ll = MBern_log_prob(recon_frames, frames) # S * B * T, log likelihood log p(x | z)

            return log_prior, ll, recon_frames

def MBern_log_prob(x_mean, x, EPS=1e-9):
    """
    the size is ... * H * W
    so I added two sum ops
    """
    return (torch.log(x_mean + EPS) * x +
                torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1)
