import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Dec_digit(nn.Module):
    """
    reconstruct the sequence
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, CUDA, DEVICE):
        super(self.__class__, self).__init__()
        self.digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

        self.prior_mu = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_std = self.prior_std.cuda()

    def forward(self, frames, z_what, z_where=None, AT=None):
        digit_mean = self.digit_mean(z_what)  # S * B * K * (28*28)
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
