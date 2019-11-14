import torch
import torch.nn as nn
import probtorch

class Dec_digit(nn.Module):
    """
    reconstruct the video
    z_what S * B * K * z_what_dim
    z_where S * B * K * D
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
        S, B, K, _ = digit_mean.shape
        digit_mean = digit_mean.view(S, B, K, 28, 28)
        if z_where is None:
            return digit_mean.detach()
        else:
            recon_mean = AT.digit_to_frame(digit_mean, z_where)
            recon_frame = torch.clamp(recon_mean.sum(-3), min=0.0, max=1.0) # S * B * T * 64 * 64

            p = probtorch.Trace()
            p.normal(loc=self.prior_mu,
                     scale=self.prior_std,
                     value=z_what_old,
                     name='z_what')
            log_recon = MBern_log_prob(recon_frame, frames) # S * B * T
            return recon_frame, log_recon, p # both S * B * T

def MBern_log_prob(x_mean, x, EPS=1e-9):
    """
    the size is ... * H * W
    so I added two sum ops
    """
    return (torch.log(x_mean + EPS) * x +
                torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1)
