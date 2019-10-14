import torch
import torch.nn as nn
import probtorch

class Dec_digit(nn.Module):
    """
    reconstruct the video
    z_what S * B * K * z_what_dim
    z_where S * B * K * D
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim, AT):
        super(self.__class__, self).__init__()
        self.digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())
        self.AT = AT

    def forward(self, frames, z_what, z_where=None):
        digit_mean = self.digit_mean(z_what)  # S * B * K * (28*28)
        S, B, K, _ = digit_mean.shape
        digit_mean = digit_mean.view(S, B, K, 28, 28)
        if z_where is None:
            return digit_mean.detach()
        else:
            recon_mean = self.AT.digit_to_frame(digit_mean, z_where)
            recon_frame = torch.clamp(recon_mean.sum(-3), min=0.0, max=1.0) # S * B * 64 * 64
            log_recon = MBern_log_prob(recon_frame, frames) # S * B
            return recon_frame, log_recon, digit_mean.detach()

    def forward_vectorized(self, frames, z_what, z_where=None):
        digit_mean = self.digit_mean(z_what)  # S * B * K * (28*28)
        S, B, K, _ = digit_mean.shape
        digit_mean = digit_mean.view(S, B, K, 28, 28)
        if z_where is None:
            return digit_mean.detach()
        else:
            recon_mean = self.AT.digit_to_frame_vectorized(digit_mean, z_where)
            recon_frame = torch.clamp(recon_mean.sum(-3), min=0.0, max=1.0) # S * B * T * 64 * 64
            log_recon = MBern_log_prob(recon_frame, frames).sum(-1) # S * B
            return recon_frame, log_recon, digit_mean.detach()

def MBern_log_prob(x_mean, x, EPS=1e-9):
    """
    the size is ... * H * W
    so I added two sum ops
    """
    return (torch.log(x_mean + EPS) * x +
                torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1)
