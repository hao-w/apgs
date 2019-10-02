import torch
import torch.nn as nn
import probtorch
from utils import MBern_log_prob

class Dec_digit(nn.Module):
    """
    reconstruct the video
    z_what S * B * K * z_what_dim
    z_where S * B * T * K * D
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim):
        super(self.__class__, self).__init__()
        self.digit_mean = nn.Sequential(nn.Linear(z_what_dim, int(0.5*num_hidden)),
                                    nn.ReLU(),
                                    nn.Linear(int(0.5*num_hidden), num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

    def forward(self, frames, z_what, crop, z_where=None, intermediate=False, FP=64, DP=28):

        digit_mean = self.digit_mean(z_what)  # S * B * K * (28*28)
        if intermediate:
            S, B, K, _ = digit_mean.shape
            return digit_mean.view(S, B, K, DP, DP).detach()
        else:
            S, B, T, K, _ = z_where.shape
            # print(z_where.shape)
            recon_mean = crop.digit_to_frame(digit_mean.view(S, B, K, DP, DP), z_where)
            recon_frame = torch.clamp(recon_mean.sum(-3), min=0.0, max=1.0) # S * B * T * 64 * 64
            log_recon = MBern_log_prob(recon_frame, frames) # S * B * T
            return recon_frame, log_recon
