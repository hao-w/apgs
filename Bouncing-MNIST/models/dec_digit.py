import torch
import torch.nn as nn
import probtorch
import math
from utils import MBern_log_prob

class Dec_digit(nn.Module):
    """
    reconstruct the video
    z_what S * B * z_what_dim
    z_where S * B * T * 2
    """
    def __init__(self, num_pixels, num_hidden, z_what_dim):
        super(self.__class__, self).__init__()
        self.digit_mean = nn.Sequential(nn.Linear(z_what_dim, num_hidden),
                                    nn.ReLU(),
                                    nn.Linear(num_hidden, num_pixels),
                                    nn.Sigmoid())

    def forward(self, z_what):
        S, B, _ = z_what.shape
        digit_mean = self.digit_mean(z_what)  # S * B * (28*28)
        return digit_mean


def dec_frame(crop, frames, digit_images, z_where, DP=28, FP=64):
    recon_mean = crop.digit_to_frame(digit_images.view(S, B, DP, DP), z_where) # S * B * T * 64 * 64
    log_recon = MBern_log_prob(recon_mean, frames) # S * B * T
    return recon_mean, log_recon
