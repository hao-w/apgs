import torch
import torch.nn as nn
import torch.nn.functional.conv2d
from torch.distributions.normal import Normal
import probtorch
import math

class Enc_corr(nn.Module):
    def __init__():
        super(self.__class__, self)
    """
    conv2d usage : input T * B * H * W
                 : kernels (S*B) * 1 * H_k * W_k
                 reshape the digit_images

    frames : B * T * 64 * 64
    digit_images : S * B * 28 * 28
    """
    def __init__(self, num_pixels, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.Tanh())
        self.where_mean = nn.Sequential(
                            nn.Linear(num_hidden, 2),
                            nn.Tanh())

        self.where_std = torch.ones(2) * 0.1
        if CUDA:
            with torch.cuda.device(device):
                self.where_std = self.where_std.cuda()
        self.where_std = nn.Parameters(self.where_std)

    def forward(self, frames, digit_images, S, sampled=True, z_where_old=None):
        q = probtorch.Trace()
        B, T, FP, _ = frames.shape
        S, B, DP, _ = digit_images.shape
        convolved = conv2d(frames.transpose(0, 1), digit_images.view(S*B, DP, DP).unsqueeze(1), groups=B)
        CP = convolved.shape[-1] # convolved output pixels ## T * S * B * CP * CP
        convolved = convolved.view(T, S, B, CP*CP)
        q_mean = self.enc_mean(self.enc_hidden(convolved)).permute(1, 2, 0) ## S * B * T * 2
        z_where = Normal(q_mean, self.where_std).sample()
        q.normal(loc=q_mean, scale=self.where_std, value=z_where, name='z_where')
        return q
