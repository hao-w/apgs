import torch
import torch.nn as nn
from torch.nn.functional import conv2d, softmax
from torch.distributions.normal import Normal
import probtorch
import math

class Enc_coor(nn.Module):
    def __init__():
        super(self.__class__, self)
    """
    conv2d usage : input T * B * H * W
                 : kernels (S*B) * 1 * H_k * W_k
                 reshape the templates

    frames : B * T * 64 * 64
    templates : S * B * 28 * 28


    """
    def __init__(self, D, num_pixels, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.where_mean = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), D),
                            nn.Tanh())

        self.where_log_std = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), D))

        self.disp_mean = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU(),
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), D),
                            nn.Tanh())

    def forward(self, dec_coor, frames, digit, sampled=True, z_where_old=None):
        q = probtorch.Trace()
        B, T, FP, _ = frames.shape
        S, B, K, DP, _ = digit.shape
        convolved = conv2d(frames.transpose(0, 1).unsqueeze(1).repeat(1, S, 1, 1, 1).unsqueeze(3).repeat(1, 1, 1, K, 1, 1).view(T, S*B*K, FP, FP), digit.view(S*B*K, DP, DP).unsqueeze(1), groups=int(S*B*K))
        CP = convolved.shape[-1] # convolved output pixels ## T * S * B * CP * CP
        convolved = convolved.view(T, S, B, K, CP, CP).view(T, S, B, K, CP*CP).permute(1, 2, 0, -2, -1) ## S * B * T * K * 1639
        convolved_disp = convolved[:, :, 1:, :, :] -  convolved[:, :, :T-1, :, :]
        disp = self.disp_mean(convolved_disp)
        hidden = self.enc_hidden(softmax(convolved, -1))
        q_mean = self.where_mean(hidden) ## S * B * T * K * D
        q_std = self.where_log_std(hidden).exp() ## S * B * T * K * D
        if sampled:
            z_where = Normal(q_mean, q_std).sample()
            _, _, log_p = dec_coor.forward(z_where, disp)
            q.normal(loc=q_mean, scale=q_std, value=z_where, name='z_where')
        else:
            _, _, log_p = dec_coor.forward(z_where_old, disp)
            q.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where')


        return q, log_p
