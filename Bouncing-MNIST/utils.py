import torch
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical

def Resample_where(z_where, weights):
    S, B, T, dim4 = z_where.shape
    ancesters = Categorical(weights.transpose(0, 1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, dim4) ## S * B * T * 2
    return torch.gather(z_where, 0, ancesters)

def MBern_log_prob(x_mean, x, EPS=1e-9):
    """
    the size is ... * H * W
    so I added two sum ops
    """
    return (torch.log(x_mean + EPS) * x +
                torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1)

def Resample_what(z_what, weights):
    S, B, dim3 = z_what.shape
    ancesters = Categorical(weights.transpose(0, 1)).sample((S, )).unsqueeze(-1).repeat(1, 1, dim3) ## S * B * dim3
    return torch.gather(z_what, 0, ancesters)