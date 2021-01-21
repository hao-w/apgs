import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def resample(var, weights):
    dim1, dim2 = var.shape
    ancesters = Categorical(weights).sample((dim1, ))
    return var[ancesters]

## some standard KL-divergence functions
def kl_normal_normal(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

def kls_normal_normal(p_mean, p_std, q_mean, q_std):
    in_kl = kl_normal_normal(p_mean, p_std, q_mean, q_std).squeeze(-1).mean()
    ex_kl = kl_normal_normal(q_mean, q_std, p_mean, p_std).squeeze(-1).mean()
    return in_kl, ex_kl