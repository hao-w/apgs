import torch
from torch.distributions.categorical import Categorical
"""
=========
TODO : provide more sampling methods like systematic resampling based on the paper:
GPU acceleration of the particle filter: the Metropolis resampler https://arxiv.org/abs/1202.6163
=========
"""
def resample(var, weights, dim_expand=True):
    dim1, dim2, dim3, dim4 = var.shape
    if dim_expand:
        if dim2 == 1:
            ancesters = Categorical(weights.permute(1, 2, 0).squeeze(0)).sample((dim1, )).unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, dim4)
        else:
            ancesters = Categorical(weights.permute(1, 2, 0)).sample((dim1, )).unsqueeze(-1).repeat(1, 1, 1, dim4)
    else:
        ancesters = Categorical(weights.transpose(0, 1)).sample((dim1, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4) ## S * B * N * K
    return torch.gather(var, 0, ancesters)
