from numbers import Number

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from scipy.stats import vonmises
import scipy.special as scs
from torch.distributions.utils import broadcast_all
import numpy as np
import math


class _standard_vonmises(Function):
    @staticmethod
    def forward(ctx, K):
        z = torch.tensor(vonmises.rvs(kappa=K.detach().numpy()), dtype=K.dtype)
        ctx.save_for_backward(K, z)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        K, z = ctx.saved_tensors
        T = 200  # truncation param
        I0 = torch.tensor(scs.i0(K.detach().numpy()), dtype=K.dtype)
        I1 = torch.tensor(scs.i1(K.detach().numpy()), dtype=K.dtype)
        Ij = I1.clone()
        dF_dK = 0
        for j in range(1, T + 1):
            j = float(j)
            Ijp1 = torch.tensor(scs.iv(j + 1, K.detach().numpy()), dtype=K.dtype)
            dF_dK += (j * Ij / K + Ijp1 - Ij * I1 / I0) * torch.sin(j * z) / j
            Ij = Ijp1.clone()
        # -dF_dK/dF_dz when I0s cancel out
        return -2 * dF_dK / torch.exp(K * torch.cos(z)) * grad_output,


class MyVonMises(torch.distributions.Distribution):
    _standard_vonmises = _standard_vonmises.apply

    arg_constraints = {'loc': constraints.real, 'K': constraints.positive}
    support = constraints.dependent
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return 1.0 - scs.i1(self.K) / scs.i0(self.K)

    def __init__(self, K, loc=0, validate_args=None):
        self.K, self.loc = broadcast_all(K, loc)
        if isinstance(K, Number) and isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.K.size()
        super(MyVonMises, self).__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        svm_sample = self._standard_vonmises(self.K.expand(shape))
        return svm_sample + self.loc.expand(shape)
