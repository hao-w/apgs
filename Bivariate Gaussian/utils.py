import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def resample(var, weights):
    dim1, dim2 = var.shape
    ancesters = Categorical(weights).sample((dim1, )).unsqueeze(-1)
    return torch.gather(var, 0, ancesters)

def Gibbs(bg, Steps, sampling=True):
    """
    perform closed-form Gibbs sampling
    return the new corrdinate after each sweep
    """
    ## randomly start from a point in 2D space
    init = Normal(torch.ones(2) * -5, torch.ones(2) * 2).sample().unsqueeze(0)
#     Mu, Sigma = bg.Joint()
#     INIT = Normal(Mu, Sigma).sample().unsqueeze(0)
    x2 = init[:, 1]
    x1 = init[:, 0]
    updates = []
    updates.append(init)
    for i in range(Steps):
        ## update x1
        cond_x1_mu, cond_x1_sigma = bg.conditional(x2, cond='x2')
        kernel_x1 = Normal(cond_x1_mu, cond_x1_sigma)
        if sampling:
            x1 = kernel_x1.sample()
        else:
            x1 = kernel_x1.mean
    #     updates.append(torch.cat((x1, x2), 0).unsqueeze(0))
        ## update x2
        cond_x2_mu, cond_x2_sigma = bg.conditional(x1, cond='x1')
        kernel_x2 = Normal(cond_x2_mu, cond_x2_sigma)
        if sampling:
            x2 = kernel_x2.sample()
        else:
            x2 = kernel_x2.mean
        updates.append(torch.cat((x1, x2), 0).unsqueeze(0))
    return torch.cat(updates, 0)

## some standard KL-divergence functions
def kl_normal_normal(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
