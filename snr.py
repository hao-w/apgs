import torch

def mmt_grad(model, iteration, fst_mmt_old, sec_mmt_old, beta_1=0.99, beta_2=0.99):
    """
    compute the exponetial moving average of the first and second moments for the gradient estimations
    """
    params = []
    fst_mmt = []
    sec_mmt = []
    for module in model:
        if isinstance(module, torch.nn.Module):
            params = params + list(module.parameters())
    if iteration == 1:
        for ind, p in enumerate(params):
            grad = p.grad.cpu().detach()
            fst_mmt.append(grad)
            sec_mmt.append(grad**2)
    else:
        assert fst_mmt_old is not None, "ERROR! NoneType found."
        assert sec_mmt_old is not None, "ERROR! NoneType found."
        for ind, p in enumerate(params):
            grad = p.grad.cpu().detach()
            unbiased_1 = (1 - beta_1**(iteration-1)) / (1 - beta_1**iteration)
            unbiased_2 = (1 - beta_2**(iteration-1)) / (1 - beta_2**iteration)
            fst_mmt.append((beta_1 * fst_mmt_old[ind] + (1 - beta_1) * grad) * unbiased_1)
            sec_mmt.append((beta_2 * sec_mmt_old[ind] + (1 - beta_2) * (grad**2)) * unbiased_2)
    return fst_mmt, sec_mmt

def snr_grad(fst_mmt, sec_mmt, EPS=1e-9):
    """
    compute the signal-to-noise ratio and variance (i.e.trace of the covariance matrix),
    which are normalized by number of paramters
    snr = 1 / (var / E[g^2]) = 1 / (1 - E^2[g] / E[g^2])
    """
    snr = None
    var = None
    N = 0.0
    for ind in range(len(fst_mmt)):
        var_p = sec_mmt[ind] - fst_mmt[ind]**2
        snr_p = 1. / (1. - (fst_mmt[ind]**2+ EPS) / (sec_mmt[ind]+EPS))
        if snr is None:
            snr = snr_p.sum()
        else:
            snr = snr + snr_p.sum()

        if var is None:
            var = var_p.sum()
        else:
            var = var + var_p.sum()
        N += var_p.numel()

    return snr / N , var / N
