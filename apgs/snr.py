import torch
# from torch.autograd.functional import jacobian
class EMA():
    """
    Compute the exponential moving average (EMA) of the gradient
    """
    def __init__(self, beta1, beta2):
        self.beta1 = beta1
        self.beta2 = beta2
        self.shadow1 = {}
        self.shadow2 = {}
        
    def register(self, name, grad):
        self.shadow1[name] = grad.clone()
        self.shadow2[name] = grad.clone()**2
        
    def update(self, name, grad):
        assert name in self.shadow1
        assert name in self.shadow2
        new_first_moment = (1.0 - self.beta1) * grad.clone() + self.beta1 * self.shadow1[name]
        new_second_moment = (1.0 - self.beta2) * (grad.clone()**2) + self.beta2 * self.shadow2[name]
        self.shadow1[name] = new_first_moment
        self.shadow2[name] = new_second_moment
    
    def snr(self, EPS=1e-12):
        N = 0.0
        Variance = 0.0
        SNR = 0.0
        for name, first_moment in self.shadow1.items():
            second_moment = self.shadow2[name]
            N += torch.numel(first_moment)
            Variance += (second_moment - (first_moment**2)).sum()
            snr_raw = 1.0 / (second_moment / (first_moment**2) - 1.0)
            SNR += snr_raw[~torch.isnan(snr_raw)].sum()
#             print((first_moment**2).sum())
        return Variance / N, SNR / N
        
        
# def SNR(models, ema, ema_iter):
#     for model in models:
#         if isinstance(model, torch.nn.Module):
#             for name, param in model.named_parameters():
#                 if param.requires_grad:
                    
                    
#     if iteration == 1:
#         for ind, p in enumerate(params):
#             grad = p.grad.cpu().detach()
#             fst_mmt.append(grad)
#             sec_mmt.append(grad**2)
#     else:
#         assert fst_mmt_old is not None, "ERROR! NoneType found."
#         assert sec_mmt_old is not None, "ERROR! NoneType found."
#         for ind, p in enumerate(params):
#             grad = p.grad.cpu().detach()
#             unbiased_1 = (1 - beta_1**(iteration-1)) / (1 - beta_1**iteration)
#             unbiased_2 = (1 - beta_2**(iteration-1)) / (1 - beta_2**iteration)
#             fst_mmt.append((beta_1 * fst_mmt_old[ind] + (1 - beta_1) * grad) * unbiased_1)
#             sec_mmt.append((beta_2 * sec_mmt_old[ind] + (1 - beta_2) * (grad**2)) * unbiased_2)
#     return fst_mmt, sec_mmt

# def snr_grad(fst_mmt, sec_mmt, EPS=1e-9):
#     """
#     compute the signal-to-noise ratio and variance (i.e.trace of the covariance matrix),
#     which are normalized by number of paramters
#     snr = 1 / (var / E[g^2]) = 1 / (1 - E^2[g] / E[g^2])
#     """
#     snr = None
#     var = None
#     N = 0.0
#     for ind in range(len(fst_mmt)):
#         var_p = sec_mmt[ind] - fst_mmt[ind]**2
#         snr_p = 1. / (1. - (fst_mmt[ind]**2+ EPS) / (sec_mmt[ind]+EPS))
#         if snr is None:
#             snr = snr_p.sum()
#         else:
#             snr = snr + snr_p.sum()

#         if var is None:
#             var = var_p.sum()
#         else:
#             var = var + var_p.sum()
#         N += var_p.numel()

#     return snr / N , var / N