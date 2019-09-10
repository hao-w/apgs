import torch
import torch.nn as nn
from utils import resample
from forward_backward_pcg_dec import *
import probtorch

def OS_pcg(models, ob, K, device):
    """
    initialize eta, using prior,
    train neural Gibbs samplers for mu and angle variable
    Gibbs sampler is used for z
    """
    S, B, N, D = ob.shape
    (oneshot_mu, oneshot_state, enc_angle, dec_x) = models
    state, angle, mu, w, eubo, elbo, theta_loss, _, _, _ = oneshot(oneshot_mu, oneshot_state, enc_angle, dec_x, ob, K)
    eubos = eubo  ## weights S * B
    theta_losss = theta_loss
    elbos = elbo.detach()
    esss = (1. / (w ** 2).sum(0)).mean()

    return eubos.sum(), elbos.sum(), theta_losss.sum(), esss.mean()
