import torch
import torch.nn as nn
from utils import resample
from forward_backward_dec import *
import probtorch

def AG_pcg(models, ob, K, mcmc_size, device):
    """
    initialize eta, using prior,
    train neural Gibbs samplers for mu and angle variable
    Gibbs sampler is used for z
    """
    S, B, N, D = ob.shape
    (oneshot_mu, oneshot_state, enc_angle, enc_mu, dec_x) = models
    with torch.cuda.device(device):
        eubos = torch.zeros(mcmc_size+1).cuda()
        elbos = torch.zeros(mcmc_size+1).cuda()
        esss = torch.zeros(mcmc_size+1).cuda()
        theta_losss = torch.zeros(mcmc_size+1).cuda()
    state, angle, mu, w, eubo, elbo, theta_loss, _, _, _ = Oneshot(oneshot_mu, oneshot_state, enc_angle, dec_x, ob, K)
    eubos[0] = eubo  ## weights S * B
    theta_losss[0] = theta_loss
    elbos[0] = elbo.detach()
    esss[0] = (1. / (w ** 2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample(state, w, idw_flag=False) ## resample state
            angle = resample(angle, w, idw_flag=False)
        else:
            state = resample(state, w_local, idw_flag=True)
            angle = resample(angle, w_local, idw_flag=True)
        ## update mu
        mu, w_mu, eubo_mu, elbo_mu, theta_loss_mu, _  = Update_mu(enc_mu, dec_x, ob, state, angle, mu)
        mu = resample(mu, w_mu, idw_flag=True)
        ## update z
        state, angle, w_local, eubo_local, elbo_local, theta_loss_local, _, _, _ = Update_state_angle(oneshot_state, enc_angle, dec_x, ob, state, angle, mu)
        eubos[m+1] = eubo_mu + eubo_local
        elbos[m+1] = elbo_mu + elbo_local
        theta_losss[m+1] = theta_loss_mu + theta_loss_local
        esss[m+1] = ((1. / (w_mu**2).sum(0)).mean() + (1. / (w_local**2).sum(0)).mean()) / 2
    return eubos.sum(), elbos.sum(), theta_losss.sum(), esss.mean()
