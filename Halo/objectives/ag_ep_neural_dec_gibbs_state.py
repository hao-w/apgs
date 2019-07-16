import torch
import torch.nn as nn
from utils import resample
from forward_backward_neural_dec_gibbs_state import *
import probtorch

def AG_dec(models, ob, K, mcmc_size, device):
    """
    initialize eta, using prior,
    train neural Gibbs samplers for mu and angle variable
    Gibbs sampler is used for z
    """
    S, B, N, D = ob.shape
    (oneshot_mu, oneshot_state, enc_angle, enc_mu, gibbs_state, dec_x) = models
    with torch.cuda.device(device):
        eubos = torch.zeros(mcmc_size+1).cuda()
        elbos = torch.zeros(mcmc_size+1).cuda()
        losss_theta = torch.zeros(mcmc_size+1).cuda()
        esss = torch.zeros(mcmc_size+1).cuda()
    state, angle, mu, w, eubo, elbo, loss_theta, _, _ = oneshot(oneshot_mu, oneshot_state, enc_angle, dec_x, ob, K)
    eubos[0] = eubo  ## weights S * B
    elbos[0] = elbo
    losss_theta[0] = loss_theta
    esss[0] = (1. / (w ** 2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample(state, w, idw_flag=False) ## resample state
            angle = resample(angle, w, idw_flag=False)
        else:
            angle = resample(angle, w_angle, idw_flag=True)
        ## update mu
        mu, w_mu, eubo_mu, elbo_mu, loss_theta_mu, _, _  = Update_mu(enc_mu, dec_x, ob, state, angle, mu, K)
        mu = resample(mu, w_mu, idw_flag=True)
        ## update z
        q_state, p_state = Update_state(gibbs_state, ob, angle, mu)
        state = q_state['zs'].value
        ##update angle
        angle, w_angle, eubo_angle, elbo_angle, loss_theta_angle, _, _ = Update_angle(enc_angle, dec_x, ob, state, mu, angle)

        eubos[m+1] = eubo_mu + eubo_angle
        elbos[m+1] = elbo_mu + elbo_angle
        losss_theta[m+1] = loss_theta_mu + loss_theta_angle
        esss[m+1] = ((1. / (w_mu**2).sum(0)).mean() + (1. / (w_angle**2).sum(0)).mean()) / 2
        # print(esss[m+1])
    return eubos.sum(), elbos.sum(), losss_theta.sum(), esss.mean()
