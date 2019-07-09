import torch
import torch.nn as nn
from utils import resample
from forward_backward_dec import *
import probtorch
"""
EUBO : loss function is  E_{p(z'|x)}[KL(p(z | z', x) || q_\f(z | z', x))]
                        + E_{p(z|x)}[KL(p(z' | z, x) || q_\f(z' | z, x))]
init_eta : initialize eta as the first step

"""
def AG_dec(models, ob, K, mcmc_size, device):
    """
    initialize eta, using prior,
    train neural Gibbs samplers for mu and angle variable
    Gibbs sampler is used for z
    """
    S, B, N, D = ob.shape
    (oneshot_mu, gibbs_state, enc_mu, enc_angle, dec_x) = models
    with torch.cuda.device(device):
        eubos = torch.zeros(mcmc_size+1).cuda()
        elbos = torch.zeros(mcmc_size+1).cuda()
        esss = torch.zeros(mcmc_size+1).cuda()
    state, angle, mu, w, eubo, elbo, _, _ = oneshot(oneshot_mu, gibbs_state, dec_x, ob, K)
    eubos[0] = eubo  ## weights S * B
    elbos[0] = elbo
    esss[0] = (1. / (w ** 2).sum(0)).mean()
    for m in range(mcmc_size):
        if m == 0:
            state = resample(state, w, idw_flag=False) ## resample state
        else:
            state = resample(state, w_state, idw_flag=True)
        ## update mu
        mu, w_mu, eubo_mu, elbo_mu, _, _  = Update_mu(enc_mu, dec_x, ob, state, angle, mu, K)
        mu = resample(mu, w_mu, idw_flag=True)
        ##update angle
        angle, w_angle, eubo_angle, elbo_angle, _, _ = Update_angle(enc_angle, dec_x, ob, state, mu, angle)
        angle = resample(angle, w_angle, idw_flag=True)
        ## update z
        state, w_state, eubo_state, elbo_state, _, _ = Update_state(gibbs_state, dec_x, ob, angle, mu, state)
        eubos[m+1] = eubo_mu + eubo_angle
        elbos[m+1] = elbo_mu + elbo_angle
        esss[m+1] = ((1. / (w_mu**2).sum(0)).mean() + (1. / (w_state**2).sum(0)).mean() + (1. / (w_angle**2).sum(0)).mean()) / 3
        # print(esss[m+1])
    return eubos.sum(), elbos.sum(), esss.mean()
