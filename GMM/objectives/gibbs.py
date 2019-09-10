import torch
import torch.nn as nn
from utils import *
from normal_gamma import *
from forward_backward_test import *
import probtorch
"""
EUBO : loss function is  E_{p(z'|x)}[KL(p(z | z', x) || q_\f(z | z', x))]
                        + E_{p(z|x)}[KL(p(z' | z, x) || q_\f(z' | z, x))]
init_eta : initialize eta as the first step

"""

def Gibbs(models, obs, SubTest_Params):
    """
    The stepwise elbo
    """
    (device, S, B, N, K, D, mcmc_size) = SubTest_Params
    ELBOs = []
    DBs_eta = []
    DBs_z = []
    Ratios = []
    (gibbs_eta, gibbs_z) = models
    obs_tau, obs_mu, log_p = gibbs_eta.sample_prior(S, B)
    q_z, p_z = gibbs_z.forward(obs, obs_tau, obs_mu, N, K, S, B)
    state = q_z['zs'].value
    log_p_x = Log_likelihood(obs, state, obs_tau, obs_mu, K, D, cluster_flag=False)
    log_w = log_p_x.sum(-1) + p_z['zs'].log_prob.sum(-1) - q_z['zs'].log_prob.sum(-1)
    ELBOs.append(log_w.mean().unsqueeze(0))
    for m in range(mcmc_size):
        obs_tau_prev = obs_tau
        obs_mu_prev = obs_mu
        q_eta, p_eta, q_nu = gibbs_eta.forward(obs, state, K, D)
        obs_tau, obs_mu, log_w_eta_f, log_w_eta_b  = Incremental_eta_test(q_eta, p_eta, obs, state, K, D, obs_tau_prev, obs_mu_prev)
        DB_eta, eubo_p_q_eta, log_w_eta, w_f_eta = detailed_balances_test(log_w_eta_f, log_w_eta_b)
        state_prev = state
        q_z, p_z = gibbs_z.forward(obs, obs_tau, obs_mu, N, K, S, B)
        state, log_w_z_f, log_w_z_b = Incremental_z_test(q_z, p_z, obs, obs_tau, obs_mu, K, D, state_prev)
        DB_z, eubo_p_q_z, log_w_z, w_f_z = detailed_balances_test(log_w_z_f, log_w_z_b)
        ELBOs.append(ELBOs[-1] + (log_w_eta + log_w_z).mean().unsqueeze(0))
        Ratios.append((log_w_eta + log_w_z).mean().unsqueeze(0))
        DBs_eta.append(DB_eta.unsqueeze(0))
        DBs_z.append(DB_z.unsqueeze(0))
    return torch.cat(DBs_eta, 0), torch.cat(DBs_z, 0), torch.cat(ELBOs, 0), torch.cat(Ratios, 0)
