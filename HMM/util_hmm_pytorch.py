import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch
from probtorch.util import log_sum_exp
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import Range1d
from bokeh.io import push_notebook, show, output_notebook
import time


def bouncing(STATE):
    box_bound = np.array([-100, 100, -100, 100])
    plot = figure(plot_width=300, plot_height=300)
    plot.x_range = Range1d(box_bound[0], box_bound[1])
    plot.y_range = Range1d(box_bound[2], box_bound[3])
    c = plot.circle(x=[STATE[0, 0]], y=[STATE[0, 1]])
    target = show(plot, notebook_handle=True)
    for t in range(STATE.shape[0]-1):
        c.data_source.data['x'] = [STATE[t+1, 0]]
        c.data_source.data['y'] = [STATE[t+1, 1]]
        push_notebook(handle=target)
        time.sleep(0.2)


def output_transition(alpha_trans_hat):
    A = alpha_trans_hat.data.numpy()
    A = (A.T / A.sum(1)).T
    return A

def step(state, dt, box_bound):
    xy_new = state[ :2] + state[2: ] * dt + np.random.randn(2)
    while(True):
        if xy_new[0] < box_bound[0]:
            x_over = abs(box_bound[0] - xy_new[0])
            xy_new[0] = box_bound[0] + x_over
            state[2] *= -1
        elif xy_new[0] > box_bound[1]:
            x_over = abs(box_bound[1] - xy_new[0])
            xy_new[0] = box_bound[1] - x_over
            state[2] *= -1
        elif xy_new[1] < box_bound[2]:
            y_over = abs(box_bound[2] - xy_new[1])
            xy_new[1] = box_bound[2] + y_over
            state[3] *= -1
        elif xy_new[1] > box_bound[3]:
            y_over = abs(box_bound[3] - xy_new[1])
            xy_new[1] = box_bound[3] - y_over
            state[3] *= -1
        else:
            state[:2] = xy_new
            break
    return state

def compute_hidden(velocity):
    if velocity[0] > 0 and velocity[1] > 0:
        return 0
    if velocity[0] > 0 and velocity[1] < 0:
        return 1
    if velocity[0] < 0 and velocity[1] < 0:
        return 2
    if velocity[0] < 0 and velocity[1] > 0:
        return 3

def transition(A, old_hidden, new_hidden):
    A[old_hidden, new_hidden] += 1
    return A

def generate_data(T, dt, init_state, plot_flag=False):
    A = np.zeros((4,4))
    STATE = np.zeros((T+1, 4))
    box_bound = np.array([-100, 100, -100, 100])
    if plot_flag:
        plot = figure(plot_width=300, plot_height=300)
        plot.x_range = Range1d(box_bound[0], box_bound[1])
        plot.y_range = Range1d(box_bound[2], box_bound[3])
        c = plot.circle(x=[init_state[0]], y=[init_state[1]])
        target = show(plot, notebook_handle=True)
    state = init_state
    STATE[0] = state
#     old_hidden = init_hidden(init_state[2:])
    for i in range(T):
        state = step(state, dt, box_bound)
        #print(state[2:])
        STATE[i+1] = state
        if i == 0:
            old_hidden = compute_hidden(state[2:])
        else:
            new_hidden = compute_hidden(state[2:])
            A = transition(A, old_hidden, new_hidden)
            old_hidden = new_hidden
        if plot_flag:
            c.data_source.data['x'] = [state[0]]
            c.data_source.data['y'] = [state[1]]
            push_notebook(handle=target)
            time.sleep(0.2)
    Y = (STATE[1:] - STATE[:T])[:, :2]
    A = (A.T / A.sum(1)).T
    return Y, A, STATE


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def plot_clusters(Xs, mus, covs):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    plot_cov_ellipse(cov=covs[0], pos=mus[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs[1], pos=mus[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs[2], pos=mus[2], nstd=2, ax=ax, alpha=0.5)
    plt.show()

## K is the number of latent states
## T is the length of the sequence
## Pi is K-length vector representing the initial state probability
## A is the K by K transition matrix, where a_ij = P(z_t = j | z_t-1 = i)
## so sum of each row equals 1.
## E is the T by K vector where e_tk = N(v_t ; mu_k, sigma_k)


def forward(log_Pi, log_A, log_E, T, K):
    log_Alpha = torch.zeros((T, K)).double()
    log_Alpha[0] = log_Pi + log_E[0]
    for t in range(1, T):
        predictor = log_sum_exp(log_Alpha[t-1].repeat(K, 1).transpose(0,1) + log_A, 0)
        log_Alpha[t] = log_E[t] + predictor
    return log_Alpha

def backward(log_A, log_E, T, K):
    log_Beta = torch.zeros((T, K)).double()
    log_Beta[T-1] = torch.zeros(K).double()
    for t in range(T-1, 0):
        emi_beta_expand = (log_Beta[t] + log_E[t]).repeat(K, 1)
        log_Beta[t-1] = log_sum_exp(emi_beta_expand + log_A, 1)
    return log_Beta

def marginal_posterior(log_Alpha, log_Beta, T, K):
    log_gammas = log_Alpha + log_Beta
    log_gammas = (log_gammas.transpose(0,1) - log_sum_exp(log_gammas, 1)).transpose(0,1)
    return log_gammas

def joint_posterior(log_Alpha, log_Beta, log_A, log_E, T, K):
    log_Eta = torch.zeros((T-1, K, K)).double()
    for t in range(T-1):
        term1 = (log_E[t+1] + log_Beta[t+1]).view(1, K).repeat(K, 1)
        term2 = log_Alpha[t].view(K, 1).repeat(1, K)
        log_joint = term2 + term1 + log_A
        log_Eta[t] = log_joint - log_sum_exp(log_joint)
    return log_Eta

def quad(a, B):
    return torch.mm(torch.mm(a.transpose(0, 1), B), a)

def log_expectation_wi_single(nu, W, D):
    ds = (nu + 1 - (torch.arange(D).double() + 1)) / 2.0
    return  - D * torch.log(torch.Tensor([2]).double()) + torch.log(torch.det(W)) - torch.digamma(ds).sum()

def log_expectation_wi_single2(nu, W, D):
    ds = (nu + 1 - (torch.arange(D).double() + 1)) / 2.0
    return  D * torch.log(torch.Tensor([2]).double()) + torch.log(torch.det(W)) + torch.digamma(ds).sum()

def log_expectation_wi2(nu_ks, W_ks, D, K):
    log_expectations = torch.zeros(K).double()
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single2(nu_ks[k], W_ks[k], D)
    return log_expectations

def log_expectation_wi(nu_ks, W_ks, D, K):
    log_expectations = torch.zeros(K).double()
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
    return log_expectations

def quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    quadratic_expectations = torch.zeros((K, N)).double()
    for k in range(K):
        quadratic_expectations[k] = D / beta_ks[k] + nu_ks[k] * torch.mul(torch.mm(Y - m_ks[k], torch.inverse(W_ks[k])), Y - m_ks[k]).sum(1)
    return quadratic_expectations.transpose(0,1)

def log_expectations_dir(alpha_hat, K):
    log_expectations = torch.zeros(K).double()
    sum_digamma = torch.digamma(alpha_hat.sum())
    for k in range(K):
        log_expectations[k] = torch.digamma(alpha_hat[k]) - sum_digamma
    return log_expectations

def vbE_step(alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    ## A K by K transition matrix, E is N by K emission matrix, both take log-form
    log_A = torch.zeros((K, K)).double()
    quadratic_expectations = quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K)

    log_expectation_lambda = log_expectation_wi(nu_ks, W_ks, D, K)
    log_E = - (D / 2) * torch.log(torch.Tensor([2*math.pi]).double()) - (1 / 2) * log_expectation_lambda - (1 / 2) * quadratic_expectations

    for j in range(K):
        log_A[j] = log_expectations_dir(alpha_trans_hat[j], K)
    ## Pi is the initial distribution in qz
    log_Pi = log_expectations_dir(alpha_init_hat, K)
    # log_Pi_q = log_Pi - logsumexp(log_Pi)
    log_Alpha = forward(log_Pi, log_A, log_E, N, K)
    log_Beta = backward(log_A, log_E, N, K)
    log_gammas = marginal_posterior(log_Alpha, log_Beta, N, K)
    log_Eta = joint_posterior(log_Alpha, log_Beta, log_A, log_E, N, K)
    return log_gammas, log_Eta

def stats(log_gammas, Y, D, K):
    gammas = torch.exp(log_gammas)
    N_ks = gammas.sum(0)
    Y_ks = torch.zeros((K, D)).double()
    S_ks = torch.zeros((K, D, D)).double()
    gammas_expanded = gammas.repeat(D, 1, 1)
    for k in range(K):
        Y_ks[k] = torch.mul(gammas_expanded[:, :, k].transpose(0,1), Y).sum(0) / N_ks[k]
        gammas_expanded2 = gammas_expanded[:, :, k].repeat(D, 1, 1).permute(2, 1, 0)
        Y_diff = Y - Y_ks[k]
        Y_bmm = torch.bmm(Y_diff.unsqueeze(2), Y_diff.unsqueeze(1))
        S_ks[k] = torch.mul(gammas_expanded2, Y_bmm).sum(0) / (N_ks[k])
    return N_ks, Y_ks, S_ks

def vbM_step(log_eta, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, N, D, K):
    eta = torch.exp(log_eta).double()
    m_ks = torch.zeros((K, D)).double()
    W_ks = torch.zeros((K, D, D)).double()
    cov_ks = torch.zeros((K, D, D)).double()
    alpha_init_hat = alpha_init_0 + N_ks
    nu_ks = nu_0+ N_ks + 1
    beta_ks = beta_0 + N_ks
    alpha_trans_hat = alpha_trans_0 + eta.sum(0)
    for k in range(K):
        m_ks[k] = (beta_0 * m_0 + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = (Y_ks[k] - m_0).view(D, 1)
        W_ks[k] = W_0 + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * torch.mul(temp2, temp2.transpose(0, 1))
        cov_ks[k] = W_ks[k] / (nu_ks[k] - D - 1)
    return alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, cov_ks

def log_C(alpha):
    return torch.lgamma(alpha.sum()) - (torch.lgamma(alpha)).sum()

def log_wishart_B(nu, W, D):
    term1 =  (nu / 2) * torch.log(torch.det(W))
    term2 = - (nu * D / 2) * torch.log(torch.Tensor([2])).double()
    term3 = - (D * (D - 1) / 4) * torch.log(torch.Tensor([math.pi])).double()
    ds = (nu + 1 - (torch.arange(D).double() + 1)) / 2.0
    term4 = - torch.lgamma(ds).sum()
    return term1 + term2 + term3 + term4

def entropy_iw(nu, W, D, E_log_sigma):
    log_B = log_wishart_B(nu, W, D)
    return - log_B + (nu + D + 1)/2 * E_log_sigma + nu * D / 2

def log_expectations_dir_trans(alpha_trans, K):
    E_log_trans = torch.zeros((K, K)).double()
    for k in range(K):
        E_log_trans[k] = log_expectations_dir(alpha_trans[k], K)
    return E_log_trans

def quad2(a, Sigma, D, K):
    batch_mul = torch.zeros(K).double()
    for k in range(K):
        ak = a[k].view(D, 1)
        batch_mul[k] = quad(ak, Sigma[k])
    return batch_mul

def kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K):
    ## input W_ks need to be inversed
    W_ks_inv = torch.zeros((K, D, D)).double()
    for k in range(K):
        W_ks_inv[k] = torch.inverse(W_ks[k])
    E_log_det_sigma = log_expectation_wi(nu_ks, W_ks, D, K)
    trace_W0_Wk_inv = torch.zeros(K).double()
    entropy = 0.0
    for k in range(K):
        trace_W0_Wk_inv[k] = torch.diag(torch.mm(W_0, W_ks_inv[k])).sum()
        entropy += entropy_iw(nu_ks[k], W_ks[k], D, E_log_det_sigma[k])
    log_B_0 = log_wishart_B(nu_0, torch.inverse(W_0), D)

    E_log_q_phi = (D / 2) * (np.log(beta_ks / (2 * np.pi)) - 1).sum() - entropy
    # E_log_q_phi = - entropy
    quad_mk_m0_Wk = quad2(m_ks - m_0, W_ks_inv, D, K)
    E_log_p_phi = ((1 / 2) * ((np.log(beta_0/(2*np.pi)) - (beta_0 / beta_ks)).sum() * D - torch.mul(nu_ks, quad_mk_m0_Wk).sum() * beta_0)).item()
    E_log_p_phi +=  (K * log_B_0 - ((nu_0 + D + 1) / 2) * E_log_det_sigma.sum() - (1/2) * torch.mul(nu_ks, trace_W0_Wk_inv).sum()).item()
    kl_phi = E_log_q_phi - E_log_p_phi
    return kl_phi

def elbo(log_gammas, log_eta, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    gammas = torch.exp(log_gammas)
    eta = torch.exp(log_eta)
    E_log_pi_hat = log_expectations_dir(alpha_init_hat, K)
    E_log_trans = log_expectations_dir_trans(alpha_trans_hat, K)

    ## kl between pz and qz
    E_log_pz = torch.mul(gammas[0], E_log_pi_hat).sum() + torch.mul(E_log_trans.repeat(N-1, 1, 1), eta).sum()
    E_log_qz =torch.mul(log_gammas, gammas).sum()
    kl_z = E_log_qz - E_log_pz

    ## kl between p_pi and q_pi
    kl_pi = log_C(alpha_init_hat) - log_C(alpha_init_0) + torch.mul(alpha_init_hat - alpha_init_0, E_log_pi_hat).sum()

    ## kl between p_A and q_A
    kl_trans = 0.0
    for j in range(K):
        kl_trans += log_C(alpha_trans_hat[j]) - log_C(alpha_trans_0[j]) + torch.mul(alpha_trans_hat[j] - alpha_trans_0[j], E_log_trans[j]).sum()

    # kl between q_phi and p_phi
    # kl_phi = kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K)
    # print('E_log_qz : %f' % E_q_phi)
    # true_kl_phi = kl_niw_true(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, K, S=500)
    # print('true kl phi : %f' % true_kl_phi)
    kl_phi = kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K)
    ##likelihood term
    log_likelihood = 0.0
    for k in range(K):
        E_log_det_sigma_k = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
        Ykmean_diff = (Y_ks[k] - m_ks[k]).view(D, 1)
        log_likelihood += N_ks[k] * (- E_log_det_sigma_k - (D / beta_ks[k]) - nu_ks[k] * (torch.diag(torch.mm(S_ks[k], torch.inverse(W_ks[k]))).sum()) - nu_ks[k] * quad(Ykmean_diff, torch.inverse(W_ks[k])) - D * torch.log(torch.Tensor([2*np.pi])).double())
    log_likelihood *= 1 / 2
    # print(kl_phi, kl_phi_true)
    # print('NLL : %f, KL_z : %f, KL_pi : %f, KL_trans : %f, KL_phi %f' % (log_likelihood[0][0] , kl_z , kl_pi , kl_trans, kl_phi))
    Elbo = log_likelihood - kl_z - kl_pi - kl_phi - kl_trans
    # Elbo = 0
    return Elbo

def output_transition(alpha_trans_hat):
    A = alpha_trans_hat.data.numpy()
    A = (A.T / A.sum(1)).T
    return A
