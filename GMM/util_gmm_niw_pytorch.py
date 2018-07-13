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

def shuffler(train_dataset):
    index = np.arange(train_dataset.shape[0])
    np.random.shuffle(index)
    return train_dataset[index, :]

def NUM_ITERS(train_dataset, batch_size):
    remainder = train_dataset.shape[0] % batch_size
    if remainder == 0:
        num_iters = int(train_dataset.shape[0] / batch_size)
    else:
        num_iters = int(((train_dataset.shape[0] - remainder) / batch_size) + 1)
    return num_iters
    
def load_iris(filename):
    features = ['sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)',
                'species']
    species = ['Iris-setosa',
               'Iris-versicolor',
               'Iris-virginica']

    iris_data = []
    f = open(filename, "r")
    for data in f:
        data = data.strip().split(",")
        if len(data) != 1:
            if data[4] == species[0]:
                label = 0
            elif data[4] == species[1]:
                label = 1
            else:
                label = 2
            iris_data.append(list(map(lambda x: float(x), data[0:4])) + [label])
    iris_data = np.array(iris_data)
    Y = iris_data[:,0:4]
    (N,D) = Y.shape
    return Y, N, D

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


def empirical_cluster(Y, N, D, K):
    # True means and covs and Z
    true_mus = np.zeros((K,D))
    true_covs = np.zeros((K,D,D))
    true_Z = np.zeros(N)

    true_mus[0] = np.mean(Y[:50],0)
    true_mus[1] = np.mean(Y[50:100],0)
    true_mus[2] = np.mean(Y[100:150],0)

    true_covs[0] = np.cov(Y[:50].T)
    true_covs[1] = np.cov(Y[50:100].T)
    true_covs[2] = np.cov(Y[100:150].T)

    true_Z[0:50] = 0
    true_Z[50:100] = 1
    true_Z[100:] = 2
    return true_mus, true_covs, true_Z

def plot_clusters(Y, mus, covs, K):
    cmap = plt.cm.get_cmap('hsv', K)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(Y[:,0], Y[:,1], 'k.', markersize=4.0)
    for i in range(K):
        plot_cov_ellipse(cov=covs[i],
                         pos=mus[i],
                         nstd=2,
                         ax=ax,
                         color=cmap(i),
                         alpha=0.1)

    plt.show()


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

def vbE_step(alpha_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    ## return gammas_nk N by K
    quadratic_expectations = quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K)
    log_expectation_lambda = log_expectation_wi(nu_ks, W_ks, D, K)
    log_expectations_pi = log_expectations_dir(alpha_hat, K)
    log_rhos = log_expectations_pi - (D / 2) * torch.log(torch.Tensor([2*math.pi]).double()) - (1 / 2) * log_expectation_lambda - (1 / 2) * quadratic_expectations

    log_gammas = (log_rhos.transpose(0,1) - log_sum_exp(log_rhos, 1)).transpose(0,1)
    return log_gammas


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


def vbM_step(alpha_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, N, D, K):
    m_ks = torch.zeros((K, D)).double()
    W_ks = torch.zeros((K, D, D)).double()
    cov_ks = torch.zeros((K, D, D)).double()
    alpha_hat = alpha_0 + N_ks
    nu_ks = nu_0+ N_ks + 1
    beta_ks = beta_0 + N_ks

    for k in range(K):
        m_ks[k] = (beta_0 * m_0 + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = (Y_ks[k] - m_0).view(D, 1)
        W_ks[k] = W_0 + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * torch.mul(temp2, temp2.transpose(0, 1))
        cov_ks[k] = W_ks[k] / (nu_ks[k] - D - 1)
    return alpha_hat, nu_ks, W_ks, m_ks, beta_ks, cov_ks

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

def elbo(log_gammas, alpha_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, alpha_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    gammas = torch.exp(log_gammas)
    E_log_pi_hat = log_expectations_dir(alpha_hat, K)
    E_log_pi_hat_expanded = E_log_pi_hat.repeat(N, 1)
    ## kl between pz and qz
    E_log_pz = torch.mul(E_log_pi_hat_expanded, gammas).sum()
    E_log_qz =torch.mul(log_gammas, gammas).sum()
    kl_z = E_log_qz - E_log_pz

    ## kl between p_pi and q_pi
    kl_pi = log_C(alpha_hat) - log_C(alpha_0) + torch.mul(alpha_hat - alpha_0, E_log_pi_hat).sum()


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
    print('NLL : %f, KL_z : %f, KL_pi : %f, KL_phi %f' % (log_likelihood[0][0] , kl_z , kl_pi, kl_phi))
    Elbo = log_likelihood - kl_z - kl_pi - kl_phi
    # Elbo = 0
    return Elbo
