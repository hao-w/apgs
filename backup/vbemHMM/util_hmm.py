import autograd.numpy as np
from autograd.numpy.linalg import inv, det
from autograd import grad
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gamma, invwishart, dirichlet, multinomial
from scipy.special import logsumexp, digamma, loggamma
from scipy.special import gamma as gafun
from matplotlib.patches import Ellipse
from numpy.linalg import inv
from matplotlib.patches import Ellipse

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
# def forward(Pi, A, E, T, K):
#     Alpha = np.zeros((T, K))
#     Alpha[0] = np.multiply(Pi, E[0])
#     for t in range(1, T):
#         predictor = np.multiply(np.tile(Alpha[t-1], (K, 1)).T, A).sum(0)
#         Alpha[t] = np.multiply(E[t], predictor)
#     return Alpha
#
# def backward(A, E, T, K):
#     Beta = np.zeros((T, K))
#     Beta[T-1] = np.ones(K)
#     for t in range(T-1, 0):
#         emi_beta_expand = np.tile(np.multiply(Beta[t], E[t]), (K, 1))
#         Beta[t-1] = np.multiply(emi_beta_expand, A).sum(1)
#     return Beta
#
# def marginal_posterior(Alpha, Beta, T, K):
#     Gamma = np.multiply(Alpha, Beta)
#     sum_Gamma = Gamma.sum(1)
#     Gamma = Gamma.T / sum_Gamma
#     return Gamma.T
#
# def joint_posterior(Alpha, Beta, A, E, T, K):
#     Eta = np.zeros((T-1, K, K))
#     for t in range(T-1):
#         term1 = np.multiply(E[t+1], Beta[t+1])
#         term1.shape = (1, K)
#         term2 = Alpha[t]
#         term2.shape = (K, 1)
#         joint = np.multiply(np.dot(term2, term1), A)
#         Eta[t] = joint / joint.sum()
#     return Eta

## K is the number of latent states
## T is the length of the sequence
## Pi is K-length vector representing the initial state probability
## A is the K by K transition matrix, where a_ij = P(z_t = j | z_t-1 = i)
## so sum of each row equals 1.
## E is the T by K vector where e_tk = N(v_t ; mu_k, sigma_k)
def forward(log_Pi, log_A, log_E, T, K):
    log_Alpha = np.zeros((T, K))
    log_Alpha[0] = log_Pi + log_E[0]
    for t in range(1, T):
        predictor = logsumexp(np.add(np.tile(log_Alpha[t-1], (K, 1)).T, log_A), 0)
        log_Alpha[t] = np.add(log_E[t], predictor)
    return log_Alpha

def backward(log_A, log_E, T, K):
    log_Beta = np.zeros((T, K))
    log_Beta[T-1] = np.zeros(K)
    for t in range(T-1, 0):
        emi_beta_expand = np.tile(np.add(log_Beta[t], log_E[t]), (K, 1))
        log_Beta[t-1] = logsumexp(np.add(emi_beta_expand, log_A), 1)
    return log_Beta

def marginal_posterior(log_Alpha, log_Beta, T, K):
    log_gammas = np.add(log_Alpha, log_Beta)
    log_gammas = (log_gammas.T - logsumexp(log_gammas, 1)).T
    return log_gammas

def joint_posterior(log_Alpha, log_Beta, log_A, log_E, T, K):
    log_Eta = np.zeros((T-1, K, K))
    for t in range(T-1):
        term1 = np.add(log_E[t+1], log_Beta[t+1])
        term1.shape = (1, K)
        term1 = np.tile(term1, (K, 1))
        term2 = log_Alpha[t]
        term2.shape = (K, 1)
        term2 = np.tile(term2, (1, K))
        log_joint = np.add(np.add(term2, term1), log_A)
        log_Eta[t] = log_joint - logsumexp(log_joint)
    return log_Eta

def quad(a, B):
    return np.dot(np.dot(a.T, B), a)

def log_expectation_wi_single(nu, W, D):
    ds = (nu + 1 - (np.arange(D) + 1)) / 2.0
    return  - D * np.log(2) + np.log(det(W)) - digamma(ds).sum()

def log_expectation_wi_single2(nu, W, D):
    ds = (nu + 1 - (np.arange(D) + 1)) / 2.0
    return  D * np.log(2) + np.log(det(W)) + digamma(ds).sum()

def log_expectation_wi2(nu_ks, W_ks, D, K):
    log_expectations = np.zeros(K)
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single2(nu_ks[k], W_ks[k], D)
    return log_expectations


def log_expectation_wi(nu_ks, W_ks, D, K):
    log_expectations = np.zeros(K)
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
    return log_expectations

def quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    quadratic_expectations = np.zeros((K, N))
    for k in range(K):
        quadratic_expectations[k] = D / beta_ks[k] + nu_ks[k] * np.multiply(np.dot(Y - m_ks[k], inv(W_ks[k])), Y - m_ks[k]).sum(1)
    return quadratic_expectations.T

def log_expectations_dir(alpha_hat, K):
    log_expectations = np.zeros(K)
    sum_digamma = digamma(alpha_hat.sum())
    for k in range(K):
        log_expectations[k] = digamma(alpha_hat[k]) - sum_digamma
    return log_expectations

def vbE_step(alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    ## A K by K transition matrix, E is N by K emission matrix, both take log-form
    log_A = np.zeros((K, K))
    quadratic_expectations = quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K)

    log_expectation_lambda = log_expectation_wi(nu_ks, W_ks, D, K)
    log_E = - (D / 2) * np.log(2*np.pi) - (1 / 2) * log_expectation_lambda - (1 / 2) * quadratic_expectations

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

def bmm(a):
    return np.einsum('ijk,ilj->ikl', np.expand_dims(a, 1), np.expand_dims(a, 2))

def stats(log_gammas, Y, D, K):
    gammas = np.exp(log_gammas)
    N_ks = gammas.sum(0)
    Y_ks = np.zeros((K, D))
    S_ks = np.zeros((K, D, D))
    gammas_expanded = np.tile(gammas, (D, 1, 1))
    for k in range(K):
        Y_ks[k] = np.multiply(gammas_expanded[:, :, k].T, Y).sum(0) / N_ks[k]
        gammas_expanded2 = np.tile(gammas_expanded[:, :, k], (D, 1, 1))
        Y_bmm = np.swapaxes(bmm(Y - Y_ks[k]), 0, 2)
        S_ks[k] = np.multiply(gammas_expanded2, Y_bmm).sum(-1) / (N_ks[k])

    return N_ks, Y_ks, S_ks



def vbM_step(log_eta, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, N, D, K):
    eta = np.exp(log_eta)
    m_ks = np.zeros((K, D))
    W_ks = np.zeros((K, D, D))
    cov_ks = np.zeros((K, D, D))
    alpha_init_hat = alpha_init_0 + N_ks
    nu_ks = nu_0+ N_ks + 1
    beta_ks = beta_0 + N_ks
    alpha_trans_hat = alpha_trans_0 + eta.sum(0)

    for k in range(K):
        m_ks[k] = (beta_0 * m_0 + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = Y_ks[k] - m_0
        temp2.shape = (D, 1)
        W_ks[k] = W_0 + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * np.dot(temp2, temp2.T)
        cov_ks[k] = W_ks[k] / (nu_ks[k] - D - 1)
    return alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, cov_ks

def log_C(alpha):
    return loggamma(alpha.sum()) - (loggamma(alpha)).sum()

def log_wishart_B(nu, W, D):
    term1 =  (nu / 2) * np.log(det(W))
    term2 = - (nu * D / 2) * np.log(2)
    term3 = - (D * (D - 1) / 4) * np.log(np.pi)
    ds = (nu + 1 - (np.arange(D) + 1)) / 2.0
    term4 = - loggamma(ds).sum()
    return term1 + term2 + term3 + term4

# def entropy_wis(nu, W, D, E_log_sigma):
#     log_B = log_wishart_B(nu, W, D)
#     return - log_B - (nu - D - 1)/2 * E_log_sigma + nu * D / 2

def entropy_iw(nu, W, D, E_log_sigma):
    log_B = log_wishart_B(nu, W, D)
    return - log_B + (nu + D + 1)/2 * E_log_sigma + nu * D / 2

def kl_niw_true(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, K, S=500):
    kl_mu_lambda = 0.0
    E_log_qz = 0.0
    for k in range(K):
        log_sum = 0.0
        log_qz = 0.0
        log_pz = 0.0
        for s in range(S):
            sigma_k = invwishart.rvs(nu_ks[k], W_ks[k])
            mu_k = multivariate_normal.rvs(m_ks[k], sigma_k / (beta_ks[k]))

            log_p_sigma_k = invwishart.logpdf(sigma_k, nu_0, W_0)
            log_p_mu_k = multivariate_normal.logpdf(mu_k, m_0, sigma_k / beta_0)

            log_q_sigma_k = invwishart.logpdf(sigma_k, nu_ks[k], W_ks[k])
            log_q_mu_k = multivariate_normal.logpdf(mu_k, m_ks[k], sigma_k / beta_ks[k])

            # log_sum += log_q_sigma_k - log_p_sigma_k
            log_sum += log_q_sigma_k + log_q_mu_k - log_p_sigma_k - log_p_mu_k
            # log_qz += log_q_sigma_k + log_q_mu_k
            # log_pz = log_p_sigma_k + log_p_mu_k
        kl_mu_lambda += log_sum / S
        # E_log_qz += log_qz / S

        return kl_mu_lambda

def log_expectations_dir_trans(alpha_trans, K):
    E_log_trans = np.zeros((K, K))
    for k in range(K):
        E_log_trans[k] = log_expectations_dir(alpha_trans[k], K)
    return E_log_trans

def quad2(a, Sigma, D, K):
    batch_mul = np.zeros(K)
    for k in range(K):
        ak = a[k]
        ak.shape = (D, 1)
        batch_mul[k] = np.dot(np.dot(ak.T, Sigma[k]), ak)
    return batch_mul

def kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K):
    ## input W_ks need to be inversed
    W_ks_inv = np.zeros((K, D, D))
    for k in range(K):
        W_ks_inv[k] = inv(W_ks[k])
    E_log_det_sigma = log_expectation_wi(nu_ks, W_ks, D, K)
    trace_W0_Wk_inv = np.zeros(K)
    entropy = 0.0
    for k in range(K):
        # log_B[k] = log_wishart_B(nu_ks[k], W_ks[k], D)
        trace_W0_Wk_inv[k] = np.diag(np.dot(W_0, W_ks_inv[k])).sum()
        entropy += entropy_iw(nu_ks[k], W_ks[k], D, E_log_det_sigma[k])
    log_B_0 = log_wishart_B(nu_0, inv(W_0), D)

    E_log_q_phi = (D / 2) * (np.log(beta_ks / (2 * np.pi)) - 1).sum() - entropy
    # E_log_q_phi = - entropy
    quad_mk_m0_Wk = quad2(m_ks - m_0, W_ks_inv, D, K)
    E_log_p_phi = (1 / 2) * ((np.log(beta_0/(2*np.pi)) - (beta_0 / beta_ks)).sum() * D - np.multiply(nu_ks, quad_mk_m0_Wk).sum() * beta_0)
    E_log_p_phi +=  K * log_B_0 - ((nu_0 + D + 1) / 2) * E_log_det_sigma.sum() - (1/2) * np.multiply(nu_ks, trace_W0_Wk_inv).sum()
    kl_phi = E_log_q_phi - E_log_p_phi
    return kl_phi

# def kl_nw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K):
#     ## input W_ks need to be inversed
#     W_ks_inv = np.zeros((K, D, D))
#     for k in range(K):
#         W_ks_inv[k] = inv(W_ks[k])
#     E_log_det_lambda = log_expectation_wi2(nu_ks, W_ks_inv, D, K)
#     trace_W0_Wk_inv = np.zeros(K)
#     entropy = 0.0
#     for k in range(K):
#         # log_B[k] = log_wishart_B(nu_ks[k], W_ks[k], D)
#         trace_W0_Wk_inv[k] = np.diag(np.dot(W_0, W_ks_inv[k])).sum()
#         entropy += entropy_wis(nu_ks[k], W_ks_inv[k], D, E_log_det_lambda[k])
#     log_B_0 = log_wishart_B(nu_0, inv(W_0), D)
#
#     # E_log_q_phi = (D / 2) * (np.log(beta_ks / (2 * np.pi)) - 1).sum() - entropy
#     E_log_q_phi = - entropy
#     quad_mk_m0_Wk = quad2(m_ks - m_0, W_ks_inv, D, K)
#     E_log_p_phi = (1 / 2) * ((np.log(beta_0/(2*np.pi)) - (beta_0 / beta_ks)).sum() * D - np.multiply(nu_ks, quad_mk_m0_Wk).sum() * beta_0)
#     # E_log_p_phi += K * log_B_0 + ((nu_0 - D - 1) / 2) * E_log_det_lambda.sum() - (1/2) * np.multiply(nu_ks, trace_W0_Wk_inv).sum()
#     kl_phi = E_log_q_phi - E_log_p_phi
#     return kl_phi

def elbo(log_gammas, log_eta, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    gammas = np.exp(log_gammas)
    eta = np.exp(log_eta)
    E_log_pi_hat = log_expectations_dir(alpha_init_hat, K)
    E_log_trans = log_expectations_dir_trans(alpha_trans_hat, K)

    ## kl between pz and qz
    E_log_pz = np.multiply(gammas[0], E_log_pi_hat).sum() + np.multiply(np.tile(E_log_trans, (N-1, 1, 1)), eta).sum()
    E_log_qz = np.multiply(log_gammas, gammas).sum()
    kl_z = E_log_qz - E_log_pz

    ## kl between p_pi and q_pi
    kl_pi = log_C(alpha_init_hat) - log_C(alpha_init_0) + np.multiply(alpha_init_hat - alpha_init_0, E_log_pi_hat).sum()

    ## kl between p_A and q_A
    kl_trans = 0.0
    for j in range(K):
        kl_trans += log_C(alpha_trans_hat[j]) - log_C(alpha_trans_0[j]) + np.multiply(alpha_trans_hat[j] - alpha_trans_0[j], E_log_trans[j]).sum()

    # kl between q_phi and p_phi
    # kl_phi = kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K)
    # print('E_log_qz : %f' % E_q_phi)
    # true_kl_phi = kl_niw_true(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, K, S=500)
    # print('true kl phi : %f' % true_kl_phi)
    kl_phi = kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K)
    print(kl_phi)
    ##likelihood term
    log_likelihood = 0.0

    for k in range(K):
        E_log_det_sigma_k = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
        Ykmean_diff = Y_ks[k] - m_ks[k]
        Ykmean_diff.shape = (D, 1)
        log_likelihood += N_ks[k] * (- E_log_det_sigma_k - (D / beta_ks[k]) - nu_ks[k] * (np.diag(np.dot(S_ks[k], inv(W_ks[k]))).sum()) - nu_ks[k] * quad(Ykmean_diff, inv(W_ks[k])) - D * np.log(2*np.pi))

    log_likelihood *= 1 / 2
    # print(kl_phi, kl_phi_true)
    print('NLL : %f, KL_z : %f, KL_pi : %f, KL_trans : %f, KL_phi %f' % (log_likelihood[0][0] , kl_z , kl_pi , kl_trans, kl_phi))
    Elbo = log_likelihood - kl_z - kl_pi - kl_phi - kl_trans
    # Elbo = 0
    return Elbo
