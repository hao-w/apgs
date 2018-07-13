import autograd.numpy as np
from autograd.numpy.linalg import inv, det
from autograd import grad
from functools import reduce
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gamma, invgamma
from scipy.special import logsumexp, digamma, loggamma
from scipy.special import gamma as gafun
from matplotlib.patches import Ellipse
from numpy.linalg import inv

markersize = 4

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

def plot_cov_ellipse(cov, pos, nstd=2.0, ax=None, **kwargs):
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

# def plot_clusters(Y, mus, covs, Z):
#     features = ['sepal length (cm)',
#             'sepal width (cm)',
#             'petal length (cm)',
#             'petal width (cm)',
#             'species']
#     species = ['Iris-setosa',
#                'Iris-versicolor',
#                'Iris-virginica']
#     f, axarr = plt.subplots(4, 4, sharex='col', sharey='row',figsize=(15, 15))
#     axarr[3,0].set_xlabel('Sepal length')
#     for i in range(4):
#         axarr[3,i].set_xlabel(features[i], fontsize=15)
#         axarr[i,0].set_ylabel(features[i], fontsize=15)
#         for j in range(4):
#             if  i == j:
#                 featurei_data = np.stack((Y[:50,i],Y[50:100,i],Y[100:150,i])).T
#                 axarr[j,i].hist(featurei_data, bins=20, histtype='bar',
#                                 color=['red', 'blue', 'green'],
#                                 stacked=True, density=True)
#             else:
#                 cluster0_indices = (Z == 0)
#                 cluster1_indices = (Z == 1)
#                 cluster2_indices = (Z == 2)
#
#                 axarr[j,i].plot(Y[cluster0_indices,i],
#                                 Y[cluster0_indices,j],
#                                 'ro', mew=0.5, label=species[0][5:])
#                 axarr[j,i].plot(Y[cluster1_indices,i],
#                                 Y[cluster1_indices,j],
#                                 'bo', mew=0.5, label=species[1][5:])
#                 axarr[j,i].plot(Y[cluster2_indices,i],
#                                 Y[cluster2_indices,j],
#                                 'go', mew=0.5, label=species[2][5:])
#
#                 plot_cov_ellipse(cov=covs[0,[i,i,j,j],[i,j,i,j]].reshape(2,2),
#                                  pos=mus[0,[i,j]],
#                                  nstd=2,
#                                  ax=axarr[j,i],
#                                  color='red',
#                                  alpha=0.1)
#                 plot_cov_ellipse(cov=covs[1,[i,i,j,j],[i,j,i,j]].reshape(2,2),
#                                  pos=mus[1,[i,j]],
#                                  nstd=2,
#                                  ax=axarr[j,i],
#                                  color='blue',
#                                  alpha=0.1)
#                 plot_cov_ellipse(cov=covs[2,[i,i,j,j],[i,j,i,j]].reshape(2,2),
#                                  pos=mus[2,[i,j]],
#                                  nstd=2,
#                                  ax=axarr[j,i],
#                                  color='green',
#                                  alpha=0.1)
#
#     #f.legend(loc = 'upper right', fontsize=20)
#     plt.show()

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



# def log_det_expectation(W_ks, nu_ks, D, K):
#     """
#     Given the nus and covariences,
#     compute the expression 'A' as shown above (for every k)
#     """
#
#     log_det_ks = np.zeros(K)
#     for k in range(K):
#         D_sum = 0.0
#         for d in range(D):
#             D_sum += digamma((nu_ks[k] + 1 - (d + 1)) / 2)
#         log_det_ks[k] = + D_sum + D * np.log(2) + np.log(det(W_ks[k]))
#
#     return log_det_ks
#
# def quad_expectation(m_ks, W_ks, beta_ks, nu_ks, Y, N, D, K):
#     """
#     Given the estimated mus and covs,
#     compute the expression 'B' as shown above (for every n and k)
#     """
#
#     quad_ks = np.zeros((N,K))
#     for k in range(K):
#         for n in range(N):
#             y_mu_diff = Y[n] - m_ks[k]
#             y_mu_diff.shape = (D, 1)
#             quad_ks[n, k] = (D / beta_ks[k]) + nu_ks[k] * (np.dot(np.dot(y_mu_diff.T, W_ks[k]), y_mu_diff)[0][0])
#
#     return quad_ks
#
# def variational_E_step(alpha_hat, m_ks, W_ks, beta_ks, nu_ks, Y, N, D, K):
#     """
#     Given the paramters of the clustres,
#     compute gamma_{n,k} for n=1...N, k=1...K
#     """
#
#     gammas = np.zeros((N,K))
#
#     log_det_ks = log_det_expectation(W_ks, nu_ks, D, K)
#     log_pi_k = log_expectations_dir(alpha_hat, K)
#
#     quad_ks = quad_expectation(m_ks, W_ks, beta_ks, nu_ks, Y, N, D, K)
#
#     log_gammas = np.zeros((N,K))
#
#     for n in range(N):
#         gamma_sum = []
#         for k in range(K):
#             log_gamma = log_pi_k[k] + (- D / 2) * np.log(2 * np.pi) + (1 / 2) * log_det_ks[k] - (1 / 2) * quad_ks[n, k]
#             gamma_sum.append(log_gamma)
#         log_gammas[n] = np.array(gamma_sum) - logsumexp(gamma_sum)
#     gammas = np.exp(log_gammas)
#
#     return gammas, log_gammas
###########
def quad(a, B):
    return np.dot(np.dot(a.T, B), a)

def log_expectation_wi_single(nu, W, D):
    ds = (nu + 1 - (np.arange(D) + 1)) / 2.0
    return D * np.log(2) + np.log(det(W)) + digamma(ds).sum()

def log_expectation_wi(nu_ks, W_ks, D, K):
    log_expectations = np.zeros(K)
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
    return log_expectations

# def log_expectation_wi(nu_ks, W_ks, D, K):
#     log_expectations = np.zeros(K)
#     for k in range(K):
#         sum_digamma = 0.0
#         for d in range(D):
#             sum_digamma += digamma((nu_ks[k] + 1 - (d + 1)) / 2)
#         log_expectations[k] = sum_digamma + D * np.log(2) + np.log(det(W_ks[k]))
#     return log_expectations

def quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    quadratic_expectations = np.zeros((K, N))
    for k in range(K):
        quadratic_expectations[k] = D / beta_ks[k] + nu_ks[k] * np.multiply(np.dot(Y - m_ks[k], W_ks[k]), Y - m_ks[k]).sum(1)
    return quadratic_expectations.T

def quadratic_expectation2(nu_ks, S_ks, m_ks, beta_ks, Y, N, D, K):
    quadratic_expectations = np.zeros((N, K))
    for k in range(K):
        for n in range(N):
            y_mu_diff = Y[n] - m_ks[k]
            y_mu_diff.shape = (D, 1)
            quadratic_expectations[n,k] = D / beta_ks[k] + nu_ks[k] * quad(y_mu_diff, S_ks[k])
    return quadratic_expectations

def log_expectations_dir(alpha_hat, K):
    log_expectations = np.zeros(K)
    sum_digamma = digamma(alpha_hat.sum())
    for k in range(K):
        log_expectations[k] = digamma(alpha_hat[k]) - sum_digamma
    return log_expectations

def vbE_step(alpha_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    ## return gammas_nk N by K
    quadratic_expectations = quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K)
    log_expectations_pi = log_expectations_dir(alpha_hat, K)
    log_expectation_lambda = log_expectation_wi(nu_ks, W_ks, D, K)
    log_rhos = log_expectations_pi - (D / 2) * np.log(2*np.pi) + (1 / 2) * log_expectation_lambda - (1 / 2) * quadratic_expectations

    log_gammas = (log_rhos.T - logsumexp(log_rhos, axis=1)).T
    return log_gammas

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

# def stats(log_gammas, Y, N, D, K):
#     gammas = np.exp(log_gammas)
#     N_ks = gammas.sum(0)
#     Y_ks = np.zeros((K, D))

def vbM_step(log_gammas, alpha_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, N, D, K):
    m_ks = np.zeros((K, D))
    W_ks = np.zeros((K, D, D))
    cov_ks = np.zeros((K, D, D))
    alpha_hat = alpha_0 + N_ks
    nu_ks = nu_0+ N_ks + 1
    beta_ks = beta_0 + N_ks

    for k in range(K):
        m_ks[k] = (beta_0 * m_0 + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = Y_ks[k] - m_0
        temp2.shape = (D, 1)
        W_ks[k] = inv(inv(W_0) + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * np.dot(temp2, temp2.T))
        # cov_ks[k] = inv(W_ks[k]) / (nu_ks[k] - D - 1)
        cov_ks[k] = inv(W_ks[k] * nu_ks[k])
    return alpha_hat, nu_ks, W_ks, m_ks, beta_ks, cov_ks

def log_C(alpha):
    return loggamma(alpha.sum()) - (loggamma(alpha)).sum()

def log_wishart_B(nu, W, D):
    term1 = (-nu / 2) * np.log(det(W))
    term2 = - (nu * D / 2) * np.log(2)
    term3 = - (D * (D - 1) / 4) * np.log(np.pi)
    ds = (nu + 1 - np.arange(D) + 1) / 2.0
    term4 = - loggamma(ds).sum()
    return term1 + term2 + term3 + term4



def entropy_wishart(nu, W, D):
    term1 =  - log_wishart_B(nu, W, D)
    term2 = nu * D / 2.0
    term3 = - ((nu - D - 1) / 2.0) * log_expectation_wi_single(nu, W, D)
    return term1 + term2 + term3


def elbo(log_gammas, alpha_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, alpha_hat, nu_ks, W_ks, m_ks, beta_ks, N, D, K):
    gammas = np.exp(log_gammas)
    log_pi_hat_ks = log_expectations_dir(alpha_hat, K)
    ## kl between pz and qz
    log_pi_hat_ks_expanded = np.tile(log_pi_hat_ks, (N, 1))
    kl_z = np.multiply(log_pi_hat_ks_expanded - log_gammas, gammas).sum()
    ## kl between p_pi and q_pi
    kl_pi = log_C(alpha_0) - log_C(alpha_hat) + np.multiply(alpha_0 - alpha_hat, log_pi_hat_ks).sum()
    ## kl between mu_ks and Lambda_ks
    log_q_mu_lambda = 0.0
    log_p_mu_lambda_term1 = 0.0
    log_p_mu_lambda_term2 = 0.0
    log_likelihood = 0.0
    for k in range(K):
        mk_diff = m_ks[k] - m_0
        mk_diff.shape = (D, 1)
        Ykmean_diff = Y_ks[k] - m_ks[k]
        Ykmean_diff.shape = (D, 1)

        log_lambda_k_hat = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
        log_q_mu_lambda += (1 / 2) * log_lambda_k_hat + (D / 2.0) * (np.log(beta_ks[k] / (2*np.pi)) - 1.0) - entropy_wishart(nu_ks[k], W_ks[k], D)
        log_p_mu_lambda_term1 += (1 / 2.0) * (D * np.log(beta_0 / (2 * np.pi)) + log_lambda_k_hat - (D * beta_0 / beta_ks[k]) - (beta_0 * nu_ks[k] * quad(mk_diff, W_ks[k])))
        log_p_mu_lambda_term2 += log_lambda_k_hat * (nu_0 - D -1 / 2.0) - (1 / 2.0) * nu_ks[k] * (np.diag(np.dot(inv(W_0), W_ks[k])).sum())
        log_likelihood += N_ks[k] * (log_lambda_k_hat - (D / beta_ks[k]) - nu_ks[k] * (np.diag(np.dot(S_ks[k], W_ks[k])).sum()) - nu_ks[k] * quad(Ykmean_diff, W_ks[k]) - D * np.log(2*np.pi))
    log_p_mu_lambda = log_p_mu_lambda_term1 + log_p_mu_lambda_term2 + K * log_wishart_B(nu_0, W_0, D)

    kl_mu_lambda = log_p_mu_lambda - log_q_mu_lambda

    ##likelihood term
    log_likelihood *= 1 / 2

    Elbo = log_likelihood + kl_z + kl_pi + kl_mu_lambda

    return Elbo[0][0]

#     S_ks = np.zeros((K, D, D))
#     for k in range(K):
#         y_k_mean = np.zeros(D)
#         for n in range(N):
#             y_k_mean += gammas[n, k] * Y[n]
#         y_k_mean /= N_ks[k]
#         Y_ks[k] = y_k_mean
#
#     for k in range(K):
#         S_k = np.zeros((D, D))
#         for n in range(N):
#             yn_diff = Y[n] - Y_ks[k]
#             yn_diff.shape = (D, 1)
#             S_k += np.dot(yn_diff, yn_diff.T) * gammas[n, k]
#         S_k /= N_ks[k]
#         S_ks[k] = S_k
#     return N_ks, Y_ks, S_ks

def vbM_step(log_gammas, alpha_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, N, D, K):
    m_ks = np.zeros((K, D))
    W_ks = np.zeros((K, D, D))
    cov_ks = np.zeros((K, D, D))
    alpha_hat = alpha_0 + N_ks
    nu_ks = nu_0+ N_ks + 1
    beta_ks = beta_0 + N_ks

    for k in range(K):
        m_ks[k] = (beta_0 * m_0 + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = Y_ks[k] - m_0
        temp2.shape = (D, 1)
        W_ks[k] = inv(inv(W_0) + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * np.dot(temp2, temp2.T))
        # cov_ks[k] = inv(W_ks[k]) / (nu_ks[k] - D - 1)
        cov_ks[k] = inv(W_ks[k] * nu_ks[k])
    return alpha_hat, nu_ks, W_ks, m_ks, beta_ks, cov_ks

def log_C(alpha):
    return loggamma(alpha.sum()) - (loggamma(alpha)).sum()

def log_wishart_B(nu, W, D):
    term1 = (-nu / 2) * np.log(det(W))
    term2 = - (nu * D / 2) * np.log(2)
    term3 = - (D * (D - 1) / 4) * np.log(np.pi)
    ds = (nu + 1 - (np.arange(D) + 1)) / 2.0
    term4 = - loggamma(ds).sum()
    return term1 + term2 + term3 + term4



def entropy_wishart(nu, W, D):
    term1 =  - log_wishart_B(nu, W, D)
    term2 = nu * D / 2.0
    term3 = - ((nu - D - 1) / 2.0) * log_expectation_wi_single(nu, W, D)
    return term1 + term2 + term3


def elbo(log_gammas, alpha_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, alpha_hat, nu_ks, W_ks, m_ks, beta_ks, N, D, K):
    gammas = np.exp(log_gammas)
    log_pi_hat_ks = log_expectations_dir(alpha_hat, K)
    ## kl between pz and qz
    log_pi_hat_ks_expanded = np.tile(log_pi_hat_ks, (N, 1))
    kl_z = np.multiply(log_pi_hat_ks_expanded - log_gammas, gammas).sum()
    kl_z *= -1
    ## kl between p_pi and q_pi
    kl_pi = log_C(alpha_0) - log_C(alpha_hat) + np.multiply(alpha_0 - alpha_hat, log_pi_hat_ks).sum()
    kl_pi *= -1
    ## kl between mu_ks and Lambda_ks
    log_q_mu_lambda = 0.0
    log_p_mu_lambda_term1 = 0.0
    log_p_mu_lambda_term2 = 0.0
    log_likelihood = 0.0
    for k in range(K):
        mk_diff = m_ks[k] - m_0
        mk_diff.shape = (D, 1)
        Ykmean_diff = Y_ks[k] - m_ks[k]
        Ykmean_diff.shape = (D, 1)

        log_lambda_k_hat = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
        log_q_mu_lambda += (1 / 2) * log_lambda_k_hat + (D / 2.0) * (np.log(beta_ks[k] / (2*np.pi)) - 1.0) - entropy_wishart(nu_ks[k], W_ks[k], D)
        log_p_mu_lambda_term1 += (1 / 2.0) * (D * np.log(beta_0 / (2 * np.pi)) + log_lambda_k_hat - (D * beta_0 / beta_ks[k]) - (beta_0 * nu_ks[k] * quad(mk_diff, W_ks[k])))
        log_p_mu_lambda_term2 += log_lambda_k_hat * ((nu_0 - D -1) / 2.0) - (1 / 2.0) * nu_ks[k] * (np.diag(np.dot(inv(W_0), W_ks[k])).sum())
        log_likelihood += N_ks[k] * (log_lambda_k_hat - (D / beta_ks[k]) - nu_ks[k] * (np.diag(np.dot(S_ks[k], W_ks[k])).sum()) - nu_ks[k] * quad(Ykmean_diff, W_ks[k]) - D * np.log(2*np.pi))
    log_p_mu_lambda = log_p_mu_lambda_term1 + log_p_mu_lambda_term2 + K * log_wishart_B(nu_0, W_0, D)

    kl_mu_lambda = log_p_mu_lambda - log_q_mu_lambda

    ##likelihood term
    log_likelihood *= 1 / 2
    print(log_likelihood , kl_z , kl_pi , kl_mu_lambda)

    Elbo = log_likelihood + kl_z + kl_pi + kl_mu_lambda

    return Elbo[0][0]
