import autograd.numpy as np
from autograd.numpy.linalg import inv, det
from autograd import grad
from functools import reduce
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gamma, invgamma
from scipy.special import digamma, polygamma
from scipy.special import gamma as gafun


def generate_video(image_length, time_length):
    video = np.empty([image_length, time_length])
    rand_pt = npr.randint(image_length)
    rand_dir = npr.randint(2) ## 0 for downwards, 1 for upwards
    for i in range(time_length):
        one_hot = np.zeros(image_length)
        one_hot[rand_pt] = 1.0
        video[:, i] = one_hot
        if rand_pt == 0:
            rand_dir = 0
        if rand_pt == 19:
            rand_dir = 1

        if rand_dir == 0:
            rand_pt += 1
        else:
            rand_pt -= 1
    return video

def generate_dataset(generate_dataset_filename, num_sequences, image_length, time_length):
    video_lists = []
    for i in range(num_sequences):
        video = generate_video(image_length, time_length)
#         plt.imshow(video, cmap='gray', vmin=0, vmax=1)
#         plt.colorbar()
        video_lists.append(video.reshape(1, image_length * time_length))
    video_arrays = np.concatenate(video_lists, 0)
    np.save(generate_dataset_filename, video_arrays)
    return video_arrays

def load_data(dataset_filename, num_sequences, image_length, time_length):
    train_dataset = np.load(dataset_filename).reshape(num_sequences, image_length, time_length)
#    for i in range(num_sequences):
#        plt.imshow(train_dataset[i, :, :], cmap='gray', vmin=0, vmax=1)
#        plt.show()
    x, y, z = train_dataset.shape
    print('dataset size : %d, image length : %d, sequence length : %d' % (x, y, z))
    return train_dataset

def compute_yhat(video, N, T):
    Y_hat = np.zeros((N, N))
    for t in range(T):
        y = video[:, t]
        y.shape = (N, 1)
        Y_hat += np.dot(y, y.T)
    return Y_hat

def init_hyper(N, K):
    ## hyperprior for alpha, gamma
    ## b is inverse of scale!!!
    alpha_a_0 = np.ones(K)
    alpha_b_0 = np.ones(K)

    r_a_0 = np.ones(K)
    r_b_0 = np.ones(K)

    rhos = np.ones(N)

    mu_x0_0_sigma_diag = np.ones(K)

    sigma_x0_a = np.ones(K)
    sigma_x0_b = np.ones(K)
    return alpha_a_0, alpha_b_0, r_a_0, r_b_0, rhos, mu_x0_0_sigma_diag, sigma_x0_a, sigma_x0_b

def init_hsss(mu_x0, video, N, T, K):
    ## randomly initialize ss
    W_A = np.eye(K)
    # G_A = 0
    # M~ = 0
    S_A = np.eye(K)
    W_C = np.eye(K)
    # G_C = 0
    S_C = np.zeros((K, N))
#    for t in range(T):
#       yt = video[:, t]
#        yt.shape = (N, 1)
#        S_C += np.dot(mu_x0, yt.T)
    return W_A, S_A, W_C, S_C

def infer_qs(W_A, S_A, W_C, S_C, Y_hat, alpha, r, rho_a, rho_b, N, T):
    # q(B) = 0
    # q(A)
    q_sigma_A = inv(np.diag(alpha) + W_A)
    q_mu_A = np.dot(q_sigma_A, S_A) # each col is a mean vector

    # q(C|rho)
    sigma_C = inv(np.diag(r) + W_C)
    G = Y_hat - eigen(S_C, sigma_C)
    G_ss = np.diagonal(G)

    #q(rho)
    q_rho_a  = (rho_a + (T / 2) ) * np.ones(N) # each element is a shape parameter
    q_rho_b = rho_b + (G_ss / 2) # each element is a scale inv parameter

    # q_sigma_C = (1 / rho_s)  * sigma_C
    q_mu_C = np.dot(sigma_C, S_C) # each col is a mean vector
    return q_sigma_A, q_mu_A, sigma_C, q_rho_a, q_rho_b, q_mu_C

def natstats(q_rho_a, q_rho_b, S_A, S_C, q_sigma_A, sigma_C, N, K):
    E_A = np.dot(S_A.T, q_sigma_A)
    E_ATA = np.dot(E_A.T, E_A) + K * q_sigma_A
    E_rho_s = np.divide(q_rho_a, q_rho_b)
    E_log_rho_s = digamma(q_rho_a) - np.log(q_rho_b)

    E_R_inv = np.diag(E_rho_s)
    E_C = np.dot(S_C.T, sigma_C)
    E_R_inv_C = np.dot(E_R_inv, E_C)
    E_CT_R_inv_C = np.dot(E_C.T, E_R_inv_C) + N * sigma_C
    return E_A, E_ATA, E_rho_s, E_log_rho_s, E_R_inv, E_C, E_R_inv_C, E_CT_R_inv_C

def update_marginals(Mu_xt, Sigma_xt, Sigma_star, Psi, Eta, N, T, K, E_A, E_CT_R_inv_C):
    Gamma_ts = np.zeros((T+1, K, K))
    Omega_ts = np.zeros((T+1, K, 1))
    Gamma_ttp1s = np.zeros((T, K, K))
    for t in range(0, T+1):
        psi_t = Psi[t]
        eta_t = Eta[t]
        sigma_xt = Sigma_xt[t]
        mu_xt = Mu_xt[t]

        if t == 0:
            Gamma_t = sigma_xt
            Omega_t = mu_xt
            sigma_star = Sigma_star[t]
            psi_tp1 = Psi[t+1]
            term4 = inv(np.eye(K) + E_CT_R_inv_C + inv(psi_tp1) - eigen(E_A.T, sigma_star))
            Gamma_ttp1 = np.dot(np.dot(sigma_star, E_A.T), term4)
            Gamma_ttp1s[t] = Gamma_ttp1
        elif t == T:
            Gamma_t = sigma_xt
            Omega_t = mu_xt

        else:
            term1 = inv(sigma_xt)
            term2 = inv(psi_t)
            Gamma_t = inv(term1 + term2)
            term3 = np.dot(term1, mu_xt) + np.dot(term2, eta_t)
            Omega_t = np.dot(Gamma_t, term3)
            sigma_star = Sigma_star[t]
            if t == T-1:
                psi_tp1_inv = np.zeros((K, K))
            else:
                psi_tp1 = Psi[t+1]
                psi_tp1_inv = inv(psi_tp1)
            term4 = inv(np.eye(K) + E_CT_R_inv_C + psi_tp1_inv - eigen(E_A.T, sigma_star))
            Gamma_ttp1 = np.dot(np.dot(sigma_star, E_A.T), term4)
            Gamma_ttp1s[t] = Gamma_ttp1
        Gamma_ts[t] = Gamma_t
        Omega_ts[t] = Omega_t
    return Gamma_ts, Omega_ts, Gamma_ttp1s

def eigen(a, B):
    return np.dot(np.dot(a.T, B), a)

def KL_gamma(p_a, p_b, q_a, q_b):
    t1 = np.multiply(q_a, np.log(np.divide(p_b, q_b)))
    t2 = np.log(gafun(q_a)) - np.log(gafun(p_a))
    t3 = np.multiply((p_a - q_a), digamma(p_a))
    t4 = np.multiply(q_b - p_b, np.divide(p_a,p_b))
    kl_gamma = t1 + t2 + t3 + t4
    return kl_gamma.sum()

def KL_gaussian(q_mu, q_cov, p_cov):
    term1 = np.log(det(np.dot(q_cov, inv(p_cov))))
    dim_k = q_cov.shape[0]
    term2_full = np.eye(dim_k) - np.dot((q_cov + np.dot(q_mu, q_mu.T)), inv(p_cov))
    term2 = np.diag(term2_full).sum()
    return (- 1 / 2) * (term1 + term2)


def KL_A(q_mu_A, q_sigma_A, p_sigma_A, K):
    kl_A = 0.0
    for j in range(K):
        kl_A += KL_gaussian(q_mu_A[:, j], q_sigma_A, p_sigma_A)
    return kl_A


def KL_C(q_mu_C, q_rhos, sigma_C, p_sigma_C, N):
    kl_C = 0.0
    for s in range(N):
        q_sigma_C = (sigma_C / q_rhos[s])
        kl_C += KL_gaussian(q_mu_C[:, s], q_sigma_C, p_sigma_C)
    return kl_C





def update_hsss(Gamma_ts, Omega_ts, Gamma_ttp1s, video, N, T):
    W_A = 0.0
    S_A = 0.0
    S_C = 0.0
    W_C = 0.0
    for t in range(T):
        Gamma_t = Gamma_ts[t]
        Gamma_tp1 = Gamma_ts[t+1]
        Omega_t = Omega_ts[t]
        Omega_tp1 = Omega_ts[t+1]
        Gamma_ttp1 = Gamma_ttp1s[t]
        yt = video[:, t]
        yt.shape = (N, 1)
        W_A += Gamma_t + np.dot(Omega_t, Omega_t.T)
        S_A += Gamma_ttp1 + np.dot(Omega_t, Omega_tp1.T)
        W_C += Gamma_tp1 + np.dot(Omega_tp1, Omega_tp1.T)
        S_C += np.dot(Omega_tp1, yt.T)
    return W_A, S_A, W_C, S_C


def update_hyper(Gamma_ts, Omega_ts, S_A, S_C, q_sigma_A, sigma_C, E_R_inv, E_rho_s, E_log_rho_s, N, rho_a, rho_b):
    ## linear system
    alpha_new_full = q_sigma_A +  reduce(np.dot, [q_sigma_A, S_A, S_A.T, q_sigma_A])
    alpha_new = 1.0 / np.diag(alpha_new_full)
    r_new_full = sigma_C + (1 / N) * reduce(np.dot, [sigma_C, S_C, E_R_inv, S_C.T, sigma_C])
    r_new = 1.0 / np.diag(r_new_full)
    ## initial hidden state
    sigma_x0_new = Gamma_ts[0]
    mu_x0_new = Omega_ts[0]

    return alpha_new, r_new, mu_x0_new, sigma_x0_new
