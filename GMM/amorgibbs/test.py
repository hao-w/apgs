
def Eubo_mcmc_init_eta(enc_eta, enc_z, obs, K, mcmc_size, device, RESAMPLE=False):
    """
    EUBO for amortized gibbs with backward transition,
    individually compute importance weights
    """
    sample_size, batch_size, N, D  = obs.shape
    eubos = torch.zeros(2*mcmc_size+1).cuda().to(device)
    elbos = torch.zeros(2*mcmc_size+1).cuda().to(device)
    esss = torch.zeros(2*mcmc_size+1).cuda().to(device)
    ## sample from prior and finish one full update
    obs_mu, obs_tau  = enc_eta.sample_prior(sample_size, batch_size)
    q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
    eubo_state_forward, elbo_state_forward, ess_state_forward, weights_state = Incremental_forward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D)
    eubos[0] = eubo_state_forward
    elbos[0] = elbo_state_forward
    esss[0] = ess_state_forward
    for m in range(mcmc_size):
        state = q_z['zs'].value
        q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
        eubo_eta_backward, elbo_eta_backward, ess_eta_backward, _ = Incremental_backward_eta(q_eta, p_eta, obs, state, K, D, obs_mu=obs_mu, obs_tau=obs_tau)
        state_r = resample_state(q_z['zs'].value, weights_state, idw_flag=True)
        q_eta, p_eta, q_nu = enc_eta(obs, state_r, K, D)
        eubo_eta_forward, elbo_eta_forward, ess_eta_forward, weights_eta  = Incremental_forward_eta(q_eta, p_eta, obs, state_r, K, D)

        eubos[1+2*m] = eubo_eta_forward - eubo_eta_backward
        elbos[1+2*m] = elbo_eta_forward - elbo_eta_backward
        esss[1+2*m] = (ess_eta_forward + ess_eta_backward) / 2

        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        q_z, p_z = enc_z(obs, obs_tau, obs_mu, N, sample_size, batch_size)
        eubo_state_backward, elbo_state_backward, ess_state_backward, _ = Incremental_backward_state(q_z, p_z, obs, obs_mu, obs_tau, K, D, state=state)

        obs_mu_r, obs_tau_r = resample_eta(q_eta['means'].value, q_eta['precisions'].value, weights_eta, idw_flag=True)
        q_z, p_z = enc_z(obs, obs_tau_r, obs_mu_r, N, sample_size, batch_size)
        eubo_state_forward, elbo_state_forward, ess_state_forward, weights_state = Incremental_forward_state(q_z, p_z, obs, obs_mu_r, obs_tau_r, K, D)

        eubos[2+2*m] = eubo_state_forward - eubo_state_backward
        elbos[m+1] = elbo_state_forward - elbo_state_backward
        esss[2+2*m] = (ess_state_forward + ess_state_backward) / 2
    return eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, enc_eta.prior_nu
