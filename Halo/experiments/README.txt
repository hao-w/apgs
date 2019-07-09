phase1 : verify the autoencoder can be learned without doing any inference (check)
phase2 : By exact LL function and Gibbs for cluster assignment, verify the angle variable can be learned properly (check)
phase3 : verify the LL function can be learned together with angle (check; but need to set the hyper-parameter noise_sigma larger than the exact sigma, otherwise local optima is always learned)
phase4 : verify the ag works, where globals = {mu}, locals = {angle}, gibbs is used for z
