import torch

def solve_rr(X, y, lamb):
    X2 = torch.matmul(X, X.t())
    G_inv = torch.inverse(X2 + lamb * torch.eye(X2.shape[0]))
    Xy = torch.matmul(X, y)
    return torch.matmul(G_inv, Xy)

def solve_als(R, H0, W0, reg=1e-3, iterations=10):
    H = H0.clone()
    W = W0.clone()
    for i in range(iterations):
        for t, Rt in enumerate(R):
            H[t] = solve_rr(W, Rt, reg)
        for t, Rt in enumerate(R.t()):
            W.t()[t] = solve_rr(H.t(), Rt, reg)
    return H, W
