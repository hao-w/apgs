import torch

def shuffler(data):
    DIM1, DIM2, DIM3 = data.shape
    indices = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)])
    indices_expand = indices.unsqueeze(-1).repeat(1, 1, DIM3)
    return torch.gather(data, 1, indices_expand)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1e-3)

# def Log_likelihood(ob, state, ob_tau, ob_mu, cluster_flag=False):
#     """
#     cluster_flag = False : return S * B * N
#     cluster_flag = True, return S * B * K
#     """
#     ob_sigma = 1. / ob_tau.sqrt()
#     labels = state.argmax(-1)
#     labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, ob.shape[-1])
#     ob_mu_expand = torch.gather(ob_mu, 2, labels_flat)
#     ob_sigma_expand = torch.gather(ob_sigma, 2, labels_flat)
#     log_ob = Normal(ob_mu_expand, ob_sigma_expand).log_prob(ob).sum(-1) # S * B * N
#     if cluster_flag:
#         log_ob = torch.cat([((labels==k).float() * log_ob).sum(-1).unsqueeze(-1) for k in range(state.shape[-1])], -1) # S * B * K
#     return log_ob
