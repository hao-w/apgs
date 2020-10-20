import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import probtorch

class Enc_digit(nn.Module):
    def __init__(self, num_pixels, num_hidden, z_what_dim):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                        nn.Linear(num_pixels, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, int(0.5*num_hidden)),
                        nn.ReLU())
        self.q_mean = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))
        self.q_log_std = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))

        # self.A = torch.randn(K*z_what_dim, K*z_what_dim).tril()
        # for k in range(K*z_what_dim):
        #     self.A[k,k] = 1.0
        #
        # if CUDA:
        #     self.A = self.A.cuda()
        # self.A = nn.Parameter(self.A)
        #

    def forward(self, cropped):
        q = probtorch.Trace()
        hidden = self.enc_hidden(cropped)
        q_mu = self.q_mean(hidden).mean(2) ## because T is on the 3rd dim in cropped
        q_std = self.q_log_std(hidden).exp().mean(2)
        z_what = Normal(q_mu, q_std).rsample() ## S * B * K * z_what_dim
        # S, B, K, D = z_what.shape
        q.normal(loc=q_mu,
                 scale=q_std,
                 value=z_what,
                 name='z_what')
        # q['z_what'].view(S, B, )
        #
        # rotated_z_what =  torch.bmm(self.A.unsqueeze(0).repeat(S*B, 1, 1), z_what.view(S, B, K*D).view(S*B, K*D).unsqueeze(-1))
        # rotated_z_what = rotated_z_what.squeeze(-1).view(S, B, K*D).view(S, B, K, D)
        return q
