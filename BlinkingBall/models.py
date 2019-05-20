import torch
import torch.nn as nn

class Enc_H(nn.Module):
    def __init__(self, K, H=None):
            super(self.__class__, self).__init__()
            if H is None:
                self.linear = nn.Linear(K, K)
            else:
                self.linear = nn.Sequential(nn.Linear(K, H),
                                        nn.Linear(H, K))

    def forward(self, Wr_i):
        H_i = self.linear(Wr_i)
        return H_i

class Enc_W(nn.Module):
    def __init__(self, K, H=None):
            super(self.__class__, self).__init__()
            if H is None:
                self.linear = nn.Linear(K, K)
            else:
                self.linear = nn.Sequential(nn.Linear(K, H),
                                            nn.Linear(H, K))

    def forward(self, Hr_u):
        W_u = self.linear(Hr_u)
        return W_u





