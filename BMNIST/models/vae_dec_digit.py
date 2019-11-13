import torch
import torch.nn as nn
import probtorch

class Decoder(nn.Module):
    def __init__(self, num_pixels, num_hidden, z_what_dim):
        super(self.__class__, self).__init__()
        self.z_mean = torch.zeros(z_what_dim).cuda()
        self.z_std = torch.ones(z_what_dim).cuda()
        self.dec_image = nn.Sequential(
                           nn.Linear(z_what_dim, int(0.5*num_hidden)),
                           nn.ReLU(),
                           nn.Linear(int(0.5*num_hidden), num_hidden),
                           nn.ReLU(),
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())

    def forward(self, images, q=None, num_samples=None):
        EPS = 1e-9
        p = probtorch.Trace()
        z = p.normal(self.z_mean,
                     self.z_std,
                     value=q['z'],
                     name='z')
        images_mean = self.dec_image(z)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
               images_mean, images, name='x')
        return p
