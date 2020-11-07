import torch
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
import torch.nn.functional as F


class Resampler():
    def __init__(self, strategy, sample_size, CUDA, device):
        super(Resampler, self).__init__()
        self.strategy = strategy
        assert self.strategy == 'systematic' or 'multinomial', "ERROR! specify resampling strategy as either systematic or multinomial."
        if self.strategy == 'systematic':
            if CUDA:
                self.uniformer = Uniform(low=torch.Tensor([0.0]).cuda().to(device), high=torch.Tensor([1.0]).cuda().to(device))
                self.spacing = torch.arange(sample_size).float().cuda().to(device)
            else:
                self.uniformer = Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))
                self.spacing = torch.arange(sample_size).float()
        self.S = sample_size
        self.CUDA = CUDA

    def sample_ancestral_index(self, log_weights):
        """
        sample ancestral indices
        """
        sample_dim, batch_dim = log_weights.shape
        if self.strategy == 'systematic':
            positions = (self.uniformer.sample((batch_dim,)) + self.spacing) / self.S
            # weights = log_weights.exp()
            normalized_weights = F.softmax(log_weights, 0)
            cumsums = torch.cumsum(normalized_weights.transpose(0, 1), dim=1)
            (normalizers, _) = torch.max(input=cumsums, dim=1, keepdim=True)
            normalized_cumsums = cumsums / normalizers ## B * S

            ancestral_index = torch.searchsorted(normalized_cumsums, positions)
            assert ancestral_index.shape == (batch_dim, sample_dim), "ERROR! systematic resampling resulted unexpected index shape."
            ancestral_index = ancestral_index.transpose(0, 1)

        elif self.strategy == 'multinomial':
            normalized_weights = F.softmax(log_weights, 0)
            ancestral_index = Categorical(normalized_weights.transpose(0, 1)).sample((sample_dim, ))
        else:
            print("ERROR! unexpected resampling strategy.")
        return ancestral_index

    def resample_4dims(self, var, ancestral_index):
        sample_dim, batch_dim, dim3, dim4 = var.shape
        gather_index = ancestral_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4)
        return torch.gather(var, 0, gather_index)

    def resample_5dims(self, var, ancestral_index):
        sample_dim, batch_dim, dim3, dim4, dim5 = var.shape
        gather_index = ancestral_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4, dim5)
        return torch.gather(var, 0, gather_index)
