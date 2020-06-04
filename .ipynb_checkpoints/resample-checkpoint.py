import torch
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from torchsearchsorted import searchsorted
import torch.nn.functional as F

"""
=========
TODO : implement more sampling methods like systematic resampling based on the paper:
GPU acceleration of the particle filter: the Metropolis resampler https://arxiv.org/abs/1202.6163
01/19/2020: merge multinomial resampler with different number of dimensions of varaibles,
            and implement systematic resampler.
=========
"""
# def resample(var, weights, dim_expand=True):
#     """
#     multinomial resampling
#     """
#     dim1, dim2, dim3, dim4 = var.shape
#     if dim_expand:
#         if dim2 == 1:
#             ancesters = Categorical(weights.permute(1, 2, 0).squeeze(0)).sample((dim1, )).unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, dim4)
#         else:
#             ancesters = Categorical(weights.permute(1, 2, 0)).sample((dim1, )).unsqueeze(-1).repeat(1, 1, 1, dim4)
#     else:
#         ancesters = Categorical(weights.transpose(0, 1)).sample((dim1, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4) ## S * B * N * K
#     return torch.gather(var, 0, ancesters)


class Resampler():
    def __init__(self, strategy, sample_size, CUDA, DEVICE):

        super(Resampler, self).__init__()


        self.strategy = strategy
        assert self.strategy == 'systematic' or 'multinomial', "ERROR! specify resampling strategy as either systematic or multinomial."
        if self.strategy == 'systematic':
            if CUDA:
                self.uniformer = Uniform(low=torch.Tensor([0.0]).cuda().to(DEVICE), high=torch.Tensor([1.0]).cuda().to(DEVICE))
                self.spacing = torch.arange(sample_size).float().cuda().to(DEVICE)
            else:
                self.uniformer = Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))
                self.spacing = torch.arange(sample_size).float()

        self.S = sample_size
        self.CUDA = CUDA
        self.DEVICE = DEVICE


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

            ancestral_index = searchsorted(a=normalized_cumsums, v=positions, side='left')
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
        # print(var.shape)
        gather_index = ancestral_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4)
        return torch.gather(var, 0, gather_index)

    def resample_5dims(self, var, ancestral_index):
        sample_dim, batch_dim, dim3, dim4, dim5 = var.shape
        gather_index = ancestral_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4, dim5)
        return torch.gather(var, 0, gather_index)

#
# def resample_all(vars, weights, num_dims):
#     """
#     resampling
#     vars: list of varaiables that need to resample , each is a 4-dimensional variable: S * B * dim3 * dim4
#     weights are : S * B
#     use torch.gather to expand the weights
#     """
#     vars_resampled = []
#     sample_dim, batch_dim = weights.shape
#     ancesters = Categorical(weights.transpose(0, 1)).sample((sample_dim, )).unsqueeze(-1).unsqueeze(-1)
#     for var in vars:
#         _, _, dim3, dim4 = var.shape
#         ancesters_expand = ancesters.repeat(1, 1, dim3, dim4) ## S * B * N * K
#         var_resampled = torch.gather(var, 0, ancesters_expand)
#         vars_resampled.append(var_resampled)
#     return vars_resampled
#
#
#
#
# def resample_all(vars, weights):
#     """
#     resampling
#     vars: list of varaiables that need to resample , each is a 4-dimensional variable: S * B * dim3 * dim4
#     weights are : S * B
#     use torch.gather to expand the weights
#     """
#     vars_resampled = []
#     sample_dim, batch_dim = weights.shape
#     ancesters = Categorical(weights.transpose(0, 1)).sample((sample_dim, )).unsqueeze(-1).unsqueeze(-1)
#     for var in vars:
#         _, _, dim3, dim4 = var.shape
#         ancesters_expand = ancesters.repeat(1, 1, dim3, dim4) ## S * B * N * K
#         var_resampled = torch.gather(var, 0, ancesters_expand)
#         vars_resampled.append(var_resampled)
#     return vars_resampled
#
#
# def resample_5(var, weights, dim_expand=True):
#     """
#     multinomial resampling
#     """
#     dim1, dim2, dim3, dim4, dim5 = var.shape
#     if dim_expand:
#         if dim2 == 1:
#             ancesters = Categorical(weights.permute(1, 2, 0).squeeze(0)).sample((dim1, )).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, dim4, dim5)
#         else:
#             ancesters = Categorical(weights.permute(1, 2, 0)).sample((dim1, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, dim4, dim5)
#     else:
#         ancesters = Categorical(weights.transpose(0, 1)).sample((dim1, )).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4, dim5) ## S * B * N * K
#     return torch.gather(var, 0, ancesters)
#
#
# def systematic_resample(var, weights, DEVICE):
#     """
#     systematic resampling implemented by Tuan-Anh Le
#     https://github.com/tuananhle7/aesmc/blob/09a824254a5eeae72669c6ac4effae810851cde2/aesmc/inference.py#L234-L269
#     """
#     dim1, dim2, dim3, dim4 = var.shape
#     indices = sample_ancestral_index(weights.transpose(0,1), DEVICE) # B * S or B * S * N
#     ancesters = indices.transpose(0,1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, dim3, dim4)
#     return torch.gather(var, 0, ancesters)
#
# # def sample_ancestral_index(weights, DEVICE):
# #     B, S = weights.size()
# #     indices = np.zeros([B, S])
# #     uniforms = np.random.uniform(size=[B, 1])
# #     pos = (uniforms + np.arange(0, S)) / S
# #     normalized_weights = weights.cpu().data.numpy()
# #     # np.ndarray [batch_size, num_particles]
# #     cumulative_weights = np.cumsum(normalized_weights, axis=1)
# #     # hack to prevent numerical issues
# #     cumulative_weights = cumulative_weights / np.max(
# #         cumulative_weights, axis=1, keepdims=True)
# #     for batch in range(B):
# #         indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])
# #     return torch.from_numpy(indices).long().cuda().to(DEVICE)
#
# from torchsearchsorted import searchsorted
# def init_uniformer(CUDA, DEVICE):
#     if CUDA:
#         uniformer = Uniform(low=torch.Tensor([0.0]).cuda().to(DEVICE), high=torch.Tensor([1.0]).cuda().to(DEVICE))
#     else:
#         uniformer = Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))
#     return uniformer
#
# """
# A open source implementation of searchsorted function in pytorch (>=1.3.0)
# https://github.com/aliutkus/torchsearchsorted
# """
# def sample_ancestral_index(weights, uniformer):
#     B, S = weights.shape
#     # indices = np.zeros([B, S])
#     uniforms = np.random.uniform(size=[B, 1])
#     pos = (uniforms + np.arange(0, S)) / S
#     normalized_weights = weights.cpu().data.numpy()
#     # np.ndarray [batch_size, num_particles]
#     cumulative_weights = np.cumsum(normalized_weights, axis=1)
#     # hack to prevent numerical issues
#     cumulative_weights = cumulative_weights / np.max(
#         cumulative_weights, axis=1, keepdims=True)
#     for batch in range(B):
#         indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])
#     return torch.from_numpy(indices).long().cuda().to(DEVICE)
