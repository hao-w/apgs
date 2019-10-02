import torch
import time
import numpy as np
from utils import *

class Eval:
    def __init__(self, NUM_GROUPS, K, D, batch_size, CUDA, device):
        super().__init__()
        self.NUM_GROUPS = NUM_GROUPS
        self.B = batch_size
        self.K = K
        self.D = D
        self.CUDA = CUDA
        self.device = device
    """
    A class for the purpose of evaluation after training, including the following functions:
    Sample_data : extract one batch of datasets based on the data pointer
    Plot_onestep : visualize the data/sample in one time Step
    Plot_chains : align all visualizaztion of all datasets in different rows
    Test_single_dataset : apply APG algorithm at test time and return necessary metrics/variables
    """
    def Sample_data_uniform(self, data_path, data_ptr):
        """
        sample one dataset from each group
        """
        Data = []
        TJ = []
        group_ind = torch.randperm(self.NUM_GROUPS)
        for i in range(self.B):
            data_g = torch.from_numpy(np.load(data_path + 'ob_%d.npy' % group_ind[0])).float()
            tj = torch.from_numpy(np.load(data_path + 'tj_%d.npy' % group_ind[0])).float()
            Data.append(data_g[data_ptr+i])
            TJ.append(tj[data_ptr+i])
        return Data, TJ

    def Test_uniform(self, models, objective, data_path, mnist_mean, crop, data_ptr, mcmc_steps, sample_size):
        Metrics = {'samples' : [], 'data' : [], 'recon' : [], 'll' : [], 'ess' : []}
        Data, TJ = self.Sample_data_uniform(data_path, data_ptr)
        tj_std = torch.ones(self.D) * 0.1
        if self.CUDA:
            with torch.cuda.device(self.device):
                tj_std = tj_std.cuda()
                mnist_mean = mnist_mean.cuda().unsqueeze(0).unsqueeze(0).repeat(sample_size, 1, 1, 1)
        for i, frames in enumerate(Data):
            Metrics['data'].append(frames)
            if self.CUDA:
                with torch.cuda.device(self.device):
                    frames = frames.cuda().unsqueeze(0)
                    tj = TJ[i].cuda().unsqueeze(0).transpose(1,2)
            metrics = objective(models, self.K, self.D, frames, mcmc_steps, mnist_mean, crop, tj, tj_std)
            Metrics['samples'].append(metrics['samples'])
            Metrics['recon'].append(metrics['recon'])
            Metrics['ess'].append(torch.cat(metrics['ess'], 0))
            Metrics['ll'].append(torch.cat(metrics['ll'], 0))
        return Metrics
