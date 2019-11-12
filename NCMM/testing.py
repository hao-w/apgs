import torch
import time
from utils import *
from apg import *
import numpy as np

class Eval:
    def __init__(self, K, D, batch_size, CUDA, device):
        super().__init__()
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
    def Sample_data_uniform(self, data_ptr):
        """
        sample one dataset from each group
        """
        data_dir = "/home/hao/Research/apg_data/ncmm/rings_10size/"

        OB = []
        Ns = [120, 160, 200, 400, 600]
        for i in range(self.B):
            data_g = torch.from_numpy(np.load(data_dir + 'ob_%d.npy' % Ns[i])).float()
            num_datasets = data_g.shape[0]
            indices = torch.arange(num_datasets)
            ob_g = data_g[indices[data_ptr]]
            OB.append(ob_g)
        return OB

    def Test_uniform(self, models, objective, data_ptr, mcmc_steps, sample_size):
        Metrics = {'samples' : [], 'data' : [], 'recon' : [], 'log_joint' : [], 'ess_z' : [], 'ess_mu' : []}
        batch = self.Sample_data_uniform(data_ptr)
        for data in batch:
            Metrics['data'].append(data)
            data = data.repeat(sample_size, 1, 1, 1)
            if self.CUDA:
                with torch.cuda.device(self.device):
                    data = data.cuda()
            metrics = objective(models, data, mcmc_steps, self.K)
            Metrics['samples'].append(metrics['samples'])
            Metrics['recon'].append(metrics['recon'])
            Metrics['ess_z'].append(metrics['ess_z'])
            Metrics['ess_mu'].append(metrics['ess_mu'])
            Metrics['log_joint'].append(metrics['log_joint'])
        return Metrics

    def Test_budget(self, models, objective, data_ptr, mcmc_steps, sample_size):
        Metrics = { 'log_joint' : [],  'ess_mu' : [], 'ess_z' : []}
        batch = self.Sample_data_uniform(data_ptr)
        for data in batch:
            data = data.repeat(sample_size, 1, 1, 1)
            if self.CUDA:
                with torch.cuda.device(self.device):
                    data = data.cuda()
            metrics = objective(models, data, mcmc_steps, self.K)
            Metrics['ess_z'].append(metrics['ess_z'][0, -1].unsqueeze(0))
            Metrics['log_joint'].append(metrics['log_joint'][0, -1].unsqueeze(0))
        Metrics['log_joint'] = torch.cat(Metrics['log_joint'], 0)
        Metrics['ess_z'] = torch.cat(Metrics['ess_z'], 0)
        return Metrics
