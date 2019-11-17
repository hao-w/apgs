import torch
import time
from utils import *
from apg import *

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
    def Sample_data_uniform(self, Data, data_ptr):
        """
        sample one dataset from each group
        """
        OB = []
        num_datasets = Data[0].shape[0]
        indices = torch.randperm(num_datasets)
        for i in range(self.B):
            ob_g = Data[0][indices[i]]
            OB.append(ob_g)
        return OB

    def Test_uniform(self, models, objective, Data, data_ptr, mcmc_steps, sample_size):
        Metrics = {'samples' : [], 'data' : [], 'recon' : [], 'log_joint' : [], 'ess' : []}
        batch = self.Sample_data_uniform(Data, data_ptr)
        for data in batch:
            Metrics['data'].append(data)
            data = data.repeat(sample_size, 1, 1, 1)
            if self.CUDA:
                with torch.cuda.device(self.device):
                    data = data.cuda()
            metrics = objective(models, data, mcmc_steps, self.K)
            Metrics['samples'].append(metrics['samples'])
            Metrics['recon'].append(metrics['recon'])
            Metrics['ess'].append(torch.cat(metrics['ess'], 0))
            Metrics['log_joint'].append(torch.cat(metrics['log_joint'], 0))
        return Metrics
