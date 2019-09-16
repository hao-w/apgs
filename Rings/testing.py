import torch
import time
from utils import *
from apg import *

class Eval:
    def __init__(self, models, K, D, sample_size, batch_size, mcmc_steps, CUDA, device):
        super().__init__()
        self.models = models
        self.S = sample_size
        self.B = batch_size
        self.K = K
        self.D = D
        self.mcmc_steps = mcmc_steps
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
        for i in range(self.B):
            num_datasets = Data[i].shape[0]
            indices = torch.arange(num_datasets)
            ob_g = Data[i*2+1][indices[data_ptr]]
            OB.append(ob_g)
        return OB

    def Test_uniform(self, objective, Data, data_ptr):
        Metrics = {'samples' : [], 'data' : []}
        batch = self.Sample_data_uniform(Data, data_ptr)
        for data in batch:
            Metrics['data'].append(data)
            data = data.repeat(self.S, 1, 1, 1)
            EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
            if self.CUDA:
                with torch.cuda.device(self.device):
                    data = data.cuda()
                    EPS = EPS.cuda() ## EPS for KL between categorial distributions
            metrics = objective(self.models, data, self.mcmc_steps, self.K)
            Metrics['samples'].append(metrics['samples'])
        return Metrics
