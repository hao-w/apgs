import torch
import time
from utils import *
from normal_gamma import *
from kls import *

class Eval:
    def __init__(self, models, K, D, batch_size, CUDA, device):
        super().__init__()
        self.models = models
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
        for i in range(self.B):
            num_datasets = Data[i].shape[0]
            indices = torch.arange(num_datasets)
            ob_g = Data[i][indices[data_ptr]]
            OB.append(ob_g)
        return OB

    def Test_uniform(self, objective, Data, data_ptr, mcmc_steps, sample_size):
        Metrics = {'samples' : [], 'data' : [], 'log_joint' : [], 'elbos' : [], 'ess_z' : [], 'ess_eta': []}
        batch = self.Sample_data_uniform(Data, data_ptr)
        log_S = torch.log(torch.Tensor([sample_size]))
        for data in batch:
            Metrics['data'].append(data)
            data = data.repeat(sample_size, 1, 1, 1)
            if self.CUDA:
                with torch.cuda.device(self.device):
                    data = data.cuda()
            metrics = objective(self.models, data, mcmc_steps, log_S)
            Metrics['samples'].append(metrics['samples'])
            # Metrics['elbos'].append(torch.cat(metrics['elbos'], 0))
            Metrics['ess'].append(torch.cat(metrics['ess'], 0))
            Metrics['ll'].append(torch.cat(metrics['ll'], 0))
        return Metrics

    def Test_budget(self, objective, Data, data_ptr, mcmc_steps, sample_size):
        Metrics = {'samples' : [], 'data' : [], 'log_joint' : [], 'ess' : []}
        batch = self.Sample_data_uniform(Data, data_ptr)
        for data in batch:
            Metrics['data'].append(data)
            data = data.repeat(sample_size, 1, 1, 1)
            if self.CUDA:
                with torch.cuda.device(self.device):
                    data = data.cuda()
            metrics = objective(self.models, data, mcmc_steps)
            # kl_step = kl_train(models, ob, reused, EPS)
            # Metrics['ess_eta'].append(metrics['ess_eta'][0, -1].unsqueeze(0))
            Metrics['ess'].append(metrics['ess'][0, -1].unsqueeze(0))

            Metrics['log_joint'].append(metrics['log_joint'][0, -1].unsqueeze(0))
        # kl_step = kl_train(models, ob, reused, EPS)

        return Metrics
    #
    def Test_ALL(self, objective, Data, mcmc_steps, Test_Params):
        Metrics = {'kl_ex' : [], 'kl_in' : [], 'll' : [], 'ess' : []}
        (S, B, DEVICE) = Test_Params
        EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
        EPS = EPS.cuda().to(DEVICE) ## EPS for KL between categorial distributions
        for g, data_g in enumerate(Data):
            print("group=%d" % (g+1))
            NUM_DATASETS = data_g.shape[0]
            NUM_BATCHES = int((NUM_DATASETS / B))
            for step in range(NUM_BATCHES):
                ob = data_g[step*B : (step+1)*B]
                ob = shuffler(ob).repeat(S, 1, 1, 1)
                ob = ob.cuda().to(DEVICE)
                metrics = objective(self.models, ob, mcmc_steps, EPS)
                Metrics['ess'].append(metrics['ess'])
                Metrics['ll'].append(metrics['ll'])
                Metrics['kl_ex'].append(metrics['kl_ex'])
                Metrics['kl_in'].append(metrics['kl_in'])

        return Metrics
