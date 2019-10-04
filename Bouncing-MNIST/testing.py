import torch
import time
import numpy as np
from utils import *

class Eval:
    def __init__(self, DATA_PATH, NUM_GROUPS, K, D):
        super().__init__()
        self.DATA_PATH = DATA_PATH
        self.NUM_GROUPS = NUM_GROUPS
        self.K = K
        self.D = D
    """
    A class for the purpose of evaluation after training, including the following functions:
    Sample_data : extract one batch of datasets based on the data pointer
    Plot_onestep : visualize the data/sample in one time Step
    Plot_chains : align all visualizaztion of all datasets in different rows
    Test_single_dataset : apply APG algorithm at test time and return necessary metrics/variables
    """

    def Test_uniform(self, APG, optimizer, data_ptr, APG_STEPS, S, B, DEVICE, PATH):
        """
        training function
        """
        Metrics = dict()

        group_ind = torch.randperm(self.NUM_GROUPS)
        data_g = torch.from_numpy(np.load(self.DATA_PATH + 'ob_%d.npy' % group_ind[0])).float()
        indices = torch.randperm(data_g.shape[0])
        b_ind = indices[data_ptr*B : (data_ptr+1)*B]
        frames = data_g[b_ind]
        Metrics['data'] = frames
        frames = frames.repeat(S, 1, 1, 1, 1).cuda().to(DEVICE)
        metrics = APG.Sweeps(APG_STEPS, S, B, frames)
        optimizer.zero_grad()
        for key in metrics.keys():
            Metrics[key] = metrics[key]
        return Metrics
