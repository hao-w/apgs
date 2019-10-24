import torch
import time
import numpy as np
from utils import *

class Eval:
    def __init__(self, APG, APG_STEPS, S, DATA_PATH, DEVICE):
        self.APG = APG
        self.APG_STEPS = APG_STEPS
        self.S = S
        self.DATA_PATH = DATA_PATH
        self.DEVICE = DEVICE
        super().__init__()
    """
    A class for the purpose of evaluation after training, including the following functions:
    """
    # def Test_single_video(self, APG, data_ptr):

    # def Test_one_batch():

    def Test_single_video(self, group_ptr, data_ptr):
        """
        testing function
        """
        data = torch.from_numpy(np.load(self.DATA_PATH + 'ob_%d.npy' % group_ptr)).float()
        frames = data[data_ptr]
        frames_input = frames.unsqueeze(0).unsqueeze(0).repeat(self.S, 1, 1, 1, 1).cuda().to(self.DEVICE)
        metrics = self.APG.Sweeps(self.APG_STEPS, self.S, 1, frames_input)
        metrics['data'] = frames
        return metrics
