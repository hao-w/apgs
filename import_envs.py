import sys
sys.path.append("../")
sys.path.append('../../')
sys.path.append('../modules/')
sys.path.append('/home/hao/Research/probtorch/')
GMM_DIR = '/home/hao/Research/apg_data/'
DGMM_DIR = '/home/hao/Research/apg_data/'
BMNIST_DIR = '/data/hao/apg_data/'
import torch
import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import probtorch
print('probtorch:', probtorch.__version__,
      'torch:', torch.__version__)
