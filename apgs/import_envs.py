import sys
sys.path.append("../")
sys.path.append('../../')
sys.path.append('/home/hao/Research/probtorch/')
DATA_DIR = '/home/hao/Research/apg_data/'
import torch
import numpy as np
import os
import probtorch
print('probtorch:', probtorch.__version__,
      'torch:', torch.__version__)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
