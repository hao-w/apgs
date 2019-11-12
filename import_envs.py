import sys
sys.path.append("../")
sys.path.append('../modules/')
sys.path.append('/home/hao/Research/probtorch/')
DATA_DIR = '/home/hao/Research/apg_data/'
import torch
import numpy as np
import probtorch
print('probtorch:', probtorch.__version__, 
      'torch:', torch.__version__)