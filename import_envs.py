import sys
sys.path.append("../")
sys.path.append("../models/")
sys.path.append("../objectives/")
sys.path.append('/home/hao/Research/probtorch/')
import torch
import probtorch
print('probtorch:', probtorch.__version__, 
      'torch:', torch.__version__)