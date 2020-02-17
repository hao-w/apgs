Source code for the amortized population Gibbs (APG) samplers in the paper **Amortized Population Gibbs Samplers with Neural Sufficient Statstics** ([arXiv](https://arxiv.org/abs/1911.01382)). The implementation was built on [ProbTorch](https://github.com/probtorch/probtorch), a deep generative modeling library which extends [PyTorch](https://pytorch.org/).

### DEPENDENCIES
1. Install PyTorch 1.3.0 [[instructions](https://github.com/pytorch/pytorch)]
2. Install ProbTorch [[instructions](https://github.com/probtorch/probtorch)]

### DATA PATH VARIABLES
In the script import_envs.py, you need to set the path variables of data used in each task. In detail,
1. In GMM task, path variable is GMM_DIR
2. In DMM task, path variable is DGMM_DIR
3. In Bouncing MNIST task, path variable is BMNIST_DIR
There variables are 'global' variables used for data simulation, training and testing, so that you don not need to specify the path in every process.


### DATA SIMULATION
In each task, there are a Jupyter notebook that generates the corpus used during training and testing. In detail,
1. In GMM task, data simulator is GMM/notebooks/sim-gmm.ipynb
2. in DMM task, data simulator is DGMM/notebooks/sim-dgmm.ipynb
3. In Bouncing MNIST task, data simulator is BMNIST/notebooks/sim-bmnist.ipynb
The hyper-parameters are specified as the ones used in the experiments.
