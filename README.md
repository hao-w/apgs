Source code for the amortized population Gibbs (APG) samplers in the paper **Amortized Population Gibbs Samplers with Neural Sufficient Statstics** ([arXiv](https://arxiv.org/abs/1911.01382)). The implementation was built on [ProbTorch](https://github.com/probtorch/probtorch), a deep generative modeling library which extends [PyTorch](https://pytorch.org/).

### DEPENDENCIES
1. Install PyTorch 1.3.0 [[instructions](https://github.com/pytorch/pytorch)]
2. Install ProbTorch [[instructions](https://github.com/probtorch/probtorch)]

### DATA SIMULATION
1. Set up the directories in the script import_envs.py which globally controls the directories of the training and testing data for all tasks.
2.  In each task, there are a Jupyter notebook that generates the corpus used during training and testing: In GMM task, data simulator is GMM/notebooks/sim-gmm.ipynb; In DMM task, data simulator is DGMM/notebooks/sim-dgmm.ipynb; In Bouncing MNIST task, data simulator is BMNIST/notebooks/sim-bmnist.ipynb. The hyper-parameters are specified as the ones used in the experiments.
