Source code for the amortized population Gibbs (APG) samplers in the paper **Amortized Population Gibbs Samplers with Neural Sufficient Statstics** ([arXiv](https://arxiv.org/abs/1911.01382)). The implementation was built on [ProbTorch](https://github.com/probtorch/probtorch), a deep generative modeling library which extends [PyTorch](https://pytorch.org/).

### Prerequisite
1. Install PyTorch >= 1.6.0 [[instructions](https://github.com/pytorch/pytorch)]
2. Install ProbTorch [[instructions](https://github.com/probtorch/probtorch)]

### Training Instructions
Each set of experiments are stored in its own subdirectory. In detail, gmm, dmm, bmnist correspond the GMM clustering task, Deep Generative Mixture Model clustering task, and the Bouncing MNIST trakcing task, respectively.
Each subdirectory includes the implementation of APG sampler, baselines, and evaluation functions that are put in jupyter notebooks for the convenience of interactivity.

#### GMM

#### DMM

#### BMNIST