Source code for the amortized population Gibbs (APG) samplers in the paper **Amortized Population Gibbs Samplers with Neural Sufficient Statstics**, which was accepted by ICML2020 ([paper](https://proceedings.icml.cc/paper/2020/hash/6d378765f17a856b7ba8bf1541cafb69-Abstract.html) [video](https://icml.cc/virtual/2020/poster/6715)). The implementation was built on [ProbTorch](https://github.com/probtorch/probtorch), a deep generative modeling library which extends [PyTorch](https://pytorch.org/).

### Prerequisite
1. Install PyTorch >= 1.6.0 [[instructions](https://github.com/pytorch/pytorch)]
2. Install ProbTorch [[instructions](https://github.com/probtorch/probtorch)]

### Training Instructions
Each set of experiments are stored in its own subdirectory. In detail, gmm, dmm, bmnist correspond the GMM clustering task, Deep Generative Mixture Model clustering task, and the Bouncing MNIST trakcing task, respectively.
Each subdirectory includes the implementation of APG sampler, baselines, and evaluation functions that are put in jupyter notebooks for the convenience of interactivity.

To train the APG sampler, go to each subdirectory and run 
```python
python apg_training.py
```

If you have any questions about the code or the paper, please feel free to email me by wu.hao10@northeastern.edu.
