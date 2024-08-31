# Unsupervised metric learning via separation framework

This research and codebase builds on _Riemannian geometry for efficient analysis of protein dynamics data_ by Diepeveen et al. (2023).

    [1] W. Diepeveen, C. Esteve-Yagüe, J. Lellmann, O. Öktem, C-B. Schönlieb.  
    Riemannian geometry for efficient analysis of protein dynamics data
    arXiv preprint arXiv:2308.07818. 2023 Aug 15.

## Setup

For toy example and fourier networks notebook, just install torch and pytorch-lightning or whatever is missing in the arbitrary order and it should work. I don't really use any fancy packages or complex dependencies so you should be able to brute force it. I added my own requirements.txt file, but I don't guarantee you can build an environment from it.

NOTE: Point cloud stuff is not guaranteed to work, these are the files from the initial version of the project.

## Framework

This repository consists of three main parts:
1. Source code for the computational part, which is located in the `src` directory.
2. The training script, `train_correction.py`, which is basically glues PyTorch Lightning runner and Hydra configuration system. 
   1. Running ```python train_correction.py``` will run the default configuration.
   2. Running ```python train_correction.py +experiment=experiment_name``` will run the experiment with the name `experiment_name`. This includes whole batches of experiments, used to reproduce experiments from the thesis.
3. Config files for all objects used in the training process, which are located in the `conf` directory. It allows quickly changing encoder parameters, loss parameters, optimizer etc. It can also be done through Hydra overriding, like ```python train_correction.py dataset=helix3d encoder=relu/fourier_s0.5_relu encoder.input_dim=3 encoder.output_dim=2```. The input dimension of the encoder should be adjusted to the dataset dimension. 