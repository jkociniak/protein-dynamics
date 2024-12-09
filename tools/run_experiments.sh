#!/bin/sh

python train_correction.py -m +experiment=helix_3d_longer training_params.name=helix_osc_3d_longer_ortho200_nh10 loss.weights.orthogonal=200 loss.weights.neg_hess_norm=10 dataset='glob(*)' encoder='glob(*)';
python train_correction.py -m +experiment=helix_3d_longer training_params.name=helix_osc_3d_longer_ortho200_nh100 loss.weights.orthogonal=200 loss.weights.neg_hess_norm=100 dataset='glob(*)' encoder='glob(*)';


