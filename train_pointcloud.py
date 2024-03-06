import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.metric_learning.metric_correction import PointCloudConvEncoder, PointCloudMLPEncoder
from src.datasets.pointcloud import prepare_data, TrajGeodesicsDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--trainer_root_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='lightning_logs')
    args = parser.parse_args()
    return args


class InspectGradients(pl.Callback):
    def __init__(self, save_dir, base_name, enable_plotting=True):
        super().__init__()
        self.counter = 0
        self.save_dir = save_dir
        self.base_name = base_name
        self.enable_plotting = enable_plotting

    def on_after_backward(self, trainer, pl_module):
        self.plot_grad_flow(pl_module.named_parameters())

    def plot_grad_flow(self, named_parameters):
        '''
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        '''

        if self.counter % 10 != 0:
            self.counter += 1
            return

        print(f'Plotting gradients at step {self.counter}')
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
                print(f'[{n}] ave_grad: {p.grad.abs().mean()}, max_grad: {p.grad.abs().max()}')
        print()

        if self.enable_plotting:
            plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
            plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
            plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
            plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(left=0, right=len(ave_grads))
            plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            plt.legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

            plt.savefig(os.path.join(self.save_dir, f'{self.base_name}_{self.counter}.png'))
            plt.close()
        self.counter += 1


class LitDeepPointCloudMetric(pl.LightningModule):
    def __init__(self, manifold, encoder_cls, encoder_params,
                 correction_metric='l2', correction_coeff=1.,
                 loss_metric='l2'):
        super().__init__()
        self.save_hyperparameters()

        self.base_manifold = manifold
        self.correction_encoder = encoder_cls(**encoder_params)

        assert loss_metric in ['l2', 'log']
        self.loss_metric = loss_metric

        assert correction_metric in ['l2', 'log']
        self.correction_metric = correction_metric

        assert isinstance(correction_coeff, float)
        assert correction_coeff >= 0.
        self.correction_coeff = correction_coeff

    def basic_step(self, batch, batch_idx):
        base_dists, corrected_dists = self(batch)

        with torch.no_grad():
            lhs = torch.sum(base_dists, dim=1) ** 2
            rhs = base_dists.shape[1] * torch.sum(base_dists ** 2, dim=1)

            if self.loss_metric == 'l2':
                base_loss = nn.functional.mse_loss(lhs, rhs)
            elif self.loss_metric == 'log':
                base_loss = nn.functional.mse_loss(torch.log(lhs + 1), torch.log(rhs + 1))
            else:
                raise ValueError(f'Unknown loss metric: {self.loss_metric}')

        # dists dimensions: (B, L-1)
        lhs = torch.sum(corrected_dists, dim=1) ** 2
        rhs = corrected_dists.shape[1] * torch.sum(corrected_dists ** 2, dim=1)

        if self.loss_metric == 'l2':
            loss = nn.functional.mse_loss(lhs, rhs)
        elif self.loss_metric == 'log':
            loss = nn.functional.mse_loss(torch.log(lhs + 1), torch.log(rhs + 1))
        else:
            raise ValueError(f'Unknown loss metric: {self.loss_metric}')

        return base_loss, loss

    def forward(self, batch):
        x = batch  # x dimensions: (B, L, N, D)
        x1 = x[:, :-1, :, :]  # dimensions: (B, L-1, N, D)
        x2 = x[:, 1:, :, :]  # dimensions: (B, L-1, N, D)

        base_dists = self.base_manifold.s_distance(x1, x2)  # dimensions: (B, L-1, L-1)
        l = base_dists.shape[1]
        base_dists = base_dists[:, range(l), range(l)]
        assert base_dists.shape == (x.shape[0], x.shape[1] - 1)

        start_reps = self.correction_encoder.forward_2d_batch(x1)  # dimensions: (B, L-1, H)
        end_reps = self.correction_encoder.forward_2d_batch(x2)  # dimensions: (B, L-1, H)

        if self.correction_metric == 'l2':
            deep_dists = torch.norm(start_reps - end_reps, dim=2)
        elif self.correction_metric == 'log':
            start_norms = torch.norm(start_reps, dim=2) ** 2 + 1
            end_norms = torch.norm(end_reps, dim=2) ** 2 + 1
            deep_dists = torch.log(start_norms / end_norms)
        else:
            raise ValueError(f'Unknown correction metric: {self.correction_metric}')

        assert deep_dists.shape == (x.shape[0], x.shape[1] - 1)

        #corrections = self.correction_coeff * deep_dists
        #print(f'base_dists: {base_dists}')
        #print(f'corrections: {corrections}')

        corrected_dists = torch.sqrt(base_dists ** 2 + (self.correction_coeff * deep_dists) ** 2)
        return base_dists, corrected_dists

    def training_step(self, batch, batch_idx):
        base_loss, loss = self.basic_step(batch, batch_idx)
        self.log('train/base_loss', base_loss, on_epoch=True, on_step=False)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        base_loss, loss = self.basic_step(batch, batch_idx)
        self.log('val/base_loss', base_loss)
        self.log('val/loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        base_loss, loss = self.basic_step(batch, batch_idx)
        self.log('test/base_loss', base_loss)
        self.log('test/loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": ReduceLROnPlateau(optimizer, patience=15, factor=0.8, verbose=True),
            #     "monitor": "val/loss",
            #     'interval': 'epoch',
            #     "frequency": 1
            # },
        }


if __name__ == '__main__':
    args = parse_args()

    seed = 42
    batch_size = 512

    struct = 1
    data_folder = os.path.join('', "data", "molecular_dynamics")
    results_folder = 'results'

    ca_pos = prepare_data(struct, data_folder)

    window_size = 10
    dataset = TrajGeodesicsDataset(ca_pos, window_size=window_size)

    # train_len = 80
    # val_len = 20
    # test_len = len(dataset) - train_len - val_len
    print('full dataset length:', len(dataset))
    dataset_lengths = [71, 22, 0]
    train_dataset, val_dataset, test_dataset = random_split(dataset, dataset_lengths,
                                                            generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # encoder = PointCloudConvEncoder
    # encoder_params = dict(n_atoms=dataset.manifold.n,
    #                       input_dim=dataset.manifold.d,
    #                       output_dim=8)

    encoder_cls = PointCloudMLPEncoder
    encoder_params = dict(n_atoms=dataset.manifold.n,
                          input_dim=dataset.manifold.d,
                          hidden_dim=1024,
                          output_dim=256)

    loss_metric = 'l2'
    correction_metric = 'log'
    correction_coeff = 1.
    model = LitDeepPointCloudMetric(dataset.manifold, encoder_cls, encoder_params,
                                    correction_metric=correction_metric, correction_coeff=correction_coeff,
                                    loss_metric=loss_metric)

    name = f'adk_full_{window_size}_windowsize_mlpmodel'

    logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir, name=name)
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator=args.accelerator,
                         devices=args.devices,
                         log_every_n_steps=10,
                         default_root_dir=args.trainer_root_dir,
                         logger=logger,
                         #callbacks=[InspectGradients(save_dir=os.getcwd(), base_name='gradflow', enable_plotting=False)],
                         #limit_val_batches=0.0,
                         )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    #trainer.test(model, ckpt_path="best", dataloaders=train_loader)
    trainer.test(model, ckpt_path="best", dataloaders=val_loader)



