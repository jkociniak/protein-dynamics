import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from src.metric_learning.euclidean import MLP, DoubleMLP, DoubleMLP2
from src.metric_learning.regularizers import CurveNegativeRegularizer, EnvelopeRegularizer, FakeRegularizer

from src.manifolds.euclidean import Euclidean, CorrectedEuclideanManifold

from src.datasets.base import ContiguousSlicesDataset, DijkstraGeodesicsDataset, FullTrajectoryDataset
from src.datasets.euclidean import (SineDataset, CircleDataset, ThirdDegreePolynomialDataset, SpiralDataset,
                                    Helix3DDataset, Sphere3DDataset,
                                    Sphere3DGeodesicsDataset)

from src.utils.callbacks import PlotPrelogs, PlotGeodesics, MonitorCorrectionCoeff, PlotGeodesicsDataset, InspectGradients


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset setup
    parser.add_argument('--dataset_seed', type=int, default=42)

    # model setup
    parser.add_argument('--encoder_hidden_dim', type=int, default=16)
    parser.add_argument('--encoder_output_dim', type=int, default=2)
    parser.add_argument('--initial_beta', type=float, default=None)
    parser.add_argument('--learnable_beta', action='store_true', default=False)

    # training setup
    parser.add_argument('--training_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--trainer_root_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='lightning_logs')

    args = parser.parse_args()
    return args


class LitManifoldMetricCorrector(pl.LightningModule):
    def __init__(self, encoder_cls, encoder_params,
                       corrected_manifold_cls, corrected_manifold_params, base_manifold_params,
                       regularizer_cls, regularizer_params):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.correction_encoder = encoder_cls(**encoder_params)
        self.corrected_manifold = corrected_manifold_cls(base_manifold_params, self.correction_encoder,
                                                         **corrected_manifold_params)

        self.regularizer = regularizer_cls(self.correction_encoder, **regularizer_params)

    def on_fit_start(self):
        layout = dict()

        layout['train'] = {}
        layout['train']['g_losses'] = ['Multiline', ['train/g_losses/base', 'train/g_losses/corrected']]
        layout['train']['hessian_norms'] = ['Multiline', ['train/hessian_norms/pos', 'train/hessian_norms/neg']]
        layout['train']['hessian_losses'] = ['Multiline', ['train/hessian_losses/pos', 'train/hessian_losses/neg']]
        layout['train']['losses'] = ['Multiline', ['train/g_losses/corrected', 'train/hessian_losses/pos',
                                                   'train/hessian_losses/neg', 'train/loss']]

        layout['val'] = {}
        layout['val']['g_losses'] = ['Multiline', ['val/g_losses/base', 'val/g_losses/corrected']]
        # layout['val']['hessian_norms'] = ['Multiline', ['val/hessian_norms/pos', 'val/hessian_norms/neg']]
        # layout['val']['hessian_losses'] = ['Multiline', ['val/hessian_losses/pos', 'val/hessian_losses/neg']]
        # layout['val']['losses'] = ['Multiline', ['val/g_losses/corrected', 'val/hessian_losses/pos',
        #                                            'val/hessian_losses/neg', 'val/loss']]

        layout['test'] = {}
        layout['test']['g_losses'] = ['Multiline', ['test/g_losses/base', 'test/g_losses/corrected']]

        layout['train_vs_val'] = {}
        layout['train_vs_val']['g_losses'] = ['Multiline', ['train/g_losses/corrected', 'val/g_losses/corrected']]

        self.logger.experiment.add_custom_scalars(layout)

    def on_train_epoch_end(self):
        self.logger.experiment.add_scalar('train/epoch', self.current_epoch, self.global_step)

    def forward(self, batch):
        x = batch  # x dimensions: (B, L, D)

        # 1. split the trajectory into start and end points of each segment
        x1 = x[:, :-1, :]  # dimensions: (B, L-1, D)
        x2 = x[:, 1:, :]  # dimensions: (B, L-1, D)

        # 2. compute base distances for each segment
        base_dists = self.corrected_manifold.base_manifold.distance(x1, x2)  # dimensions: (B, L-1)
        assert base_dists.shape == (x.shape[0], x.shape[1] - 1)

        # 3. compute correction distances for each segment
        corrected_dists = self.corrected_manifold.distance(x1, x2)  # dimensions: (B, L-1)
        assert corrected_dists.shape == (x.shape[0], x.shape[1] - 1)

        return base_dists, corrected_dists

    def compute_geodesic_loss(self, batch, batch_idx):
        base_dists, corrected_dists = self(batch)  # dists dimensions: (B, L-1)

        with torch.no_grad():
            lhs = torch.sum(base_dists, dim=1) ** 2
            rhs = base_dists.shape[1] * torch.sum(base_dists ** 2, dim=1)
            base_loss = nn.functional.mse_loss(lhs, rhs)

        lhs = torch.sum(corrected_dists, dim=1) ** 2
        rhs = corrected_dists.shape[1] * torch.sum(corrected_dists ** 2, dim=1)
        loss = nn.functional.mse_loss(lhs, rhs)

        return base_loss, loss

    def training_step(self, batch, batch_idx):
        encoder_opt, cc_opt = self.optimizers()
        encoder_opt.zero_grad()
        cc_opt.zero_grad()

        # 0. (DEBUG) compute batch encodings
        with torch.no_grad():
            x_enc = self.correction_encoder.forward_2d_batch(batch)  # dimensions: (B, L, enc_dim)
            mean_enc_norm = torch.linalg.norm(x_enc, dim=2).mean()
        self.logger.experiment.add_scalar('train/mean_enc_norm', mean_enc_norm, self.global_step)

        # 1. compute the main loss, based on the geodesic equation
        base_g_loss, g_loss = self.compute_geodesic_loss(batch, batch_idx)
        self.logger.experiment.add_scalar('train/g_losses/base', base_g_loss, self.global_step)
        self.logger.experiment.add_scalar('train/g_losses/corrected', g_loss, self.global_step)

        # 2. compute the hessian norms for both types of samples
        out = self.regularizer(batch)
        pos_hess_norm = out['pos_hess_norm']
        neg_hess_norm = out['neg_hess_norm']
        pos_mean_hess_loss = out['pos_mean_hess_loss']
        neg_mean_hess_loss = out['neg_mean_hess_loss']
        mean_smoothness_error = out.get('mean_smoothness_error', 0.)
        self.logger.experiment.add_scalar('train/hessian_norms/pos', pos_hess_norm, self.global_step)
        self.logger.experiment.add_scalar('train/hessian_norms/neg', neg_hess_norm, self.global_step)
        self.logger.experiment.add_scalar('train/hessian_losses/pos', pos_mean_hess_loss, self.global_step)
        self.logger.experiment.add_scalar('train/hessian_losses/neg', neg_mean_hess_loss, self.global_step)
        self.logger.experiment.add_scalar('train/mean_smoothness_error', mean_smoothness_error, self.global_step)

        # 3. compute the final loss
        loss = g_loss + pos_mean_hess_loss + neg_mean_hess_loss + mean_smoothness_error
        self.logger.experiment.add_scalar('train/loss', loss, self.global_step)

        self.manual_backward(loss)

        encoder_opt.step()
        cc_opt.step()

        sch1, _ = self.lr_schedulers()
        sch1.step()

        return loss

    def validation_step(self, batch, batch_idx):
        base_g_loss, g_loss = self.compute_geodesic_loss(batch, batch_idx)
        self.logger.experiment.add_scalar('val/g_losses/base', base_g_loss, self.global_step)
        self.logger.experiment.add_scalar('val/g_losses/corrected', g_loss, self.global_step)
        return g_loss

    def test_step(self, batch, batch_idx):
        base_g_loss, g_loss = self.compute_geodesic_loss(batch, batch_idx)
        self.logger.experiment.add_scalar('test/g_losses/base', base_g_loss, self.global_step)
        self.logger.experiment.add_scalar('test/g_losses/corrected', g_loss, self.global_step)
        return g_loss

    def configure_optimizers(self):
        # if type(self.correction_encoder) == DoubleMLP:
        #     base_mlp_optim = torch.optim.Adam(self.correction_encoder.base_mlp.parameters(), lr=1e-3)
        #     base_mlp_scheduler = {
        #         "scheduler": torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, gamma=2.,
        #                                                           milestones=[50, 100, 150, 200, 250, 300]),
        #         "interval": "epoch",
        #         "frequency": 1,
        #         "monitor": "val/loss",
        #         "strict": True,
        #         "name": 'encoder_lr',
        #     }
        encoder_optimizer = torch.optim.Adam(self.correction_encoder.parameters(), lr=1e-3, weight_decay=1e-3)
        encoder_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiplicativeLR(encoder_optimizer, lr_lambda=lambda epoch: 1.), #torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, gamma=2., milestones=[50]),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/loss",
            "strict": True,
            "name": 'encoder_lr',
        }

        cc_optimizer = torch.optim.Adam([self.corrected_manifold.beta], lr=1.)
        cc_scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiplicativeLR(cc_optimizer, lr_lambda=lambda epoch: 1.),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/loss",
            "strict": True,
            "name": 'cc_lr',
        }

        return [encoder_optimizer, cc_optimizer], [encoder_scheduler, cc_scheduler]


if __name__ == '__main__':
    # parse args and ensure reproducibility
    args = parse_args()

    # setup dataset and dataloaders
    n_points = 1000
    #base_dataset = SineDataset(n_points=n_points, center=True, scale=1.)
    #base_dataset = CircleDataset(0, torch.pi, n_points=n_points, center=True)
    base_dataset = ThirdDegreePolynomialDataset(a=-1, b=0, c=1, n_points=n_points, center=True)
    #base_dataset = SpiralDataset(0, 2. * torch.pi, n_points=n_points, center=True)
    #base_dataset = Helix3DDataset(n_points=n_points, starting_angle=0, ending_angle=torch.pi, center=True)
    # window_size = 1000
    # stride = 1
    # dataset = ContiguousSlicesDataset(base_dataset, window_size=window_size, stride=stride)
    dataset = FullTrajectoryDataset(base_dataset)

    # base_dataset = Sphere3DDataset(n_points=1000, polar_angle_range=(0, 0.5 * torch.pi),
    #                                azimuth_range=(0, 0.5 * torch.pi))  # a quarter of a sphere
    # dataset = Sphere3DGeodesicsDataset(base_dataset, target=torch.tensor([0., 0., 1.], dtype=torch.float32),
    #                                    min_segment_length=2e-2, path_length=78)

    dataset_lengths = [0.7, 0.2, 0.1]
    train_dataset, val_dataset, test_dataset = random_split(dataset, dataset_lengths,
                                                            generator=torch.Generator().manual_seed(args.dataset_seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print('full dataset length:', len(dataset))
    print('train dataset length:', len(train_dataset))
    print('val dataset length:', len(val_dataset))
    print('test dataset length:', len(test_dataset))
    print('batch size:', args.batch_size)

    # setup model and training
    pl.seed_everything(args.training_seed)

    base_manifold_cls = Euclidean
    base_manifold_params = dict(d=base_dataset.points.shape[1])

    # encoder_cls = DoubleMLP
    # encoder_params = dict(input_dim=base_dataset.points.shape[1],  # we assume dataset of shape (N, D)
    #                       hidden_dim=128,
    #                       num_hidden_layers=5,
    #                       output_dim=2,
    #                       nonlinearity='relu',
    #                       outermost_linear=True,
    #                       fourier_features=16,
    #                       weights=(1., 1.))

    encoder_cls = DoubleMLP2
    base_mlp_params = dict(in_features=base_dataset.points.shape[1],
                           hidden_features=64,
                           out_features=2,
                           num_hidden_layers=2,
                           outermost_linear=True,
                           fourier_features=None)
    fourier_mlp_params = dict(in_features=base_dataset.points.shape[1],
                              hidden_features=64,
                              out_features=2,
                              num_hidden_layers=0,
                              outermost_linear=True,
                              fourier_features=2048)
    encoder_params = dict(base_mlp_params=base_mlp_params, fourier_mlp_params=fourier_mlp_params, weights=(1., 0.5))

    corrected_manifold_cls = CorrectedEuclideanManifold
    corrected_manifold_params = dict(beta=100., learnable_beta=args.learnable_beta)

    regularizer_cls = CurveNegativeRegularizer
    regularizer_params = dict(pos_hess_weight=0.,
                              neg_hess_weight=10.,
                              min_pos_hess_norm=0.1,
                              max_neg_hess_norm=1e-1,
                              smoothness_weight=0.)

    model = LitManifoldMetricCorrector(encoder_cls, encoder_params,
                                       corrected_manifold_cls, corrected_manifold_params, base_manifold_params,
                                       regularizer_cls, regularizer_params)

    name = f'ReLU_MLPs_analysis'
    logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir, name=name)

    default_callbacks = [MonitorCorrectionCoeff(),
                         pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval='step')]

    custom_callbacks = [PlotPrelogs(base_dataset, freq=10)]
        #InspectGradients('.', 'grads', enable_plotting=False)]
                        #PlotGeodesics(base_dataset, freq=10)]  # must be done on CPU

    callbacks = default_callbacks + custom_callbacks

    #torch.autograd.set_detect_anomaly(True)

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator=args.accelerator,
                         devices=args.devices,
                         default_root_dir=args.trainer_root_dir,
                         logger=logger,
                         log_every_n_steps=1,  # we use 1 batch so we want to log at every batch
                         callbacks=callbacks,
                         limit_val_batches=0.,
                         )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    trainer.test(model, ckpt_path="best", dataloaders=test_loader)



