import os
import numpy as np

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..datasets.base import BaseGeodesicsDataset


class MonitorCorrectionCoeff(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        coeff = pl_module.corrected_manifold.beta.data
        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)  # make sure we are using tensorboard
        trainer.logger.experiment.add_scalar('correction_coeff', coeff, global_step=trainer.global_step)


class ScheduleNegSamplesEps(pl.Callback):
    # to work with EnvelopeRegularizer
    def __init__(self, freq=1, factor=0.8):
        super().__init__()
        self.freq = freq
        self.factor = factor

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            print(f'Initial neg samples epsilon: {pl_module.neg_samples_eps}')
            return

        if trainer.current_epoch % self.freq != 0:
            return

        new_eps = pl_module.neg_samples_eps * self.factor
        print(f'Updating neg samples epsilon from {pl_module.neg_samples_eps:04f} to {new_eps:04f}')
        pl_module.neg_samples_eps = new_eps


class InspectGradients(pl.Callback):
    def __init__(self, save_dir, base_name, enable_plotting=True, freq=1):
        super().__init__()
        self.save_dir = save_dir
        self.base_name = base_name
        self.enable_plotting = enable_plotting
        self.freq = freq

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.freq != 0:
            return

        print(f'Plotting gradients at step {trainer.global_step}')

        '''
                Plots the gradients flowing through different layers in the net during training.
                Can be used for checking for possible gradient vanishing / exploding problems.

                Usage: Plug this function in Trainer class after loss.backwards() as
                "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
                '''

        ave_grads = []
        max_grads = []
        layers = []
        for n, p in pl_module.named_parameters():
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

            plt.savefig(os.path.join(self.save_dir, f'{self.base_name}_{trainer.global_step}.png'))
            plt.close()


class PlotPrelogs(pl.Callback):
    def __init__(self, dataset, start_epoch=0, freq=10, density=30, eps=0.5, target=None):
        super().__init__()
        self.start_epoch = start_epoch
        self.freq = freq
        self.density = density
        self.eps = eps

        self.points = dataset.points
        self.dim = dataset.points.shape[1]
        assert self.dim in [2, 3], "Only 2D and 3D points are supported"

        assert isinstance(target, (type(None), torch.Tensor)), "Target must be a tensor or None"
        self.target = target
        if self.target is not None:
            assert self.target.shape[0] == dataset.points.shape[1], "Target must have the same dimension as the points"

        if self.dim == 2:
            x, y, target = self.get_mesh(dataset.points)
            self.x = x
            self.y = y
            self.target = target
        else:
            x, y, z, target = self.get_mesh(dataset.points)
            self.x = x
            self.y = y
            self.z = z
            self.target = target

    def get_mesh(self, points):
        # points should be L x D tensor
        if points.shape[1] == 2:
            x = torch.linspace(points[:, 0].min() - self.eps, points[:, 0].max() + self.eps, self.density)
            y = torch.linspace(points[:, 1].min() - self.eps, points[:, 1].max() + self.eps, self.density)
            if self.target is None:
                target = points[-1, :].reshape(-1, 1, 2)
            else:
                target = self.target
            return x, y, target
        elif points.shape[1] == 3:
            x = torch.linspace(points[:, 0].min() - self.eps, points[:, 0].max() + self.eps, self.density)
            y = torch.linspace(points[:, 1].min() - self.eps, points[:, 1].max() + self.eps, self.density)
            z = torch.linspace(points[:, 2].min() - self.eps, points[:, 2].max() + self.eps, self.density)
            if self.target is None:
                target = points[-1, :].reshape(-1, 1, 3)
            else:
                target = self.target
            return x, y, z, target
        else:
            raise ValueError("Only 2D and 3D points are supported")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return

        if trainer.current_epoch % self.freq != 0:
            return

        if self.dim == 2:
            self.plot_prelogs_2D(trainer, pl_module)
        elif self.dim == 3:
            self.plot_prelogs_3D(trainer, pl_module)
        else:
            raise ValueError("Only 2D and 3D points are supported")

    def plot_prelogs_2D(self, trainer, pl_module):
        xv, yv = np.meshgrid(self.x, self.y)
        xy = np.vstack([xv.ravel(), yv.ravel()]).T
        ps = torch.from_numpy(xy[None])

        ps = ps.to(pl_module.device)
        self.target = self.target.to(pl_module.device)

        with torch.no_grad():
            total_prelogs, _, _, _, = pl_module.corrected_manifold.prelog(ps.float(), self.target.float())
            total_prelogs = total_prelogs.squeeze([0, 2]).detach().cpu().numpy()

        # initialize figure with only one ax
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=50)

        # # plot base prelogs
        # base_prelogs_x = base_prelogs[:, 0].detach().numpy()
        # base_prelogs_y = base_prelogs[:, 1].detach().numpy()
        # ax.quiver(xv, yv, base_prelogs_x, base_prelogs_y, width=0.002, color='green', label='Base prelogs', alpha=0.3)
        #
        # plot correction prelogs
        # corr_prelogs_x = correction_prelogs[:, 0].detach().numpy()
        # corr_prelogs_y = correction_prelogs[:, 1].detach().numpy()
        # ax[0].quiver(xv, yv, corr_prelogs_x, corr_prelogs_y, width=0.002, color='red', label='Correction prelogs')

        ax.plot(self.points[:, 0], self.points[:, 1], label='True', color='black', alpha=0.5)
        # plot total prelogs
        total_prelogs_x = total_prelogs[:, 0]
        total_prelogs_y = total_prelogs[:, 1]
        ax.quiver(xv, yv, total_prelogs_x, total_prelogs_y, width=0.002, color='blue', label='Distance grads')

        # enable legend
        ax.legend()

        # upload to tensorboard
        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)  # make sure we are using tensorboard
        trainer.logger.experiment.add_figure('Prelogs', fig, global_step=trainer.global_step)
        plt.close(fig)

    def plot_prelogs_3D(self, trainer, pl_module):
        xv, yv, zv = np.meshgrid(self.x, self.y, self.z)
        xyz = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
        ps = torch.from_numpy(xyz[None]).float()
        target = self.target.float()[None, None]

        total_prelogs, _, _, _, = pl_module.corrected_manifold.prelog(ps, target)
        total_prelogs = total_prelogs.squeeze([0, 2, 3])

        # initialize figure with only one ax
        fig = plt.figure(figsize=(10, 10), dpi=50)
        ax = fig.add_subplot(111, projection='3d')

        # plot total prelogs
        total_prelogs = total_prelogs.detach().numpy()
        ax.quiver(xv.ravel(), yv.ravel(), zv.ravel(),
                  total_prelogs[:, 0], total_prelogs[:, 1], total_prelogs[:, 2],
                  normalize=True, length=0.2, color='blue', label='Distance grads')

        # enable legend
        ax.legend()

        # upload to tensorboard
        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)
        trainer.logger.experiment.add_figure('Prelogs', fig, global_step=trainer.global_step)
        plt.close(fig)


class PlotGeodesics(pl.Callback):
    def __init__(self, dataset, start_epoch=0, freq=10, n_interps=21, window_size=200, plot_negative_samples=False):
        super().__init__()
        self.start_epoch = start_epoch
        self.freq = freq

        self.dataset = dataset
        self.n_points = len(dataset)

        self.n_interps = n_interps
        self.window_size = window_size


        # we will plot predicted geodesics for window size, sliding by window size,
        # so we need to define appropriate list of colors for each window
        # the colors will change gradually from red to blue
        first_color = np.array([1, 0, 0])
        last_color = np.array([0, 0, 1])
        n_windows = self.n_points // window_size
        self.colors = np.linspace(first_color, last_color, n_windows)

        assert isinstance(plot_negative_samples, bool)
        self.negative_samples = plot_negative_samples

    def on_train_epoch_end(self, trainer, pl_module):
        # we want to run callback every 10th epoch
        if trainer.current_epoch < self.start_epoch:
            return

        if trainer.current_epoch % self.freq != 0:
            return

        old_device = pl_module.device
        pl_module.to('cpu')

        def plot_geodesic(ax, i, j, color):
            p0 = self.dataset.points[i][None, None]
            assert p0.shape[-1] in [2, 3], "Only 2D and 3D points are supported"
            p1 = self.dataset.points[j - 1][None, None]

            ts = torch.linspace(0, 1, self.n_interps)
            preds = torch.zeros(self.n_interps, p0.shape[-1])
            for i, t in enumerate(ts):
                t = torch.tensor([t], dtype=torch.float32)
                preds[i] = pl_module.corrected_manifold.geodesic(p0, p1, t).squeeze()

            preds = preds.detach().unbind(dim=1)
            ax.scatter(*preds, s=8, label='predicted', color=color)

        pts = self.dataset.points.unbind(dim=1)
        assert len(pts) in [2, 3], "Only 2D and 3D points are supported"

        # initialize figure with two axes
        projection = '3d' if len(pts) == 3 else None
        fig = plt.figure(figsize=(24, 10), dpi=50)
        ax = [fig.add_subplot(1, 2, i + 1, projection=projection) for i in range(2)]

        # 1. partial geodesics (+ negative samples optionally)
        ax[0].plot(*pts, label='true', color='black', alpha=0.5)
        for i in range(0, self.n_points, self.window_size):
            color = self.colors[i // self.window_size]
            j = min(i + self.window_size, self.n_points)
            plot_geodesic(ax[0], i, j, color)
        ax[0].legend()

        # 2. full geodesic
        ax[1].plot(*pts, label='true', color='black', alpha=0.5)
        plot_geodesic(ax[1], 0, self.n_points, 'blue')
        ax[1].legend()

        pl_module.to(old_device)

        # upload to tensorboard
        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)
        trainer.logger.experiment.add_figure('Geodesics', fig, global_step=trainer.global_step)
        plt.close(fig)


class PlotGeodesicsDataset(pl.Callback):
    def __init__(self, dataset: BaseGeodesicsDataset, start_epoch=0, freq=10, max_geodesics=10):
        super().__init__()
        self.start_epoch = start_epoch
        self.freq = freq
        self.max_geodesics = max_geodesics

        self.dataset = dataset
        self.n_points = len(dataset)

        # we will plot predicted geodesics for window size, sliding by window size,
        # so we need to define appropriate list of colors for each window
        # the colors will change gradually from red to blue
        # first_color = np.array([1, 0, 0])
        # last_color = np.array([0, 0, 1])
        # n_windows = self.n_points // window_size
        # self.colors = np.linspace(first_color, last_color, n_windows)

    def on_train_epoch_end(self, trainer, pl_module):
        # we want to run callback every 10th epoch
        if trainer.current_epoch < self.start_epoch:
            return

        if trainer.current_epoch % self.freq != 0:
            return

        # initialize figure with two axes
        # initialize figure with only one ax
        fig = plt.figure(figsize=(10, 10), dpi=50)
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(self.dataset.geodesics)):
            if i == self.max_geodesics:
                break
            geodesic = self.dataset.geodesics[i]
            ax.plot(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2], label='true', color='black', alpha=0.5)

            p0 = geodesic[0][None, None]
            p1 = geodesic[-1][None, None]
            n_interps = len(geodesic)
            ts = torch.linspace(0, 1, n_interps)
            preds = torch.zeros(n_interps, p0.shape[-1])

            for j, t in enumerate(ts):
                preds[j] = pl_module.corrected_manifold.geodesic(p0, p1, torch.tensor([t])).squeeze()

            preds = [p.numpy() for p in preds.detach().unbind(dim=1)]
            color = np.random.rand(3, )
            ax.scatter(*preds, s=8, label='predicted', color=color)

        ax.view_init(30, 30)

        # upload to tensorboard
        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)
        trainer.logger.experiment.add_figure('Geodesics', fig, global_step=trainer.global_step)