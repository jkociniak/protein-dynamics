import os
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils.tensor import gradients
from src.datasets.euclidean import MNISTPCADataset


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
        print('BEFORE CLIPPING')
        for n, p in pl_module.named_parameters():
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
                print(f'[{n}] ave_grad: {p.grad.abs().mean()}, max_grad: {p.grad.abs().max()}')
        torch.nn.utils.clip_grad_norm(pl_module.encoder.parameters(), 1.0)

        print('AFTER CLIPPING')
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

            if self.save_dir is None:
                plt.show()
            else:
                plt.savefig(os.path.join(self.save_dir, f'{self.base_name}_{trainer.global_step}.png'))

            plt.close()


class MyTensorBoardLogger(pl.Callback, ABC):
    """ Abstract class for logging, able to save arbitrary metrics, tensors, and figures."""
    def __init__(self, start_epoch=0, freq=10):
        super().__init__()
        self.start_epoch = start_epoch
        self.freq = freq

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return

        if (trainer.current_epoch + 1) % self.freq != 0:
            return

        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)  # make sure we are using tensorboard

        #pl_module.manifold.eval()
        figs, metrics, tensors = self.plot(pl_module.manifold, current_epoch=trainer.current_epoch)
        #pl_module.manifold.train()

        for name, metric in metrics.items():
            pl_module.log(name, metric, on_step=True, on_epoch=False)

        for name, fig in figs.items():
            trainer.logger.experiment.add_figure(name, fig, global_step=trainer.global_step)
            plt.close(fig)

        log_dir = trainer.logger.log_dir
        tensors_dir = os.path.join(log_dir, f'{trainer.current_epoch}')
        # create a directory named current_epoch to save the tensors
        if tensors is not None:
            os.makedirs(tensors_dir, exist_ok=True)

        for name, tensor in tensors.items():
            tensor_path = os.path.join(tensors_dir, f'{name}.pt')
            torch.save(tensor, tensor_path)

    @abstractmethod
    def plot(self, manifold, **kwargs) -> tuple:
        # should return figs, metrics, tensors
        # figs and metrics will be uploaded to tensorboard
        # tensors will be saved in the logdir, in the folder named after the current epoch
        pass


class GeneralLogger(MyTensorBoardLogger, ABC):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.points = dataset.points

    @staticmethod
    def get_mesh_2d(traj, eps, density):
        x_ticks = torch.linspace(traj[:, 0].min() - eps, traj[:, 0].max() + eps, density)
        y_ticks = torch.linspace(traj[:, 1].min() - eps, traj[:, 1].max() + eps, density)
        xv, yv = np.meshgrid(x_ticks, y_ticks)
        x, y = xv.ravel(), yv.ravel()
        xy = np.vstack([x, y]).T
        xy = torch.from_numpy(xy)
        return xy

    @staticmethod
    def get_mesh_3d(traj, eps, density):
        x_ticks = torch.linspace(traj[:, 0].min() - eps, traj[:, 0].max() + eps, density)
        y_ticks = torch.linspace(traj[:, 1].min() - eps, traj[:, 1].max() + eps, density)
        z_ticks = torch.linspace(traj[:, 2].min() - eps, traj[:, 2].max() + eps, density)
        xv, yv, zv = np.meshgrid(x_ticks, y_ticks, z_ticks)
        x, y, z = xv.ravel(), yv.ravel(), zv.ravel()
        xyz = np.vstack([x, y, z]).T
        xyz = torch.from_numpy(xyz)
        return xyz

    @staticmethod
    def compute_interps(manifold, pts, starting_idx, ending_idx, n_interps, rgd_params, use_rgd=True):
        p0 = pts[starting_idx][None, None]
        p1 = pts[ending_idx - 1][None, None]

        ts = torch.linspace(0, 1, n_interps)
        preds = torch.zeros(n_interps, p0.shape[-1])

        manifold.eval()
        for i, t in tqdm(enumerate(ts),
                         desc='Calculating interpolations',
                         total=n_interps):
            t = torch.tensor([t], dtype=torch.float32)
            if use_rgd:
                preds[i] = manifold.geodesic(p0, p1, t, **rgd_params).squeeze()
            else:
                # this is for Euclidean manifold
                preds[i] = manifold.geodesic(p0, p1, t).squeeze()
        manifold.train()

        return preds

    @staticmethod
    def compute_level_set(manifold, pts):
        if pts.shape[-1] == 2:
            mesh = GeneralLogger.get_mesh_2d(pts, 0.1, 100)
        elif pts.shape[-1] == 3:
            mesh = GeneralLogger.get_mesh_3d(pts, 0.1, 30)
        else:
            raise ValueError('Only 2D and 3D data is supported')

        enc, coords = manifold.correction_encoder(mesh[None])

        if pts.shape[-1] == 2:
            enc = enc.reshape(100, 100).detach().numpy()
        elif pts.shape[-1] == 3:
            enc = enc.reshape(30, 30, 30, 2).detach().numpy()
        else:
            raise ValueError('Only 2D and 3D data is supported')
        return enc

    @staticmethod
    def compute_encoder_grads(manifold, pts):
        enc, coords = manifold.correction_encoder(pts[None])
        grads = gradients(enc, coords)
        grads = grads.squeeze(0, 2).detach().cpu()
        return grads

    @staticmethod
    def compute_logs(manifold, pts, ending_idx=-1, use_mesh=False):
        if use_mesh:
            if pts.shape[-1] == 2:
                x = GeneralLogger.get_mesh_2d(pts, 0.1, 30)[None]
            elif pts.shape[-1] == 3:
                x = GeneralLogger.get_mesh_3d(pts, 0.1, 10)[None]
            else:
                raise ValueError('Only 2D and 3D data is supported')
        else:
            x = pts[None]
        target = pts[ending_idx][None, None]
        logs = manifold.log(x, target)
        logs = logs.squeeze(0, 2).detach().cpu()
        return logs

    @staticmethod
    def compute_eigenvalues(manifold, pts):
        total_mt, base_mt, corr_mt = manifold.metric_tensor(pts[None], debug=True)
        (eigenvalues) = np.linalg.eigh(corr_mt.detach().cpu())
        return eigenvalues


class SineExperimentsLogger(GeneralLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interp_params = {
            'starting_idx': 5,
            'ending_idx': 95,
            'n_interps': 90,
            'rgd_params': dict(print_iterations=False, max_iter=500, step_size=0.1, tol=1e-4)
        }

    def on_train_start(self, trainer, pl_module):
        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)  # make sure we are using tensorboard
        log_dir = trainer.logger.log_dir

        gt_path = os.path.join(log_dir, f'gt.pt')
        torch.save(self.points, gt_path)

        base_preds = self.compute_interps(pl_module.manifold.base_manifold, self.points, **self.interp_params, use_rgd=False)
        bp_path = os.path.join(log_dir, f'base_preds.pt')
        torch.save(base_preds, bp_path)

    def compute_interps_local(self, manifold):
        assert (self.interp_params['ending_idx'] - self.interp_params['starting_idx']) == 90
        preds_local = torch.zeros(3, 30, self.points.shape[-1])
        # divide into three equal parts
        x1 = 5
        x2 = 35
        x3 = 65
        x4 = 95
        intervals = [(x1, x2), (x2, x3), (x3, x4)]
        for i, (s, e) in enumerate(intervals):
            interp_params = self.interp_params.copy()
            interp_params['starting_idx'] = s
            interp_params['ending_idx'] = e
            interp_params['n_interps'] = 30
            preds_local[i] = self.compute_interps(manifold, self.points, **interp_params)

        return preds_local

    def plot(self, manifold, **kwargs):
        figs = {}
        metrics = {}
        tensors = {}

        preds = self.compute_interps(manifold, self.points, **self.interp_params)
        tensors['preds'] = preds

        preds_local = self.compute_interps_local(manifold)
        tensors['preds_local'] = preds_local

        encoder_vals = self.compute_level_set(manifold, self.points)
        tensors['encoder_vals'] = encoder_vals

        eigv = self.compute_eigenvalues(manifold, self.points)
        metrics['eigv'] = eigv

        return figs, metrics, tensors


class MNISTLogger(GeneralLogger):
    def __init__(self, dataset, **kwargs):
        assert isinstance(dataset, MNISTPCADataset)
        super().__init__(dataset, **kwargs)
        sorted_indices = sorted(range(self.dataset.indices.shape[0]), key=lambda i: self.dataset.labels[i])
        self.sorted_indices = sorted_indices

    def plot(self, manifold, **kwargs):
        device = next(manifold.correction_encoder.parameters()).device
        manifold.alpha = manifold.alpha.to(device=device)
        manifold.beta = manifold.beta.to(device=device)
        self.points = self.points.to(device=device)
        figs = {}
        metrics = {}
        tensors = {}

        eigv = self.compute_eigenvalues(manifold, self.points)
        tensors['eigv'] = eigv

        metrics['distance_matrix_loss'] = self.compute_distance_matrix_loss(manifold)

        # Draw the distance matrix
        figs['distance_matrix_comp'] = self.draw_distance_matrix_comp(manifold)

        return figs, metrics, tensors

    def compute_distance_matrix_loss(self, manifold):
        pts = self.points[self.sorted_indices]
        corr_distance_matrix = manifold.pairwise_distance(pts[None], pts[None]).squeeze().detach()
        N = len(self.sorted_indices)
        ref = torch.zeros(N, N, device=pts.device)
        ref[N//2:, :N//2] = 1
        ref[:N//2, N//2:] = 1
        loss = torch.nn.functional.mse_loss(corr_distance_matrix, ref)
        print(f'Distance matrix loss shape: {loss.shape}')
        return loss

    def draw_distance_matrix_comp(self, manifold):
        pts = self.points[self.sorted_indices]
        if self.dataset.n_components is None:
            k = pts.shape[1]
        else:
            k = self.dataset.n_components
        distance_matrix = torch.linalg.vector_norm(pts[:, None, :] - pts[None, :, :], ord=2, dim=-1)
        corr_distance_matrix = manifold.pairwise_distance(pts[None], pts[None]).squeeze().detach()

        # Create a figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))

        self.plot_heatmap(ax[0], distance_matrix.detach().cpu(), f'L2 Distance Matrix (first {k} PCA components)')
        self.plot_heatmap(ax[1], corr_distance_matrix.detach().cpu(), f'Corrected Distance Matrix')

        return fig

    @staticmethod
    def plot_heatmap(ax, mat, title):
        # Plot the distance matrix
        im = ax.imshow(mat, cmap='viridis')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Distance', rotation=-90, va="bottom")

        # Set labels and title
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Point Index')
        ax.set_title(title)
