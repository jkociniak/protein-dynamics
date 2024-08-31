from abc import ABC, abstractmethod
import os
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from src.utils.tensor import gradients


class MyPlotter(pl.Callback, ABC):
    """ Abstract class for plotting, able to log arbitrary metrics, save tensors, plot and save figures."""
    def __init__(self, points, start_epoch=0, freq=10):
        super().__init__()
        self.points = points
        self.start_epoch = start_epoch
        self.freq = freq

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return

        if (trainer.current_epoch + 1) % self.freq != 0:
            return

        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)  # make sure we are using tensorboard

        # save checkpoint of pl_module
        figs, metrics, tensors = self.plot(pl_module.manifold, current_epoch=trainer.current_epoch)

        for name, metric in metrics.items():
            trainer.logger.experiment.add_scalar(name, metric, global_step=trainer.global_step)

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


class Plotter2D(MyPlotter):
    """ Callbacks usable for 2-dimensional data."""

    def __init__(self, points, gradient_plot_params=None, interp_params=None, **kwargs):
        super().__init__(points, **kwargs)
        # if gradient_plot_params is None:
        #     gradient_plot_params = dict(
        #         enable=True,
        #         starting_idx=10,
        #         ending_idx=15
        #     )
        # self.enable_gradients = gradient_plot_params['enable']
        # del gradient_plot_params['enable']
        # self.gradient_plot_params = gradient_plot_params

        if interp_params is None:
            interp_params = {
                'starting_idx': 5,
                'ending_idx': 95,
                'n_interps': 90,
                'rgd_params': dict(print_iterations=False, max_iter=500, step_size=0.1, tol=1e-4)
            }
        self.interp_params = interp_params

    def on_train_start(self, trainer, pl_module):
        assert isinstance(trainer.logger, pl.loggers.TensorBoardLogger)  # make sure we are using tensorboard
        log_dir = trainer.logger.log_dir

        gt_path = os.path.join(log_dir, f'gt.pt')
        torch.save(self.points, gt_path)

        base_preds = self.compute_interps(pl_module.manifold.base_manifold, self.points, **self.interp_params, use_rgd=False)
        bp_path = os.path.join(log_dir, f'base_preds.pt')
        torch.save(base_preds, bp_path)

    def plot(self, manifold, **kwargs):
        figs = {}
        metrics = {}
        tensors = {}

        preds = self.compute_interps(manifold, self.points, **self.interp_params)
        tensors['preds'] = preds

        # preds_local = self.compute_interps_local(manifold)
        # tensors['preds_local'] = preds_local

        # encoder_vals = self.compute_level_set(manifold, self.points)
        # tensors['encoder_vals'] = encoder_vals

        # logs_mesh = self.compute_logs(manifold, self.points, use_mesh=True)
        # tensors['logs'] = logs_mesh

        grads = self.compute_encoder_grads(manifold, self.points)
        tensors['grads'] = grads

        eigenvalues = self.compute_eigenvalues(manifold, self.points)
        tensors['eigv'] = eigenvalues

        logs_traj = self.compute_logs(manifold, self.points)
        tensors['logs_traj'] = logs_traj

        return figs, metrics, tensors

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
            mesh = Plotter2D.get_mesh_2d(pts, 0.1, 100)
        elif pts.shape[-1] == 3:
            mesh = Plotter2D.get_mesh_3d(pts, 0.1, 30)
        else:
            raise ValueError('Only 2D and 3D data is supported')

        enc, coords = manifold.correction_encoder(mesh[None])

        if pts.shape[-1] == 2:
            enc = enc.reshape(100, 100).detach().numpy()
        elif pts.shape[-1] == 3:
            print(enc.shape)
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
    def compute_logs(manifold, pts, use_mesh=False):
        if use_mesh:
            if pts.shape[-1] == 2:
                x = Plotter2D.get_mesh_2d(pts, 0.1, 30)[None]
            elif pts.shape[-1] == 3:
                x = Plotter2D.get_mesh_3d(pts, 0.1, 10)[None]
            else:
                raise ValueError('Only 2D and 3D data is supported')
        else:
            x = pts[None]
        target = pts[-1][None, None]
        logs = manifold.log(x, target)
        logs = logs.squeeze(0, 2).detach().cpu()
        return logs

    @staticmethod
    def compute_eigenvalues(manifold, pts):
        total_mt, base_mt, corr_mt = manifold.metric_tensor(pts[None], debug=True)
        (eigenvalues) = np.linalg.eigh(corr_mt.detach())
        return eigenvalues


class PlotterND(MyPlotter):
    """ Callbacks usable for n-dimensional data."""

    def __init__(self, points, gradient_plot_params=None, geodesic_eval_params=None, **kwargs):
        super().__init__(points, **kwargs)
        if gradient_plot_params is None:
            gradient_plot_params = dict(
                enable=True,
                starting_idx=10,
                ending_idx=15
            )
        self.enable_gradients = gradient_plot_params['enable']
        del gradient_plot_params['enable']
        self.gradient_plot_params = gradient_plot_params

        if geodesic_eval_params is None:
            geodesic_eval_params = dict(
                enable=True,
                starting_idx=30,
                ending_idx=50,
                rgd_params=dict(debug=False, max_iter=500, step_size=0.1)
            )
        self.enable_geodesic_eval = geodesic_eval_params['enable']
        del geodesic_eval_params['enable']
        self.geodesic_eval_params = geodesic_eval_params

    def plot(self, manifold, **kwargs):
        figs = {}
        metrics = {}

        if self.enable_gradients:
            figs['Gradients'] = self.plot_gradient_analysis(manifold, self.points, **self.gradient_plot_params)

        if self.enable_geodesic_eval:
            loss = self.eval_geodesics(manifold, self.points, **self.geodesic_eval_params)
            metrics['interp_rmse'] = loss

        return figs, metrics, {}

    @staticmethod
    def plot_gradient_analysis(manifold, points, starting_idx, ending_idx):
        pts = points[starting_idx:ending_idx]
        pts_enc, coords = manifold.correction_encoder(pts)
        grads = gradients(pts_enc, coords)  # dimensions: (L, enc_dim, D)
        grads_gram = torch.einsum("Lid,Ljd->Lij", grads, grads)  # dimensions: (L, enc_dim, enc_dim)
        grads_gram = grads_gram.detach().numpy()

        # self.gram_matrices.append(grads_gram)

        # plot the heatmap of the gradient's gram matrix
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(121)
        cax = ax.matshow(grads_gram[0], cmap='viridis')
        fig.colorbar(cax)

        # plot labels
        ax.set_title('Gradient Gram Matrix at point ' + str(starting_idx))
        ax.set_xticks(range(grads_gram.shape[0]))
        ax.set_yticks(range(grads_gram.shape[1]))
        ax.set_xticklabels(range(grads_gram.shape[0]))
        ax.set_yticklabels(range(grads_gram.shape[1]))

        ax = fig.add_subplot(122)
        cov_matrix = grads_gram[0].T @ grads_gram[0]
        (eigenvalues, eigenvectors) = np.linalg.eigh(cov_matrix)
        ax.plot(eigenvalues, 'o-')
        ax.set_title('Eigenvalues of the Gradient Cov Matrix at point ' + str(starting_idx))
        return fig

    @staticmethod
    def eval_geodesics(manifold, curve, starting_idx, ending_idx, rgd_params=None):
        if rgd_params is None:
            rgd_params = dict(debug=False, max_iter=100, step_size=1.)  # defaults
        p0 = curve[starting_idx][None, None]
        p1 = curve[ending_idx][None, None]
        n_interps = ending_idx - starting_idx + 1

        ts = torch.linspace(0, 1, n_interps)
        preds = torch.zeros(n_interps, p0.shape[-1])

        for i, t in enumerate(ts):
            t = torch.tensor([t], dtype=torch.float32)
            preds[i] = manifold.geodesic(p0, p1, t, **rgd_params).squeeze()

        interp_rmse = torch.linalg.norm(preds - curve[starting_idx:ending_idx + 1], dim=1).mean()
        return interp_rmse


class Plotter3D(MyPlotter):
    def __init__(self, points, dist_grad_params=None, geodesic_plot_params=None, **kwargs):
        super().__init__(points, **kwargs)
        self.points = points
        if dist_grad_params is None:
            dist_grad_params = dict(enable=True,
                                    mesh_eps=0.5,
                                    mesh_density=10)

        self.enable_dist_grads = dist_grad_params['enable']
        del dist_grad_params['enable']
        self.dist_grad_params = dist_grad_params

        eps = self.dist_grad_params['mesh_eps']
        density = self.dist_grad_params['mesh_density']
        x = torch.linspace(points[:, 0].min() - eps, points[:, 0].max() + eps, density)
        y = torch.linspace(points[:, 1].min() - eps, points[:, 1].max() + eps, density)
        z = torch.linspace(points[:, 2].min() - eps, points[:, 2].max() + eps, density)
        self.target = points[-1].reshape(1, 1, points.shape[1])

        xv, yv, zv = np.meshgrid(x, y, z)
        self.mesh_indices = (xv.ravel(), yv.ravel(), zv.ravel())
        xyz = np.vstack(self.mesh_indices).T
        self.ps = torch.from_numpy(xyz[None])

        if geodesic_plot_params is None:
            geodesic_plot_params = dict(enable=True,
                                        starting_idx=0,
                                        ending_idx=points.shape[0] - 1,
                                        n_interps=10,
                                        rgd_params=None,
                                        eval=False)
        self.enable_geodesic_plot = geodesic_plot_params['enable']
        del geodesic_plot_params['enable']
        self.geodesic_plot_params = geodesic_plot_params

    def plot(self, manifold):
        figs = {}
        metrics = {}

        if self.enable_geodesic_plot:
            fig, loss = self.plot_geodesics_3D(manifold, self.points, **self.geodesic_plot_params)
            figs['Geodesics'] = fig
            if loss is not None:
                metrics['interp_rmse'] = loss

        if self.enable_dist_grads:
            figs['DistGrads'] = self.plot_dist_grads_3D(manifold, self.points, self.mesh_indices, self.target)

        return figs, metrics

    @staticmethod
    def plot_dist_grads_3D(manifold, points, mesh_indices, target):
        fig = plt.figure(figsize=(20, 10), dpi=100)
        ax = fig.add_subplot(121, projection='3d')

        prelogs, _, _, _, = manifold.prelog(mesh_indices, target)
        prelogs = prelogs.squeeze([0, 2, 3]).detach().cpu().numpy()
        prelogs = 1e-1 * prelogs / np.linalg.norm(prelogs, axis=1, keepdims=True)

        ax.plot(points[:, 0], points[:, 1], points[:, 2], label='True', color='black', alpha=0.5)
        ax.quiver(*mesh_indices, prelogs[:, 0], prelogs[:, 1], prelogs[:, 2],
                  alpha=0.7, linewidths=0.3, arrow_length_ratio=0.6, color='blue',
                  label='Distance grads (Euclidean)')
        ax.legend()

        ax = fig.add_subplot(122, projection='3d')

        logs = manifold.log(mesh_indices, target)
        logs = logs.squeeze([0, 2, 3]).detach().cpu().numpy()
        logs = 1e-1 * logs / np.linalg.norm(logs, axis=1, keepdims=True)

        ax.plot(points[:, 0], points[:, 1], points[:, 2], label='True', color='black', alpha=0.5)
        ax.quiver(*mesh_indices, logs[:, 0], logs[:, 1], logs[:, 2],
                  alpha=0.7, linewidths=0.3, arrow_length_ratio=0.6, color='blue',
                  label='Distance grads (Riemannian)')
        ax.legend()

        return fig

    @staticmethod
    def plot_geodesics_3D(manifold, curve, starting_idx, ending_idx, n_interps, rgd_params, eval):
        p0 = curve[starting_idx][None, None]
        p1 = curve[ending_idx][None, None]

        if eval:
            n_interps = ending_idx - starting_idx + 1

        ts = torch.linspace(0, 1, n_interps)
        preds = torch.zeros(n_interps, p0.shape[-1])

        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        for i, t in enumerate(ts):
            t = torch.tensor([t], dtype=torch.float32)
            preds[i] = manifold.geodesic(p0, p1, t, **rgd_params).squeeze()

        # curve plot
        ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2], label='true', color='black', s=0.2)

        # preds plot
        ax.scatter(*preds.detach().unbind(dim=1), label='interpolations')

        # first pred
        ax.scatter(curve[starting_idx, 0], curve[starting_idx, 1], curve[starting_idx, 2],
                   label='first point')
        # second pred
        ax.scatter(curve[ending_idx, 0], curve[ending_idx, 1], curve[ending_idx, 2],
                   label='last point')

        # if self.last_negatives is not None:
        #     ax.scatter(self.last_negatives[:, 0], self.last_negatives[:, 1], self.last_negatives[:, 2],
        #                label='negative examples', s=0.2)

        if eval:
            metric = torch.linalg.norm(preds - curve[starting_idx:ending_idx + 1], dim=1).mean()
        else:
            metric = None

        return fig, metric


class Plotter3DPointCloud(MyPlotter):
    def __init__(self, points, geodesic_plot_params=None, **kwargs):
        super().__init__(points, **kwargs)

        if geodesic_plot_params is None:
            geodesic_plot_params = dict(enable=True,
                                        starting_idx=0,
                                        ending_idx=points.shape[0] - 1,
                                        n_interps=10,
                                        rgd_params=None,
                                        eval=False)
        self.geodesic_plot_params = geodesic_plot_params

    def plot(self, manifold):
        figs = {}
        metrics = {}

        fig, loss = self.plot_geodesics_3D(manifold, self.points, **self.geodesic_plot_params)
        figs['Geodesics'] = fig
        if loss is not None:
            metrics['interp_rmse'] = loss

        return figs, metrics


