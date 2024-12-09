import os
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import plotly.graph_objects as go

from src.utils.tensor import gradients
from src.datasets.euclidean.mnist import MNISTPCADataset


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

class FlexibleLogger(pl.Callback, ABC):
    """ Abstract class for logging, able to save arbitrary metrics, tensors, and figures to TensorBoard or Weights and Biases."""

    def __init__(self, start_epoch=0, freq=10):
        super().__init__()
        self.start_epoch = start_epoch
        self.freq = freq

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return

        if trainer.current_epoch % self.freq != 0:
            return

        assert isinstance(trainer.logger,
                          (TensorBoardLogger, WandbLogger)), "Logger must be either TensorBoardLogger or WandbLogger"

        self.last_negatives = pl_module.loss.last_negatives
        self.last_neg_directions = pl_module.loss.last_neg_directions
        self.last_neg_points = pl_module.loss.last_neg_points

        # pl_module.manifold.eval()
        figs, metrics, tensors = self.plot(pl_module.manifold, current_epoch=trainer.current_epoch)
        # pl_module.manifold.train()

        # Log metrics
        for name, metric in metrics.items():
            print(f'Logging metric {name} with value {metric} at epoch {trainer.current_epoch}')
            if isinstance(trainer.logger, TensorBoardLogger):
                trainer.logger.experiment.add_scalar(name, metric, global_step=trainer.global_step)
            else:  # WandbLogger
                pl_module.log(name, metric)

        # Log figures
        for name, fig in figs.items():
            if isinstance(trainer.logger, TensorBoardLogger):
                trainer.logger.experiment.add_figure(name, fig, global_step=trainer.global_step)
            else:  # WandbLogger
                trainer.logger.experiment.log({name: fig})
            if isinstance(fig, plt.Figure):
                plt.close(fig)

        # Save tensors
        if tensors is not None:
            if isinstance(trainer.logger, TensorBoardLogger):
                tensors_dir = os.path.join(trainer.logger.log_dir, f'tensors/epoch_{trainer.current_epoch}')
            else:  # WandbLogger
                tensors_dir = os.path.join(trainer.logger.experiment.dir, f'tensors/epoch_{trainer.current_epoch}')

            os.makedirs(tensors_dir, exist_ok=True)

            for name, tensor in tensors.items():
                tensor_path = os.path.join(tensors_dir, f'{name}.pt')
                torch.save(tensor, tensor_path)
                if isinstance(trainer.logger, WandbLogger):
                    trainer.logger.experiment.save(tensor_path)

    @abstractmethod
    def plot(self, manifold, **kwargs) -> tuple:
        # should return figs, metrics, tensors
        # figs and metrics will be uploaded to the logger
        # tensors will be saved in the logger's directory, in a folder named after the current epoch
        pass


class GeneralLogger(FlexibleLogger, ABC):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.points = dataset.points

    @staticmethod
    def get_mesh_2d(pc, eps, density):
        x_ticks = torch.linspace(pc[:, 0].min() - eps, pc[:, 0].max() + eps, density)
        y_ticks = torch.linspace(pc[:, 1].min() - eps, pc[:, 1].max() + eps, density)
        xv, yv = np.meshgrid(x_ticks, y_ticks)
        x, y = xv.ravel(), yv.ravel()
        xy = np.vstack([x, y]).T
        xy = torch.from_numpy(xy)
        return xy

    @staticmethod
    def get_mesh_3d(pc, eps, density):
        x_ticks = torch.linspace(pc[:, 0].min() - eps, pc[:, 0].max() + eps, density)
        y_ticks = torch.linspace(pc[:, 1].min() - eps, pc[:, 1].max() + eps, density)
        z_ticks = torch.linspace(pc[:, 2].min() - eps, pc[:, 2].max() + eps, density)
        xv, yv, zv = np.meshgrid(x_ticks, y_ticks, z_ticks)
        x, y, z = xv.ravel(), yv.ravel(), zv.ravel()
        xyz = np.vstack([x, y, z]).T
        xyz = torch.from_numpy(xyz)
        return xyz

    @staticmethod
    def compute_interps(manifold, pts, starting_idx, ending_idx, n_interps, rgd_params, use_rgd=True):
        p0 = pts[starting_idx][None, None]
        p1 = pts[ending_idx][None, None]

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
            enc = enc.reshape(30, 30, 30, -1).detach().numpy()
        else:
            raise ValueError('Only 2D and 3D data is supported')

        return enc, mesh

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
        eigenvalues, _ = np.linalg.eigh(corr_mt.detach().cpu())
        return eigenvalues


class SineExperimentsLogger(GeneralLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interp_params = {
            'starting_idx': 8,
            'ending_idx': 156,
            'n_interps': 20,
            'rgd_params': dict(print_iterations=False, max_iter=500, step_size=0.1, tol=1e-4)
        }

        self.interp_indices = [(86, 0), (0, 86), (17, 0), (0, 17), (17, 29), (29, 17), (29, 86), (86, 29)]

    def plot(self, manifold, **kwargs):
        if self.points.shape[-1] == 2:
            return self.plot_2d(manifold, **kwargs)
        elif self.points.shape[-1] == 3:
            return self.plot_3d(manifold, **kwargs)
        else:
            return self.plot_nd()

    @staticmethod
    def create_arrow_plot_3d(points, vectors, colors, name='Vectors'):
        arrow_length = 0.05  # Adjust this value to change the length of the arrows

        # Create lines
        x_lines = []
        y_lines = []
        z_lines = []
        line_colors = []
        for point, vector, color in zip(points, vectors, colors):
            x_lines.extend([point[0], point[0] + arrow_length * vector[0], None])
            y_lines.extend([point[1], point[1] + arrow_length * vector[1], None])
            z_lines.extend([point[2], point[2] + arrow_length * vector[2], None])
            line_colors.extend([color, color, color])

        lines = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(color=line_colors, width=2),
            hoverinfo="none",
            name=f"{name}"
        )

        return [lines]

    @staticmethod
    def create_arrow_plot_2d(points, vectors, colors, name='Vectors'):
        arrow_length = 0.05  # Adjust this value to change the length of the arrows

        # Create lines
        x_lines = []
        y_lines = []
        line_colors = []
        for point, vector, color in zip(points, vectors, colors):
            x_lines.extend([point[0], point[0] + arrow_length * vector[0], None])
            y_lines.extend([point[1], point[1] + arrow_length * vector[1], None])
            #line_colors.extend([color, color])

        lines = go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(width=2),
            hoverinfo="none",
            name=f"{name}"
        )

        return [lines]

    def on_train_start(self, trainer, pl_module):
        if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
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

    def plot_2d(self, manifold, **kwargs):
        figs = {}
        metrics = {}
        tensors = {}

        preds = self.compute_interps(manifold, self.points, **self.interp_params).detach()
        # tensors['preds'] = preds

        fig = go.Figure([go.Scatter(x=self.points[:, 0], y=self.points[:, 1], mode='markers', name='Ground Truth'),
                         go.Scatter(x=preds[:, 0], y=preds[:, 1], mode='markers', name='Predictions')])

        figs['interpolations_global'] = fig

        encoder_vals, mesh = self.compute_level_set(manifold, self.points)

        X = mesh[:, 0]
        Y = mesh[:, 1]

        contour = go.Contour(
            x=X.flatten(),
            y=Y.flatten(),
            z=encoder_vals.flatten(),
            autocontour=True,
            name=f'SDF Level Set'
        )
        fig = go.Figure(contour)
        figs['level_set'] = fig

        gt_fig = go.Figure()

        normals = self.dataset.data_list[0].normal_basis
        nsd = self.dataset.data_list[0].normal_space_dims

        max_nsd = nsd.max().item()
        for i in range(0, max_nsd):
            d_mask = nsd >= (i + 1)
            normal_bases = [normals[i] for i, mask in enumerate(d_mask) if mask.item()]
            normal_bases = torch.stack(normal_bases, dim=0)
            arrows = self.create_arrow_plot_2d(
                self.points[d_mask],
                normal_bases[:, i, :],
                colors=['blue'] * normal_bases.shape[0],
                name=f'Estimated normal field {i}'
            )
            gt_fig.add_traces(arrows)

        pts_enc, coords = manifold.correction_encoder(self.points)
        grads = gradients(pts_enc, coords).detach()  # dimensions: (N, enc_dim, D)
        grads = grads / torch.linalg.norm(grads, dim=-1, keepdim=True)

        for i in range(grads.shape[1]):
            arrows = self.create_arrow_plot_2d(
                self.points,
                grads[:, i, :],
                colors=['red'] * grads.shape[0],
                name=f'Gradient {i}'
            )
            gt_fig.add_traces(arrows)

        gt_fig.add_trace(go.Scatter(
            x=self.points[:, 0],
            y=self.points[:, 1],
            mode='markers',
            marker=dict(color='black', size=3),
            name='Dataset Points'
        ))

        for i in range(self.last_neg_directions.shape[1]):
            gt_fig.add_traces(self.create_arrow_plot_2d(self.last_neg_points,
                                                        self.last_neg_directions[:, i, :],
                                                        colors=['yellow'] * len(self.last_neg_directions),
                                                        name=f'Last normal frame, vector {i+1}'))

        gt_fig.add_trace(go.Scatter(
            x=self.last_negatives[:, 0],
            y=self.last_negatives[:, 1],
            mode='markers',
            marker=dict(color='red', size=3),
            name='Last Negative Points'
        ))

        gt_fig.update_layout(
            title='Ground truth visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                aspectmode='data'
            ),
            legend=dict(x=1.05, y=0.5)
        )
        figs['gt'] = gt_fig

        return figs, metrics, tensors

    def plot_3d(self, manifold, **kwargs):
        figs = {}
        metrics = {}
        tensors = {}

        for i, (s, e) in enumerate(self.interp_indices):
            print(f'Computing interpolations for indices {s} and {e}')
            interp_params = self.interp_params.copy()
            interp_params['starting_idx'] = s
            interp_params['ending_idx'] = e
            preds = self.compute_interps(manifold, self.points, **interp_params).detach()

            fig = go.Figure(data=[go.Scatter3d(x=self.points[:, 0], y=self.points[:, 1], z=self.points[:, 2],
                                               mode='markers', marker=dict(size=2), name='Ground Truth'),
                                  go.Scatter3d(x=preds[:, 0], y=preds[:, 1], z=preds[:, 2],
                                               mode='markers', marker=dict(size=2), name='Predictions')])

            ids = [s, e]
            fig.add_trace(
                go.Scatter3d(x=self.points[ids, 0], y=self.points[ids, 1], z=self.points[ids, 2], mode='markers', name='Endpoints',
                             marker=dict(size=5)))
            fig.update_layout(
                title=f'Interpolation visualisation (Start: {s}, End: {e})',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                legend=dict(x=1.05, y=0.5)
            )
            figs[f'interpolations_{s}_{e}'] = fig

        # encoder_vals, mesh = self.compute_level_set(manifold, self.points)
        #
        # X = mesh[:, 0]
        # Y = mesh[:, 1]
        # Z = mesh[:, 2]
        # encoder_vals = np.linalg.norm(encoder_vals, axis=-1)
        #
        # contour = go.Isosurface(
        #     x=X.flatten(),
        #     y=Y.flatten(),
        #     z=Z.flatten(),
        #     value=encoder_vals.flatten(),
        #     opacity=0.5,
        #     isomin=0.,
        #     isomax=0.,
        #     colorscale='Viridis',
        #     name=f'SDF Level Set'
        # )
        # fig = go.Figure(contour)
        # figs['level_set'] = fig

        #eigv = self.compute_eigenvalues(manifold, self.points)
        #tensors['eigv'] = eigv

        normals = self.dataset.data_list[0].normal_basis
        nsd = self.dataset.data_list[0].normal_space_dims

        gt_fig = go.Figure()

        max_nsd = nsd.max().item()
        for i in range(0, max_nsd):
            d_mask = nsd >= (i + 1)
            normal_bases = [normals[j][i] for j, mask in enumerate(d_mask) if mask.item()]
            normal_bases = torch.stack(normal_bases, dim=0)
            arrows = self.create_arrow_plot_3d(
                self.points[d_mask],
                normal_bases,
                colors=['blue'] * normal_bases.shape[0],
                name=f'Estimated normal field {i}'
            )
            gt_fig.add_traces(arrows)

        pts_enc, coords = manifold.correction_encoder(self.points)
        grads = gradients(pts_enc, coords).detach()  # dimensions: (N, enc_dim, D)

        # Add unsmoothed normal vectors with colors
        for i in range(grads.shape[1]):
            unsmoothed_arrows = self.create_arrow_plot_3d(
                self.points,
                grads[:, i, :],
                colors=['red'] * grads.shape[0],
                name=f'Gradient {i}'
            )
            gt_fig.add_traces(unsmoothed_arrows)
        # Add the dataset points
        gt_fig.add_trace(go.Scatter3d(
            x=self.points[:, 0],
            y=self.points[:, 1],
            z=self.points[:, 2],
            mode='markers',
            marker=dict(color='black', size=2),
            name='Dataset Points'
        ))

        gt_fig.add_trace(go.Scatter3d(
            x=self.last_negatives[:, 0],
            y=self.last_negatives[:, 1],
            z=self.last_negatives[:, 2],
            mode='markers',
            marker=dict(color='red', size=2),
            name='Last Negative Points'
        ))

        gt_fig.update_layout(
            title='Ground truth visualisation',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            legend=dict(x=1.05, y=0.5)
        )
        figs['gt'] = gt_fig

        return figs, metrics, tensors

    def plot_nd(self):
        figs = {}
        metrics = {}
        tensors = {}
        return figs, metrics, tensors


class MNISTLogger(GeneralLogger):
    def __init__(self, dataset, **kwargs):
        assert isinstance(dataset, MNISTPCADataset) or isinstance(dataset, torch.utils.data.Subset)
        super().__init__(dataset, **kwargs)
        self.sorted_indices = sorted(dataset.indices, key=lambda i: dataset.dataset.labels[i])

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        def validate_fun(manifold, batch):
            x, label = batch
            sorted_indices = sorted(range(x.shape[0]), key=lambda i: label[i])
            distance_loss = self.distance_matrix_loss(manifold, x[sorted_indices])
            return 'distance_matrix_loss', distance_loss
        pl_module.validate_fun = validate_fun

    def plot(self, manifold, **kwargs):
        device = next(manifold.correction_encoder.parameters()).device
        self.points = self.points.to(device=device)
        figs = {}
        metrics = {}
        tensors = {}

        eigv = self.compute_eigenvalues(manifold, self.points)
        tensors['eigv'] = eigv

        pts = self.points[self.sorted_indices]
        metrics['train/distance_matrix_loss'] = self.distance_matrix_loss(manifold, pts)

        # Draw the distance matrix
        figs['distance_matrix_comp'] = self.draw_distance_matrix_comp(manifold)
        figs['gradient_analysis'] = self.draw_gradient_analysis(manifold)

        return figs, metrics, tensors

    @staticmethod
    def distance_matrix_loss(manifold, pts):
        # assume that the points are sorted by label
        corr_distance_matrix = manifold.pairwise_distance(pts[None], pts[None]).squeeze().detach()
        N = corr_distance_matrix.shape[0]
        assert N % 2 == 0
        corr_distance_matrix[N // 2:, :N // 2] = 1 / corr_distance_matrix[N // 2:, :N // 2]
        corr_distance_matrix[:N // 2, N // 2:] = 1 / corr_distance_matrix[:N // 2, N // 2:]
        loss = torch.nn.functional.mse_loss(corr_distance_matrix, torch.zeros(N, N, device=pts.device))
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
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))

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

    def draw_gradient_analysis(self, manifold):
        fig = plt.figure(figsize=(18, 12))
        i = 1
        for point_id in [0, 50, 150, 199]:
            pts = self.points[point_id][None, None]
            pts_enc, coords = manifold.correction_encoder(pts)
            grads = gradients(pts_enc, coords)[0, 0, ...]  # dimensions: (enc_dim, D)
            grads_gram = torch.einsum("id,jd->ij", grads, grads)  # dimensions: (enc_dim, enc_dim)
            grads_gram = grads_gram.detach().cpu().numpy()

            # self.gram_matrices.append(grads_gram)

            # plot the heatmap of the gradient's gram matrix

            ax = fig.add_subplot(2, 4, i)
            cax = ax.matshow(grads_gram, cmap='viridis')
            fig.colorbar(cax)

            i += 1

            # plot labels
            ax.set_title(f'Gradient Gram Matrix at point {point_id}')
            ax.set_xticks(range(grads_gram.shape[0]))
            ax.set_yticks(range(grads_gram.shape[1]))
            ax.set_xticklabels(range(grads_gram.shape[0]))
            ax.set_yticklabels(range(grads_gram.shape[1]))

            ax = fig.add_subplot(2, 4, i)
            cov_matrix = grads_gram.T @ grads_gram
            (eigenvalues, eigenvectors) = np.linalg.eigh(cov_matrix)
            ax.plot(eigenvalues, 'o-')
            ax.set_title(f'Eigenvalues of the Gradient Cov Matrix at point {point_id}')

            i += 1

        return fig
