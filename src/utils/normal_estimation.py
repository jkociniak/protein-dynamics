import torch
import numpy as np
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors
from queue import Queue
from sklearn.decomposition import PCA

import torch_geometric.transforms as T
from src.utils.tensor import gradients


class SmoothDistanceFunction(nn.Module):
    def __init__(self, dataset, radius=0.1):
        super().__init__()
        self.dataset = dataset
        assert radius > 0
        self.radius = radius

    def forward(self, x):
        assert x.shape[-1] == self.dataset.shape[-1]
        bsh = x.shape[:-1]
        x = x.reshape(-1, self.dataset.shape[-1])
        diffs = x[:, None] - self.dataset[None]
        p2p_dists = torch.linalg.norm(diffs + 1e-12, dim=-1)
        out = -self.radius * torch.logsumexp(-p2p_dists / self.radius, dim=-1)
        out = out.reshape(*bsh, 1)
        return out

    def update_radius(self, new_radius):
        self.radius = new_radius


class NormalEstimator:
    @staticmethod
    def estimate_tubular_points(sdf, points,
                                num_noisy_samples, noise_scale,
                                optimization_steps, lr,
                                sdf_level_set, sdf_loss_weight):
        assert len(points.shape) == 2, 'Points must have shape (N, D)'

        # Generate noisy samples for all datapoints
        points_tensor = points.clone().detach()  # dimensions (N, D)

        noisy_batch_shape = (points.shape[0],
                             num_noisy_samples,
                             points.shape[-1])
        noisy_samples = points_tensor.unsqueeze(1) + noise_scale * torch.randn(
            *noisy_batch_shape)  # dimensions (N, num_noisy_samples, D)

        # Optimization process with loss tracking
        optimized_samples = noisy_samples.view(-1, points.shape[-1]).clone().detach().requires_grad_(True)  # dimensions (N*num_noisy_samples, D)
        optimizer = torch.optim.Adam([optimized_samples], lr=lr)
        sdf_losses = []
        l2_losses = []
        total_losses = []

        for step in range(optimization_steps):
            optimizer.zero_grad()

            sdf_values = sdf(optimized_samples)
            sdf_loss = nn.MSELoss()(sdf_values, torch.ones_like(sdf_values) * sdf_level_set)
            proj = points_tensor[:, None, :]
            proj = proj.expand(points.shape[0], num_noisy_samples, points.shape[-1])
            proj = proj.reshape(-1, points.shape[-1])
            l2_loss = nn.MSELoss()(optimized_samples, proj)

            total_loss = sdf_loss_weight * sdf_loss + (1 - sdf_loss_weight) * l2_loss

            total_loss.backward()
            optimizer.step()

            sdf_losses.append(sdf_loss.item())
            l2_losses.append(l2_loss.item())
            total_losses.append(total_loss.item())

        optimization_info = {
            'sdf_losses': sdf_losses,
            'l2_losses': l2_losses,
            'total_losses': total_losses,
        }

        optimized_samples = optimized_samples.reshape(*noisy_batch_shape).detach()

        return optimized_samples, optimization_info

    @staticmethod
    def estimate_boundary(sdf, tubular_samples, boundary_sdf_thr, boundary_neighbor_thr):
        assert len(tubular_samples.shape) == 3, 'Tubular samples must have shape (B, N, D)'
        z = sdf(tubular_samples)  # (B, N)
        cts = (z > boundary_sdf_thr).squeeze() # (B, N)
        cts = cts.sum(axis=-1) # (B)
        count_thr = boundary_neighbor_thr * tubular_samples.shape[-2]
        boundary = cts >= count_thr
        return boundary

    @staticmethod
    def extract_normal_vectors(sdf, tubular_samples):
        ts_flat = tubular_samples.view(-1, tubular_samples.shape[-1]).detach().clone().requires_grad_(True)
        sdf_values = sdf(ts_flat)
        grads = gradients(sdf_values, ts_flat)  # dimensions (B*N*num_noisy_samples, 1, 3)
        grads = grads.squeeze(-2)  # dimensions (B*N*num_noisy_samples, 3)
        grads = grads.reshape(*tubular_samples.shape)

        # normalize and randomize orientation
        grads = grads / torch.norm(grads, dim=-1, keepdim=True)
        signs = torch.ones_like(grads[..., 0])
        signs = torch.bernoulli(signs * 0.5) * 2 - 1
        grads = grads * signs[..., None]
        grads = grads.detach().numpy()

        return grads

    @staticmethod
    def compute_orthonormal_frames(normals):
        B, N, D = normals.shape
        assert N > D, 'Number of points must be greater than dimension of the ambient space'
        on_frames = np.zeros((B, D, D))
        for i, frame in enumerate(normals):
            pca = PCA(n_components=D)
            pca.fit(frame)
            on_frames[i] = pca.components_
        return on_frames

    @staticmethod
    def estimate_dimensionality(normals, threshold=0.85):
        B, N, D = normals.shape
        assert N > D, 'Number of points must be greater than dimension of the ambient space'
        on_frames = np.zeros((B, D, D))

        # Perform PCA on normal vectors
        singular_values = np.zeros((B, D))
        estimated_dimensions = np.zeros(B)
        for loc, normal_set in enumerate(normals):
            pca = PCA(n_components=D)
            pca.fit(normal_set)
            on_frames[loc] = pca.components_

            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            estimated_dim = np.argmax(cumulative_variance_ratio >= threshold) + 1

            estimated_dimensions[loc] = estimated_dim
            singular_values[loc] = pca.singular_values_

        return on_frames, singular_values, estimated_dimensions.astype(int)

    @staticmethod
    def compute_boundary_normals(points):
        pts = torch.from_numpy(points)
        sdf_cfg = {
            'radius': 10 ** (-1.6)
        }
        sdf = SmoothDistanceFunction(pts, **sdf_cfg)

        normal_estimation_cfg = dict(
            noise_scale=0.1,
            num_noisy_samples=20,
            optimization_steps=50,
            lr=0.1,
            sdf_level_set=0.,
            sdf_loss_weight=0.9,
        )

        tubular_samples, optimization_info = NormalEstimator.estimate_tubular_points(sdf, pts, **normal_estimation_cfg)

        tubular_samples = tubular_samples[0]

        boundary_cfg = {
            'boundary_sdf_thr': -0.007,
            'boundary_neighbor_thr': 0.9  # how many % of tubular samples must lie "outside" to categorize point a boundary
        }
        boundary_mask = NormalEstimator.estimate_boundary(sdf, tubular_samples, **boundary_cfg)

        b_points = points[boundary_mask]
        b_tubular_samples = tubular_samples[boundary_mask]
        b_normals = NormalEstimator.extract_normal_vectors(sdf, b_tubular_samples)

        return b_points, b_normals

    @staticmethod
    def propagate_orientations(b_points, b_frames, estimated_dimensions, propagate_cfg):
        new_frames = b_frames.copy()
        for dim in sorted(np.unique(estimated_dimensions)):
            if dim == b_points.shape[-1]:
                break
            idx = np.where(estimated_dimensions == dim)[0]
            print(f'Propagating frames for dimension {dim} with indices {idx}')
            if dim == 1:
                propagate_fun = NormalEstimator.propagate_normal_orientations_1codim
            else:
                propagate_fun = NormalEstimator.propagate_frame_orientations_ND
            new_frames[idx] = propagate_fun(b_points[idx], b_frames[idx], **propagate_cfg)

        return new_frames

    @staticmethod
    def propagate_frame_orientations_ND(points, frames, k=3, reference_index=0):
        points = np.asarray(points)
        frames = np.asarray(frames)

        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
        _, indices = nn.kneighbors(points)

        visited = np.zeros(len(frames), dtype=bool)
        aligned = np.zeros(len(frames), dtype=bool)

        queue = Queue()
        queue.put(reference_index)
        aligned[reference_index] = True

        while not queue.empty():
            current_index = queue.get()

            # ADD NEIGHBORS TO QUEUE
            for neighbor_index in indices[current_index]:
                if not visited[neighbor_index]:
                    queue.put(neighbor_index)
                    visited[neighbor_index] = True

            # PROCESS CURRENT FRAME
            current_frame = frames[current_index]  # (2, 3)

            # PROJECT NEIGHBORS ONTO CURRENT FRAME
            proj_matrix = current_frame.T @ current_frame

            valid_neighbors = [n_id for n_id in indices[current_index]
                               if n_id != current_index and aligned[n_id]]

            if len(valid_neighbors) > 0:
                neighbor_frames = frames[valid_neighbors]
                mean_neighbor_frame = np.mean(neighbor_frames, axis=0)
                projected_mean = np.einsum('ij,kj->ki', proj_matrix, mean_neighbor_frame)
                projected_mean /= np.linalg.norm(projected_mean, axis=1, keepdims=True)
                frames[current_index] = projected_mean

            aligned[current_index] = True

            # ADD THE NEXT CONNECTED COMPONENT WHEN NECESSARY
            if queue.empty() and not np.all(visited):
                next_index = np.where(~visited)[0][0]
                queue.put(next_index)

        return frames

    @staticmethod
    def propagate_normal_orientations_1codim(points, normals, k=3, reference_index=None):
        """
        Propagate normal orientations using a kNN graph.

        Parameters:
        points (numpy.ndarray): Point cloud data with shape (n_samples, 3)
        normals (numpy.ndarray): Normal vectors with shape (n_samples, frame_dim, 3)
        knn_indices: indices from kNN graph
        reference_index (int): Index of the reference normal. If None, the first normal is used.

        Returns:
        numpy.ndarray: Corrected normal vectors
        """
        # Ensure inputs are numpy arrays
        points = np.asarray(points)
        normals = np.asarray(normals)

        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
        _, indices = nn.kneighbors(points)

        # Initialize flags to keep track of processed normals
        processed = np.zeros(len(normals), dtype=bool)

        # Choose reference normal
        if reference_index is None:
            reference_index = 0

        # Initialize queue for breadth-first traversal
        queue = Queue()
        queue.put(reference_index)
        processed[reference_index] = True

        while not queue.empty():
            current_index = queue.get()
            current_normal = normals[current_index, 0]

            # Check neighbors
            for neighbor_index in indices[current_index]:
                if not processed[neighbor_index]:
                    neighbor_normal = normals[neighbor_index, 0]

                    # If normals point in opposite directions, flip the neighbor
                    if np.dot(current_normal, neighbor_normal) < 0:
                        normals[neighbor_index, 0] *= -1

                    queue.put(neighbor_index)
                    processed[neighbor_index] = True

            if queue.empty() and not all(processed):
                i = 0
                while processed[i]:
                    i += 1
                queue.put(i)
                processed[i] = True

        print(processed)
        return normals


class EstimateNormalsTransform(T.BaseTransform):
    def __init__(self, normal_estimation_cfg=None):
        if normal_estimation_cfg is None:
            print('Overriding normal_estimation_cfg to default')
            normal_estimation_cfg = dict(
                log_sdf_radius=0.1,
                noise_scale=0.1,
                num_noisy_samples=20,
                optimization_steps=50,
                lr=0.1,
                sdf_level_set=0.,
                sdf_loss_weight=0.9,
                boundary_cfg=dict(boundary_sdf_thr=-0.007,
                                  boundary_neighbor_thr=0.9)
                # how many % of tubular samples must lie "outside" to categorize point a boundary)
            )
        self.normal_estimation_cfg = normal_estimation_cfg

    def call_internal(self, data):
        normal_estimation_cfg = self.normal_estimation_cfg

        sdf_radius = 10 ** normal_estimation_cfg.pop('log_sdf_radius')
        boundary_cfg = normal_estimation_cfg.pop('boundary_cfg')

        points = data.pos  # (n_points, ambient_dim)
        torch.save(points, 'points_runner.pt')

        sdf = SmoothDistanceFunction(points, radius=sdf_radius)
        tubular_samples, optimization_info = NormalEstimator.estimate_tubular_points(sdf, points,
                                                                                     **normal_estimation_cfg)
        self.normals_optimization_info = optimization_info
        self.tubular_samples = tubular_samples

        #plot_point_cloud(tubular_samples.reshape(-1, 2), 'tubular_samples.png')

        boundary_mask = NormalEstimator.estimate_boundary(sdf, tubular_samples, **boundary_cfg)
        print(f'BOUNDARY MASK: {boundary_mask}')

        all_normals = np.zeros((points.shape[0], points.shape[1], points.shape[1]))
        estimated_dimensions = np.zeros(points.shape[0], dtype=int)

        b_normals = NormalEstimator.extract_normal_vectors(sdf, tubular_samples[boundary_mask])
        #plot_multi_quiver(points, b_normals * 5e-2, 'b_normals.png')
        on_frames, singular_values, b_dims = NormalEstimator.estimate_dimensionality(b_normals)

        all_normals[boundary_mask] = on_frames
        estimated_dimensions[boundary_mask] = b_dims

        print(f'Estimated dimensions: {estimated_dimensions}')
        #plot_multi_quiver(points, on_frames * 5e-2, 'on_frames.png')

        normal_basis = []
        valid_normals_mask = (estimated_dimensions > 0) & (estimated_dimensions < points.shape[-1])
        for i in range(points.shape[0]):
            if estimated_dimensions[i] == 0 or estimated_dimensions[i] == points.shape[-1]:
                normal_basis.append(torch.tensor([]))
                estimated_dimensions[i] = 0
            else:
                normal_basis.append(torch.from_numpy(all_normals[i, :estimated_dimensions[i]]).float())

        valid_normals_mask = torch.from_numpy(valid_normals_mask).bool()
        estimated_dimensions = torch.from_numpy(estimated_dimensions).int()

        return normal_basis, boundary_mask, valid_normals_mask, estimated_dimensions

    def __call__(self, data):
        normal_basis, boundary_mask, valid_normals_mask, normal_space_dims = self.call_internal(data)

        data.normal_basis = normal_basis
        data.boundary_mask = boundary_mask
        data.valid_normals_mask = valid_normals_mask
        data.normal_space_dims = normal_space_dims

        return data


def estimate_normals(sdf, points,
                     num_noisy_samples, noise_scale,
                     optimization_steps, lr,
                     sdf_level_set, sdf_loss_weight,
                     smooth=False):
    assert len(points.shape) in [2, 3]
    if len(points.shape) == 2:
        points = points.unsqueeze(0)

    # Generate noisy samples for all datapoints
    points_tensor = torch.tensor(points, dtype=torch.float32)  # dimensions (B, N, 3)

    noisy_batch_shape = (*points.shape[0:2],
                         num_noisy_samples,
                         points.shape[-1])
    noisy_samples = points_tensor.unsqueeze(2) + noise_scale * torch.randn(
        *noisy_batch_shape)  # dimensions (B, N, num_noisy_samples, 3)

    # Optimization process with loss tracking
    optimized_samples = noisy_samples.view(-1, points.shape[-1]).clone().detach().requires_grad_(
        True)  # dimensions (B*N*num_noisy_samples, 3)
    optimizer = torch.optim.Adam([optimized_samples], lr=lr)
    sdf_losses = []
    l2_losses = []
    total_losses = []

    for step in range(optimization_steps):
        optimizer.zero_grad()

        sdf_values = sdf(optimized_samples)
        sdf_loss = nn.MSELoss()(sdf_values, torch.ones_like(sdf_values) * sdf_level_set)

        proj = points_tensor[:, :, None, :]
        proj = proj.expand(*points.shape[:2], num_noisy_samples, points.shape[-1])
        proj = proj.reshape(-1, points.shape[-1])
        l2_loss = nn.MSELoss()(optimized_samples, proj)

        total_loss = sdf_loss_weight * sdf_loss + (1 - sdf_loss_weight) * l2_loss

        total_loss.backward()
        optimizer.step()

        sdf_losses.append(sdf_loss.item())
        l2_losses.append(l2_loss.item())
        total_losses.append(total_loss.item())

    # Compute gradients (normals) at optimized samples
    sdf_values = sdf(optimized_samples).reshape(-1, 1)
    grads = gradients(sdf_values, optimized_samples)  # dimensions (B*N*num_noisy_samples, 1, 3)
    grads = grads.squeeze(1)  # dimensions (B*N*num_noisy_samples, 3)
    normalized_gradients = grads / torch.norm(grads, dim=-1, keepdim=True)
    normalized_gradients = normalized_gradients.reshape(*noisy_batch_shape)

    smoothed_gradients = None
    knn = None
    if smooth:
        n_neighbors = 5
        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(
            optimized_samples.detach().numpy())
        distances, indices = knn.kneighbors(optimized_samples.detach().numpy())

        sigma = 10
        smoothed_gradients = torch.zeros_like(normalized_gradients)
        for i in range(len(optimized_samples)):
            weights = torch.exp(-torch.tensor(distances[i]) ** 2 / (2 * sigma ** 2))
            weights /= weights.sum()
            indices = indices - 1  # Adjust indices to start from 0
            smoothed_gradients[i] = (normalized_gradients.reshape(-1, 3)[indices[i]] * weights.unsqueeze(1)).sum(
                dim=0)
        smoothed_gradients = smoothed_gradients / torch.norm(smoothed_gradients, dim=1, keepdim=True)

    optimization_info = {
        'sdf_losses': sdf_losses,
        'l2_losses': l2_losses,
        'total_losses': total_losses,
        'sdf_loss_weight': sdf_loss_weight,
        'knn': knn
    }

    optimized_samples = optimized_samples.reshape(*noisy_batch_shape).detach()
    normalized_gradients = normalized_gradients.reshape(*noisy_batch_shape).detach()
    if smooth:
        smoothed_gradients = smoothed_gradients.reshape(*noisy_batch_shape).detach()

    return (optimized_samples, normalized_gradients, smoothed_gradients,
            optimization_info)