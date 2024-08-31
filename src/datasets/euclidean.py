from abc import ABC, abstractmethod
from typing import List
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group
from pytorch_lightning import seed_everything

from src.utils.geometry import rot_matrix_2d

from .base import BaseGeodesicsDataset


class EuclideanEmbeddedSubmanifoldDataset(Dataset, ABC):
    def __init__(self, n_points=1000, center=True, normalize=True, seed=42, **kwargs):
        super().__init__()
        assert n_points > 0, 'Number of points must be greater than 0'
        self.n_points = n_points
        seed_everything(seed)
        points = self.generate_points()
        self.points = points

        if center:
            self.points -= self.points.mean(dim=0)

        if normalize:
            self.points /= self.points.norm(dim=1).max()

    @abstractmethod
    def generate_points(self):
        pass

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx]


class SineDataset(EuclideanEmbeddedSubmanifoldDataset):
    def generate_points(self):
        x = torch.arange(1, self.n_points + 1, dtype=torch.float32)
        x = torch.pi * x / (self.n_points + 1)
        y = torch.sin(x)
        return torch.stack([x, y], dim=1)


class CircleDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle, ending_angle, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        # starting angle and ending angle are in radians
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        return torch.stack([torch.cos(t), torch.sin(t)], dim=1)


class ThirdDegreePolynomialDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, a=-1., b=0., c=1., **kwargs):
        self.a = a
        self.b = b
        self.c = c
        super().__init__(**kwargs)

    def generate_points(self):
        x = torch.linspace(-1, 1, self.n_points)
        y = (x - self.a) * (x - self.b) * (x - self.c)
        return torch.stack([x, y], dim=1)


class SpiralDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle, ending_angle, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        x = t * torch.cos(t)
        y = t * torch.sin(t)
        return torch.stack([x, y], dim=1)


class Helix3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle=0., ending_angle=torch.pi, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        x = torch.cos(t)
        y = torch.sin(t)
        z = t
        return torch.stack([x, y, z], dim=1)


class OscillatingHelix3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle=0., ending_angle=torch.pi, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        r = 1 + 0.1 * torch.sin(10 * t)
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        z = t
        return torch.stack([x, y, z], dim=1)


class Sphere3DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, polar_angle_range, azimuth_range, **kwargs):
        assert len(polar_angle_range) == 2, 'Polar angle range must be a 2-tuple'
        assert 0 <= polar_angle_range[0] < torch.pi, 'Invalid polar angle starting value'
        assert 0 <= polar_angle_range[1] < torch.pi, 'Invalid polar angle ending value'
        assert polar_angle_range[0] < polar_angle_range[1], 'Invalid polar angle range'

        assert len(azimuth_range) == 2, 'Azimuth range must be a 2-tuple'
        assert -torch.pi <= azimuth_range[0] < torch.pi, 'Invalid azimuth starting value'
        assert -torch.pi <= azimuth_range[1] < torch.pi, 'Invalid azimuth ending value'
        assert azimuth_range[0] < azimuth_range[1], 'Invalid azimuth range'

        self.polar_angle_range = polar_angle_range
        self.azimuth_range = azimuth_range
        super().__init__(**kwargs)

    def generate_points(self):
        all_points = []
        while len(all_points) < self.n_points:
            # 1. sample a standard gaussian distribution for each coordinate
            points = torch.randn(self.n_points, 3)
            # 2. normalize the points to lie on the unit sphere
            points /= points.norm(dim=1, keepdim=True)
            # 3. filter the points that are within the specified polar angle and azimuth range
            polar_angle = torch.acos(points[:, 2])
            azimuth = torch.atan2(points[:, 1], points[:, 0])
            mask = (self.polar_angle_range[0] <= polar_angle) & (polar_angle <= self.polar_angle_range[1]) & \
                   (self.azimuth_range[0] <= azimuth) & (azimuth <= self.azimuth_range[1])

            points = points[mask]
            all_points.append(points)

        points = torch.cat(all_points, dim=0)
        if points.shape[0] > self.n_points:
            points = points[:self.n_points, :]

        return points


class Sphere3DGeodesicsDataset(BaseGeodesicsDataset):
    def __init__(self, base_dataset, target, min_segment_length=5e-2, path_length=15):
        self.base_dataset = base_dataset
        assert isinstance(target, torch.Tensor), 'Target must be a tensor'
        assert target.shape == (3,) and torch.linalg.norm(target) == 1, 'Target must be 3D unit vector'
        self.target = target

        assert isinstance(min_segment_length, float) and min_segment_length > 0, 'Minimum segment length must be a float greater than 0'
        self.min_segment_length = min_segment_length

        assert path_length > 1, 'Max path length must be greater than 1'
        self.path_length = path_length
        super().__init__()

    def generate_geodesics(self) -> List[torch.Tensor]:
        # we treat points from the base dataset as the starting points
        # and the target as the ending point
        # we generate interpolation with a minimum segment length
        geodesics = []

        for starting_point in self.base_dataset.points:
            # we use the great-circle distance formula
            # https://en.wikipedia.org/wiki/Great-circle_distance
            # to interpolate the points
            angle = torch.acos(starting_point.dot(self.target))
            n_points = max(int(angle / self.min_segment_length), 2)
            t = torch.linspace(0, 1, n_points).reshape(-1, 1)

            if t.shape[0] < self.path_length:
                continue
            t = t[-self.path_length:, :]
            u = starting_point.reshape(1, -1)
            v = self.target.reshape(1, -1)
            interpolated_points = (torch.sin((1 - t) * angle) * u + torch.sin(t * angle) * v) / torch.sin(angle)
            geodesics.append(interpolated_points)

        return geodesics


class Debug4DCurve(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle=0., ending_angle=torch.pi, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        x = torch.cos(t)
        y = torch.sin(t)
        z = t
        w = t ** 2
        return torch.stack([x, y, z, w], dim=1)


class Debug8DCurve(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, starting_angle=0., ending_angle=torch.pi, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        x = torch.cos(t)
        y = torch.sin(t)
        z = t
        w = t ** 2
        a = t ** 3
        b = t ** 4
        c = t ** 5
        d = torch.exp(t)
        return torch.stack([x, y, z, w, a, b, c, d], dim=1)


class DebugNDCircle(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, n_dimensions=2, **kwargs):
        self.n_dimensions = n_dimensions
        super().__init__(**kwargs)

    def generate_points(self):
        start = torch.tensor([1, 0])
        end = torch.tensor([0, 1])

        angle = torch.tensor(torch.pi / 2)
        t = torch.linspace(0, 1, self.n_points).reshape(-1, 1)

        u = start.reshape(1, -1)
        v = end.reshape(1, -1)
        points = (torch.sin((1 - t) * angle) * u + torch.sin(t * angle) * v) / torch.sin(angle)
        new_points = torch.zeros((self.n_points, self.n_dimensions))
        new_points[:, :2] = points
        rv = special_ortho_group(self.n_dimensions, seed=42)
        rot = rv.rvs()
        new_points = np.einsum('ij,Ni->Nj', rot, new_points)
        return torch.from_numpy(new_points).float()


class HingeAccordion2DDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, a, phi_start, phi_end, **kwargs):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(phi_start, torch.Tensor):
            phi_start = torch.tensor(phi_start)

        if not isinstance(phi_end, torch.Tensor):
            phi_end = torch.tensor(phi_end)

        assert phi_start.shape == phi_end.shape
        assert len(phi_start.shape) == 1

        assert len(a.shape) == 1
        assert phi_start.shape[0] == a.shape[0] - 1

        self.a = a
        self.phi_start = phi_start
        self.phi_end = phi_end

        super().__init__(**kwargs)

    def generate_points(self):
        ts = []
        t_start = self.phi_start
        for i in range(self.phi_start.shape[0]):
            phi_s = self.phi_start[i]
            phi_e = self.phi_end[i]
            num = self.n_points // self.phi_start.shape[0]
            t = torch.linspace(phi_s, phi_e, num)[None]
            phis = t_start.repeat(num, 1)
            phis[:, i] = t
            ts.append(phis)
            t_start = t_start.clone()
            t_start[i] = phi_e

        t = torch.cat(ts, dim=0)

        # # common settings
        # n_nodes = self.a.shape[0] + 1
        # x = torch.ones(n_nodes, 1)
        #
        # node_ids = list(range(n_nodes))
        # edge_index = torch.tensor([node_ids[:-1], node_ids[1:]], dtype=torch.long)

        # generate graphs one by one
        pcs = []
        for phi in t:
            pos = self.generate_accordion_graph(self.a, phi)
            pcs.append(pos.flatten())

        pcs = torch.stack(pcs, dim=0)
        return pcs

    @staticmethod
    def generate_accordion_graph(a, phi):
        assert phi.shape[0] == a.shape[0] - 1
        xs = [torch.zeros(2), torch.tensor([a[0], 0])]  # first 2 vertices

        for i, (aa, pp) in enumerate(zip(a[1:], phi)):
            pp_true = torch.pi * (1 - pp) if i % 2 == 0 else torch.pi * (1 + pp)
            rot = rot_matrix_2d(pp_true)

            src = xs[-1]
            dir = src - xs[-2]
            dir /= torch.linalg.norm(dir)
            new_x = src + aa * rot @ dir
            xs.append(new_x)

        return torch.stack(xs)


class MNISTPCADataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, root, n_components, keep_classes=None, **kwargs):
        self.n_components = n_components
        self.base_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True)
        data = self.base_dataset.data.view(self.base_dataset.data.shape[0], -1).float()

        self.keep_classes = None
        if keep_classes is not None:
            keep_classes = [int(i) for i in keep_classes]
            for i in keep_classes:
                assert i in range(10), 'Invalid class index'
            keep_classes = list(sorted(set(keep_classes)))

            self.keep_classes = keep_classes

            # filter data by checking if label is in classes list
            filter = torch.isin(self.base_dataset.targets, torch.tensor(self.keep_classes))
            self.og_data = data[filter]
            self.og_labels = self.base_dataset.targets[filter]
            suffix = ''.join([str(i) for i in self.keep_classes])
            name = f'mnist_pca_{n_components}_classes_{suffix}.pt'
        else:
            self.og_data = data
            self.og_labels = self.base_dataset.targets
            name = f'mnist_pca_{n_components}.pt'

        self.mean = self.og_data.mean(dim=0)
        self.max_norm = self.og_data.norm(dim=1).max()
        self.pca = PCA(n_components=n_components)
        self.pca.fit((self.og_data - self.mean) / self.max_norm)

        self.indices = None
        self.labels = None

        self.path = os.path.join(root, name)

        super().__init__(**kwargs)

    def reconstruct_from_pca_interpolant(self, start, end, t, pred):
        if self.n_components is None:
            return self.pca.inverse_transform(pred) * self.max_norm + self.mean
        else:
            s_pca = self.pca.transform(start[None])
            e_pca = self.pca.transform(end[None])
            pred_pca = s_pca * (1 - t) + e_pca * t
            pred_pca[:, self.n_components:] = pred
            pred_og = self.pca.inverse_transform(pred_pca) * self.max_norm + self.mean
            return pred_og

    def generate_points(self):
        # get random samples from each class
        samples_per_class = self.n_points // 10 if self.keep_classes is None else self.n_points // len(self.keep_classes)

        # indices in the original dataset (self.og_data)
        indices = []
        for i in range(10):
            if self.keep_classes is not None and i not in self.keep_classes:
                continue

            idx = torch.where(self.og_labels == i)[0]
            idx = idx[torch.randperm(len(idx))[:samples_per_class]]
            indices.extend(idx)

        idx = torch.randperm(len(indices))
        self.indices = torch.tensor(indices)[idx]

        pca_data = self.pca.transform((self.og_data - self.mean) / self.max_norm)
        pca_data = torch.from_numpy(pca_data).to(dtype=torch.float32)
        data = pca_data[self.indices]
        self.labels = self.og_labels[self.indices]

        if self.n_components is not None:
            assert len(data.shape) == 2 and data.shape[1] == self.n_components, 'Invalid data shape'
        else:
            assert len(data.shape) == 2 and data.shape[1] == 784, 'Invalid data shape'
        return data
