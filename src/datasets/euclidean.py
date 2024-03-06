from abc import ABC, abstractmethod
from typing import List

import torch
from torch.utils.data import Dataset

from .base import BaseGeodesicsDataset


class EuclideanEmbeddedSubmanifoldDataset(Dataset, ABC):
    def __init__(self, n_points=1000, center=False, scale=1.0, normalize=False):
        super().__init__()
        assert n_points > 0, 'Number of points must be greater than 0'
        self.n_points = n_points
        points = self.generate_points()
        self.points = points

        if center:
            self.points -= self.points.mean(dim=0)

        if normalize:
            self.points /= self.points.norm(dim=1, keepdim=True).max()

        assert isinstance(scale, (int, float)), 'Scale must be a number'
        self.scale = scale
        self.points *= scale

    @abstractmethod
    def generate_points(self):
        pass

    def __len__(self):
        return self.n_points

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
    def __init__(self, starting_angle, ending_angle, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        x = t * torch.cos(t)
        y = t * torch.sin(t)
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
