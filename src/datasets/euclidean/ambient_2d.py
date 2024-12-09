import torch
import numpy as np

from src.datasets.euclidean.base import EuclideanEmbeddedSubmanifoldDataset
from src.datasets.euclidean import DatasetRegistry

register = DatasetRegistry.register


@register()
class SineDataset(EuclideanEmbeddedSubmanifoldDataset):
    param_sets = {
        'Default sine': dict(n_points=100)
    }

    def generate_points(self):
        x = torch.arange(1, self.n_points + 1, dtype=torch.float32)
        x = torch.pi * x / (self.n_points + 1)
        y = torch.sin(x)
        return torch.stack([x, y], dim=1)


@register()
class NoisySineDataset(EuclideanEmbeddedSubmanifoldDataset):
    param_sets = {
        'Noisy sine (small noise)': dict(n_points=100, noise_scale=0.1),
        'Noisy sine (large noise)': dict(n_points=200, noise_scale=0.3)
    }

    def __init__(self, noise_scale=0.1, **kwargs):
        self.noise_scale = noise_scale
        super().__init__(**kwargs)

    def generate_points(self):
        x = torch.arange(1, self.n_points + 1, dtype=torch.float32)
        x = torch.pi * x / (self.n_points + 1)
        y = torch.sin(x) + self.noise_scale * torch.randn_like(x)
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


@register()
class Disk2DDataset(EuclideanEmbeddedSubmanifoldDataset):
    param_sets = {
        '2D Disk': dict(),
    }

    def __init__(self, radius=1, **kwargs):
        self.radius = radius
        super().__init__(**kwargs)

    def generate_points(self):
        r = np.sqrt(np.random.uniform(0, self.radius, self.n_points))
        theta = np.random.uniform(0, 2*np.pi, self.n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return torch.from_numpy(np.column_stack((x, y))).float()


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


class QuarterCircleDataset(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_points(self):
        r = np.sqrt(np.random.uniform(0, 1, self.n_points))
        theta = np.random.uniform(0, np.pi / 2, self.n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        out = torch.from_numpy(np.column_stack((x, y)))
        return out