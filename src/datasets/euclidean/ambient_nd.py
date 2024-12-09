import torch
import numpy as np

from scipy.stats import special_ortho_group

from src.datasets.euclidean.base import EuclideanEmbeddedSubmanifoldDataset


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