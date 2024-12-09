from abc import ABC, abstractmethod

import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
from torch.utils.data import Dataset


class EuclideanEmbeddedSubmanifoldDataset(Dataset, ABC):
    def __init__(self, n_points=1000, noise_scale=None,
                 center=True, normalize=True, seed=42, **kwargs):
        super().__init__()
        assert n_points > 0, 'Number of points must be greater than 0'
        self.n_points = n_points
        seed_everything(seed)
        points = self.generate_points()

        if noise_scale is not None:
            assert isinstance(noise_scale, float) and noise_scale >= 0, 'Noise scale must be a non-negative float'
            points += noise_scale * torch.randn_like(points)

        if center:
            points -= points.mean(dim=0)

        if normalize:
            points /= points.norm(dim=1).max()

        data = Data(x=None,
                    edge_index=None,
                    edge_attr=None,
                    pos=points)

        self.data_list = [data]

    @abstractmethod
    def generate_points(self):
        # should return (n_points, D) tensor
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data_list[0]
