import torch

from ..euclidean.base import EuclideanEmbeddedSubmanifoldDataset
from ..euclidean import DatasetRegistry

register = DatasetRegistry.register


@register()
class Debug2DTwoArcs(EuclideanEmbeddedSubmanifoldDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_points(self):
        # starting angle and ending angle are in radians
        t = torch.linspace(0, torch.pi, self.n_points)
        arc1 = torch.stack([torch.cos(t), torch.sin(t)], dim=1)  # arc from 1 to -1
        arc2 = -arc1.clone() - torch.tensor([1., 0.])  # arc from -2 to 0
        pc = torch.cat([arc1, arc2], dim=-1)  # a 4D dataset
        return pc