from typing import List

import torch

from src.datasets.pointcloud.base import BaseGeodesicsDataset


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
