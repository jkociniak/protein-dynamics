from typing import List
from abc import ABC, abstractmethod

import torch
import networkx as nx


class BaseGeodesicsDataset(torch.utils.data.Dataset, ABC):
    def __init__(self):
        super().__init__()
        self.geodesics = self.generate_geodesics()

    @abstractmethod
    def generate_geodesics(self) -> List[torch.Tensor]:
        pass

    def __len__(self):
        return len(self.geodesics)

    def __getitem__(self, idx):
        x = self.geodesics[idx]
        return x


class DijkstraGeodesicsDataset(BaseGeodesicsDataset):
    def __init__(self, base_dataset, base_manifold, path_length=15):
        self.base_dataset = base_dataset
        self.base_manifold = base_manifold

        assert path_length > 1, 'Max path length must be greater than 1'
        self.path_length = path_length
        self.dists = None
        super().__init__()

    def generate_geodesics(self):
        # compute the pairwise distances between all pairs of conformations
        pts = self.base_dataset.points[None]  # we add single batch dim to make it [1, N, D]
        dists = self.base_manifold.pairwise_distance(pts, pts)  # dimensions: (1, N, N)
        self.dists = dists.squeeze(0).numpy()

        # compute the shortest paths between all pairs of conformations to approximate geodesics
        graph = nx.from_numpy_array(self.dists)

        all_paths = nx.all_pairs_dijkstra(graph)
        geodesics = []
        visited_pairs = set()
        for src, (dists, paths) in all_paths:
            for tgt, path in paths.items():
                conds = [(tgt, src) not in visited_pairs,
                         (src, tgt) not in visited_pairs,
                         src != tgt]
                if not all(conds):
                    continue

                if len(path) < self.path_length:
                    continue

                path = path[-self.path_length:]  # we take last path_length elements
                geodesic = self.base_dataset.points[path]
                geodesics.append(geodesic)
                visited_pairs.add((src, tgt))

        return geodesics