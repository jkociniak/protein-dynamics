import os

import torch
from torch.utils.data import Dataset
import networkx as nx
import mdtraj as md

from src.manifolds.pointcloud import PointCloudManifold


class BaseGeodesicsDataset(Dataset):
    def __init__(self, ca_pos, alpha=1.):
        self.ca_pos = ca_pos  # dimensions: (M, N, D)
        self.n_pointclouds, self.n_atoms, self.atom_dim = ca_pos.shape
        self.manifold = PointCloudManifold(self.atom_dim, self.n_atoms, base=ca_pos[0], alpha=alpha)
        self.proteins = self.align_conformations()

    def align_conformations(self):
        # constuct rotation matrix
        rot_xz = torch.zeros(3, 3)
        rot_xz[2, 0] = 1.
        rot_xz[1, 1] = 1.
        rot_xz[0, 2] = -1.
        self.manifold.base_point = torch.einsum("ba,ia->ib", rot_xz, self.manifold.base_point)

        rot_xy = torch.zeros(3, 3)
        theta = torch.tensor([- torch.pi * 1 / 3])
        rot_xy[0, 0] = torch.cos(theta)
        rot_xy[0, 1] = - torch.sin(theta)
        rot_xy[1, 0] = torch.sin(theta)
        rot_xy[1, 1] = torch.cos(theta)
        rot_xy[2, 2] = 1.
        self.manifold.base_point = torch.einsum("ba,ia->ib", rot_xy, self.manifold.base_point)

        # align all proteins with base
        proteins = self.manifold.align_mpoint(self.ca_pos[None], base=self.manifold.base_point).squeeze()
        return proteins

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TrajGeodesicsDataset(BaseGeodesicsDataset):
    def __init__(self, ca_pos, alpha=1., window_size=10, use_n_samples=None):
        super().__init__(ca_pos, alpha=alpha)
        assert window_size > 1, 'Window size must be greater than 1'
        self.window_size = window_size
        self.use_n_samples = use_n_samples
        if self.use_n_samples is not None:
            assert self.use_n_samples > 0, 'Number of samples to use must be greater than 0'

    def __len__(self):
        if self.use_n_samples is not None:
            return self.use_n_samples
        else:
            return self.proteins.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        x = self.proteins[idx:idx+self.window_size, :, :]
        return x


class DijkstraGeodesicsDataset(BaseGeodesicsDataset):
    def __init__(self, ca_pos, alpha=1.):
        super().__init__(ca_pos, alpha=alpha)

        # compute the pairwise distances between all pairs of conformations
        pw_dist, corrections = self.manifold.s_distance_decomposed(self.proteins.unsqueeze(0),
                                                                   self.proteins.unsqueeze(0))
        self.pw_dist = pw_dist.squeeze(0).numpy()  # dimensions: (M, M)
        self.corrections = corrections.squeeze(0).numpy()  # dimensions: (M, M)
        self.pairwise_distances = (pw_dist + self.manifold.alpha * corrections).squeeze(0).numpy()  # dimensions: (M, M)

        # compute the shortest paths between all pairs of conformations to approximate geodesics
        self.graph = nx.from_numpy_array(self.pairwise_distances)
        self.all_geodesics = self.compute_geodesics()

    def compute_geodesics(self):
        all_paths = nx.all_pairs_dijkstra(self.graph)
        all_geodesics = []
        visited_pairs = set()
        for src, (dists, paths) in all_paths:
            for tgt, path in paths.items():
                conds = [(tgt, src) not in visited_pairs,
                         (src, tgt) not in visited_pairs,
                         src != tgt]
                if all(conds):
                    # print(f'Path length: {len(path)}')
                    assert len(path) == 2
                    geodesic_xyz = ca_pos[path, :, :]
                    all_geodesics.append(geodesic_xyz)

                    if len(path) == 3:
                        print(f'Added geodesic {path} from {src} to {tgt}')
                    visited_pairs.add((src, tgt))

        return all_geodesics

    def __len__(self):
        return len(self.all_geodesics)

    def __getitem__(self, idx):
        x = self.all_geodesics[idx]
        return x


def prepare_data(struct, data_folder):
    if struct == 1:
        trajectory_path = os.path.join(data_folder, "4ake", "dims0001_fit-core.dcd")
        topology_path = os.path.join(data_folder, "4ake", "adk4ake.psf")
    elif struct == 2:
        trajectory_path = os.path.join(data_folder, "covid_spike", "MDtraj_sarscov_2.dcd")
        topology_path = os.path.join(data_folder, "covid_spike",
                                     "DESRES-Trajectory_sarscov2-12212688-5-2-no-water.pdb")
    else:
        raise ValueError(f'Unknown struct: {struct}')

    traj = md.load(trajectory_path, top=topology_path)
    print(f'Shape of full data: {traj.xyz.shape} (L x N x D)')

    indices = [m.index for m in traj.topology.atoms_by_name('CA')]
    ca_pos = 10 * torch.tensor(traj.xyz[:, indices, :])

    # if struct == 2:
    #     ca_pos = ca_pos[0:-1:2]

    num_timesteps, num_atoms, num_dims = ca_pos.shape
    print(f'Shape of CA data: {ca_pos.shape} (L x N x D)')
    print(f'Number of CA atoms: {num_atoms}')
    print(f'Number of timesteps: {num_timesteps}')
    print(f'CA atom data dimensionality: {num_dims} (should be 3 for 3D data)')

    return ca_pos


if __name__ == '__main__':
    seed = 42
    batch_size = 4

    struct = 1
    data_folder = os.path.join('../..', "data", "molecular_dynamics")
    results_folder = 'results'

    ca_pos = prepare_data(struct, data_folder)
    dataset = DijkstraGeodesicsDataset(ca_pos)