import os
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch.utils.data import Dataset
import networkx as nx
import mdtraj as md
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
import trimesh

from src.utils.geometry import rot_matrix_2d
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
                    geodesic_xyz = self.proteins[path, :, :]
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


class PointCloudDataset(Dataset, ABC):
    def __init__(self, n_points=1000, seed=42, center=False, **kwargs):
        super().__init__()
        assert n_points > 0, 'Number of points must be greater than 0'
        self.n_points = n_points
        seed_everything(seed)
        points = self.generate_points()
        self.points = points  # dimension (n_points, n_vertices, 2)

        if center:
            self.points -= self.points.mean(dim=1, keepdim=True)  # center each snapshot

    @abstractmethod
    def generate_points(self):
        pass

    def __len__(self):
        return self.n_points

    def __getitem__(self, idx):
        return self.points[idx]


class VPC2DDataset(PointCloudDataset):
    def __init__(self, starting_angle, ending_angle, **kwargs):
        self.starting_angle = starting_angle
        self.ending_angle = ending_angle
        super().__init__(**kwargs)

    def generate_points(self):
        t = torch.linspace(self.starting_angle, self.ending_angle, self.n_points)
        # points will be of dimension (n_points, n_vertices, 2)
        #    / A
        #  B
        #    \ C
        # intially we fix B = (0, 0) and C = (1, 0) and only move A
        # we can then translate and rotate each snapshot because we will be using
        # an equivariant model

        As = torch.stack([torch.cos(t), torch.sin(t)], dim=1)  # (n_points, 2)
        Bs = torch.stack([torch.zeros_like(t), torch.zeros_like(t)], dim=1)  # (n_points, 2)
        Cs = torch.stack([torch.ones_like(t), torch.zeros_like(t)], dim=1)  # (n_points, 2)
        return torch.stack([As, Bs, Cs], dim=1)


class Bunny3DDataset(PointCloudDataset):
    def __init__(self, bunny_path, **kwargs):
        self.bunny_path = bunny_path
        self.normals = None
        super().__init__(**kwargs)

    def generate_points(self):
        mesh = trimesh.load('bunny.obj')
        return mesh.vertices


class GraphDataset(Dataset, ABC):
    def __init__(self, n_graphs=1000, seed=42, center_pos=False, **kwargs):
        super().__init__()
        assert n_graphs > 0, 'Number of points must be greater than 0'
        self.n_graphs = n_graphs
        seed_everything(seed)
        graphs = self.generate_graphs()
        self.graphs = graphs  # dimension (n_points, n_vertices, 2)

        if center_pos:
            for i, graph in enumerate(self.graphs):
                graph.pos -= graph.pos.mean(dim=0, keepdim=True)

    @abstractmethod
    def generate_graphs(self):
        pass

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):
        return self.graphs[idx]


class HingeAccordion2DDataset(GraphDataset):
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

    def generate_graphs(self):
        ts = []
        t_start = self.phi_start
        for i in range(self.phi_start.shape[0]):
            phi_s = self.phi_start[i]
            phi_e = self.phi_end[i]
            num = self.n_graphs // self.phi_start.shape[0]
            t = torch.linspace(phi_s, phi_e, num)[None]
            phis = t_start.repeat(num, 1)
            phis[:, i] = t
            ts.append(phis)
            t_start = t_start.clone()
            t_start[i] = phi_e

        t = torch.cat(ts, dim=0)

        # common settings
        n_nodes = self.a.shape[0] + 1
        x = torch.ones(n_nodes, 1)

        node_ids = list(range(n_nodes))
        edge_index = torch.tensor([node_ids[:-1], node_ids[1:]], dtype=torch.long)

        # generate graphs one by one
        graphs = []
        for phi in t:
            pos = self.generate_accordion_graph(self.a, phi)
            graph = Data(x=x, pos=pos, edge_index=edge_index)
            graphs.append(graph)

        return graphs

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


class Accordion2DDataset(GraphDataset):
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

    def generate_graphs(self):
        t = np.linspace(self.phi_start, self.phi_end, num=self.n_graphs)
        t = torch.from_numpy(t)

        # common settings
        n_nodes = self.a.shape[0] + 1
        x = torch.ones(n_nodes, 1)

        node_ids = list(range(n_nodes))
        edge_index = torch.tensor([node_ids[:-1], node_ids[1:]], dtype=torch.long)

        # generate graphs one by one
        graphs = []
        for i in range(self.n_graphs):
            phi = t[i]
            pos = self.generate_accordion_graph(self.a, phi)
            graph = Data(x=x, pos=pos, edge_index=edge_index)
            graphs.append(graph)

        return graphs

    @staticmethod
    def generate_accordion_graph(a, phi):
        assert phi.shape[0] == a.shape[0] - 1
        xs = [torch.zeros(2), torch.tensor([a[0], 0])]  # first 2 vertices

        for i, (aa, pp) in enumerate(zip(a[1:], phi)):
            pp_true = torch.pi * (1 - pp) if i % 2 == 0 else np.pi * (1 + pp)
            rot = rot_matrix_2d(pp_true)

            src = xs[-1]
            dir = src - xs[-2]
            dir /= torch.linalg.norm(dir)
            new_x = src + aa * rot @ dir
            xs.append(new_x)

        return torch.stack(xs)