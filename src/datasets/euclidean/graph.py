import torch

from src.datasets.euclidean.base import EuclideanEmbeddedSubmanifoldDataset
from src.utils.tensor import rot_matrix_2d


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