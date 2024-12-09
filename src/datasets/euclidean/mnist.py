import os

import torchvision
from sklearn.decomposition import PCA
import torch

from src.datasets.euclidean.base import EuclideanEmbeddedSubmanifoldDataset
from src.datasets.euclidean import DatasetRegistry


@DatasetRegistry.register()
class MNISTPCADataset(EuclideanEmbeddedSubmanifoldDataset):
    param_sets = {
        'MNIST PCA 2D, CLASSES [4, 9]': dict(root='data', n_components=2, keep_classes=[4, 9]),
        'MNIST PCA 2D, CLASSES [3, 9, 0]': dict(root='data', n_components=2, keep_classes=[3, 9, 0]),
        'MNIST PCA 5D, CLASSES [3, 9, 0]': dict(root='data', n_components=5, keep_classes=[3, 9, 0]),
    }

    def __init__(self, root, n_components, keep_classes=None, **kwargs):
        self.n_components = n_components
        self.base_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True)
        data = self.base_dataset.data.view(self.base_dataset.data.shape[0], -1).float()

        self.keep_classes = None
        if keep_classes is not None:
            keep_classes = [int(i) for i in keep_classes]
            for i in keep_classes:
                assert i in range(10), 'Invalid class index'
            keep_classes = list(sorted(set(keep_classes)))

            self.keep_classes = keep_classes

            # filter data by checking if label is in classes list
            filter = torch.isin(self.base_dataset.targets, torch.tensor(self.keep_classes))
            self.og_data = data[filter]
            self.og_labels = self.base_dataset.targets[filter]
            suffix = ''.join([str(i) for i in self.keep_classes])
            name = f'mnist_pca_{n_components}_classes_{suffix}.pt'
        else:
            self.og_data = data
            self.og_labels = self.base_dataset.targets
            name = f'mnist_pca_{n_components}.pt'

        self.mean = self.og_data.mean(dim=0)
        self.max_norm = self.og_data.norm(dim=1).max()
        self.pca = PCA(n_components=None)
        self.pca.fit((self.og_data - self.mean) / self.max_norm)

        self.indices = None
        self.labels = None

        self.path = os.path.join(root, name)

        super().__init__(**kwargs)

    def reconstruct_from_pca_interpolant(self, start, end, t, pred, reconstruct_from_pca_only=False):
        """
        Reconstructs the original data point from the PCA interpolant
        :param start: 784-dimensional point
        :param end: 784-dimensional point
        :param t: time on geodesic (between (0, 1))
        :param pred: K-dimensional point, where K is the number of principal components
        :return:
        """
        if self.n_components is None:
            return self.pca.inverse_transform(pred) * self.max_norm + self.mean
        else:
            s_pca = self.pca.transform(start[None])
            e_pca = self.pca.transform(end[None])
            if reconstruct_from_pca_only:
                pred_pca = torch.zeros(1, 784)
            else:
                pred_pca = s_pca * (1 - t) + e_pca * t
            pred_pca[:, :self.n_components] = pred
            pred_og = torch.from_numpy(self.pca.inverse_transform(pred_pca)) * self.max_norm + self.mean
            return pred_og

    def generate_points(self):
        # get random samples from each class
        samples_per_class = self.n_points // 10 if self.keep_classes is None else self.n_points // len(self.keep_classes)

        # indices in the original dataset (self.og_data)
        indices = []
        for i in range(10):
            if self.keep_classes is not None and i not in self.keep_classes:
                continue

            idx = torch.where(self.og_labels == i)[0]
            idx = idx[torch.randperm(len(idx))[:samples_per_class]]
            indices.extend(idx)

        idx = torch.randperm(len(indices))
        self.indices = torch.tensor(indices)[idx]

        pca_data = self.pca.transform((self.og_data - self.mean) / self.max_norm)[:, :self.n_components]
        pca_data = torch.from_numpy(pca_data).to(dtype=torch.float32)
        data = pca_data[self.indices]
        self.labels = self.og_labels[self.indices]

        if self.n_components is not None:
            assert len(data.shape) == 2 and data.shape[1] == self.n_components, 'Invalid data shape'
        else:
            assert len(data.shape) == 2 and data.shape[1] == 784, 'Invalid data shape'
        return data

