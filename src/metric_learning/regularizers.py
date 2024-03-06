from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.func import vmap


class MetricTensorRegularizer(nn.Module, ABC):
    def __init__(self, correction_encoder, pos_hess_weight=0., neg_hess_weight=10., min_pos_hess_norm=1., max_neg_hess_norm=1e-1,
                 smoothness_weight=0.):
        super().__init__()
        self.correction_encoder = correction_encoder
        #self.pos_samples_cache = []
        #self.neg_samples_cache = []

        assert isinstance(pos_hess_weight, float) and pos_hess_weight >= 0.
        self.pos_hess_weight = pos_hess_weight

        assert isinstance(neg_hess_weight, float) and neg_hess_weight >= 0.
        self.neg_hess_weight = neg_hess_weight

        assert isinstance(min_pos_hess_norm, float) and min_pos_hess_norm > 0.
        self.min_pos_hess_norm = min_pos_hess_norm

        assert isinstance(max_neg_hess_norm, float) and max_neg_hess_norm > 0.
        self.max_neg_hess_norm = max_neg_hess_norm

        assert isinstance(smoothness_weight, float) and smoothness_weight >= 0.
        self.smoothness_weight = smoothness_weight

    def forward(self, x):
        if self.smoothness_weight > 0:
            return self.forward_with_smoothing(x)
        else:
            return self.forward_no_smoothing(x)

    def forward_no_smoothing(self, x):
        positives = self.generate_positives(x)
        # self.pos_samples_cache.extend(positives)
        if len(positives) == 0:
            pos_hess_norm = torch.tensor(0., dtype=torch.float32)
        else:
            pos_hess_norm, _ = self.correction_encoder.hessian_fro_norm(positives)
        pos_hess_norm = torch.clamp(pos_hess_norm, min=self.min_pos_hess_norm)
        pos_mean_hess_loss = self.pos_hess_weight * pos_hess_norm

        negatives = self.generate_negatives(x)
        # self.neg_samples_cache.extend(negatives)
        if len(negatives) == 0:
            neg_hess_norm = torch.tensor(0., dtype=torch.float32)
        else:
            neg_hess_norm, _ = self.correction_encoder.hessian_fro_norm(negatives)

        neg_hess_norm = torch.clamp(neg_hess_norm, max=self.max_neg_hess_norm)
        neg_mean_hess_loss = - self.neg_hess_weight * neg_hess_norm

        out = {'pos_mean_hess_loss': pos_mean_hess_loss, 'neg_mean_hess_loss': neg_mean_hess_loss,
               'pos_hess_norm': pos_hess_norm, 'neg_hess_norm': neg_hess_norm}
        return out

    def forward_with_smoothing(self, x):
        assert self.smoothness_weight > 0
        assert self.pos_hess_weight == 0, "Smoothing is not compatible with positive hessian regularization"
        pos_hess_norm = torch.tensor(0., dtype=torch.float32)
        pos_mean_hess_loss = torch.tensor(0., dtype=torch.float32)

        negatives = x
        # self.neg_samples_cache.extend(negatives)
        if len(negatives) == 0:
            neg_hess_norm = torch.tensor(0., dtype=torch.float32)
            mean_smoothness_error = torch.tensor(0., dtype=torch.float32)
        else:
            neg_hess_norm, _, mean_smoothness_error = self.correction_encoder.hessian_fro_norm(negatives, smoothness_loss=True)

        neg_hess_norm = torch.clamp(neg_hess_norm, max=self.max_neg_hess_norm)
        neg_mean_hess_loss = - self.neg_hess_weight * neg_hess_norm

        mean_smoothness_loss = self.smoothness_weight * mean_smoothness_error

        out = {'pos_mean_hess_loss': pos_mean_hess_loss, 'pos_hess_norm': neg_mean_hess_loss,
               'neg_mean_hess_loss': neg_mean_hess_loss, 'neg_hess_norm': neg_hess_norm,
               'mean_smoothness_error': mean_smoothness_error, 'mean_smoothness_loss': mean_smoothness_loss}
        return out

    def clear_caches(self):
        #self.pos_samples_cache = []
        #self.neg_samples_cache = []
        pass

    @abstractmethod
    def generate_positives(self, x):
        pass

    @abstractmethod
    def generate_negatives(self, x):
        pass


class EnvelopeRegularizer(MetricTensorRegularizer):
    def __init__(self, *args, neg_samples_eps=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(neg_samples_eps, float) and neg_samples_eps > 0.
        self.neg_samples_eps = neg_samples_eps

    def generate_negatives(self, x):
        return x

    def generate_positives(self, x):
        assert len(x.shape) == 3  # dimensions: (B, L, D)
        assert x.shape[2] == 2  # implementation only for 2D data

        # 1. compute the start and end points of each segment
        x1 = x[:, :-1, :]
        x2 = x[:, 1:, :]

        # 2. compute the midpoints of each segment
        def perpendicular_segment(segment, eps=0.05):
            # segment is represented by a tuple of two 2D points
            # so dims should be (2, 2)
            # Calculate midpoint of the segment
            start, end = segment
            midpoint = (start + end) / 2
            d = end - start
            perp = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32).to(d.device) @ d
            perp /= torch.linalg.norm(perp)
            perp *= eps
            perp_segment = torch.stack([midpoint - perp, midpoint + perp], dim=0)
            return perp_segment

        perp_fn = vmap(vmap(perpendicular_segment, in_dims=(0, None)), in_dims=(0, None))
        input = torch.stack([x1, x2], dim=2)

        perp_segments = perp_fn(input, self.neg_samples_eps)  # dimensions: (B, L-1, 2, 2)
        perp_segments = perp_segments.view(-1, 2 * perp_segments.shape[1], 2)  # dimensions: (B, 2 * (L-1), 2)

        return perp_segments


class CurveNegativeRegularizer(MetricTensorRegularizer):
    def generate_positives(self, x):
        return torch.tensor([], dtype=torch.float32)

    def generate_negatives(self, x):
        return x


class FakeRegularizer(MetricTensorRegularizer):
    def generate_positives(self, x):
        return torch.tensor([], dtype=torch.float32)

    def generate_negatives(self, x):
        return torch.tensor([], dtype=torch.float32)
