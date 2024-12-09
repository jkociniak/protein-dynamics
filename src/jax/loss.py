import equinox as eqx
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from functools import partial


@dataclass
class Loss(eqx.Module):
    weights: Dict[str, float]
    params: Dict[str, float]
    normal_pretraining_batches: int = 0
    last_negatives: Optional[jnp.ndarray] = None

    def __init__(self, weights: Dict[str, float], params: Dict[str, float],
                 normal_pretraining_batches: int = 0):
        self.weights = weights
        self.params = params
        assert normal_pretraining_batches >= 0, 'Normal pretraining epochs must be non-negative'
        self.normal_pretraining_batches = normal_pretraining_batches
        self.last_negatives = None

        # Validate required weights
        assert self.weights['manifold_norm'] > 0, 'Manifold loss must be enabled'
        assert self.weights['non_manifold_norm'] > 0, 'Non-manifold loss must be enabled'

    def __call__(self, manifold: eqx.Module, batch: Dict) -> Dict[str, jnp.ndarray]:
        # Validate batch
        assert batch['batch'].max() == 0, "We expect the batch to be a single point cloud"
        pts = batch['pos']
        assert len(pts.shape) == 2, "Coordinates batch must be 2D - (num_points, ambient_dim)"
        assert pts.shape[-1] == manifold.base_manifold.d, "Batch must have same dimension as manifold"
        assert manifold.correction_decoder is None, 'Decoder loss is not implemented yet'

        if 'normal_space_dims' in batch and batch['normal_space_dims'] is not None:
            losses = self.forward_with_normals(
                manifold, pts,
                batch['normal_space_dims'],
                normal_basis=batch.get('normal_basis', None)
            )
        else:
            losses = self.forward_without_normals(manifold, pts)

        return losses

    def forward_without_normals(self, manifold: eqx.Module,
                                pts: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        losses = {}

        # Get encoder output and gradients
        out, coords = manifold.correction_encoder(pts)
        grads = jax.grad(lambda x: manifold.correction_encoder(x)[0])(coords)

        # Manifold norm loss
        losses['manifold_norm'] = self.zero_norm_loss(out)
        neg_directions = jnp.array(grads)  # Create copy of gradients

        # Generate and process negative samples
        nm_pts = self.generate_negatives(pts, neg_directions)
        nm_pts = jnp.reshape(nm_pts, (-1, nm_pts.shape[-1]))
        self.last_negatives = nm_pts

        nm_out, nm_coords = manifold.correction_encoder(nm_pts)
        losses['non_manifold_norm'] = self.high_norm_loss(
            nm_out, self.params['non_manifold_alpha']
        )

        # Eikonal losses if enabled
        if self.weights['manifold_eikonal'] > 0:
            losses['manifold_eikonal'] = self.eikonal_loss(grads)

        if self.weights['non_manifold_eikonal'] > 0:
            nm_grads = jax.grad(lambda x: manifold.correction_encoder(x)[0])(nm_coords)
            losses['non_manifold_eikonal'] = self.eikonal_loss(nm_grads)

        # Orthogonal loss if enabled
        if self.weights['orthogonal'] > 0:
            losses['orthogonal'] = self.orthogonal_loss(grads)

        return losses

    def forward_with_normals(self, manifold: eqx.Module, pts: jnp.ndarray,
                             normal_space_dims: jnp.ndarray,
                             normal_basis: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        losses = {}

        # Get encoder output and gradients
        out, coords = manifold.correction_encoder(pts)
        grads = jax.grad(lambda x: manifold.correction_encoder(x)[0])(coords)

        # Manifold norm loss
        losses['manifold_norm'] = self.zero_norm_loss(out)

        # Process negative directions
        neg_directions = jnp.array(grads)

        # Zero out tangent dimensions
        for i, nsd in enumerate(normal_space_dims):
            assert 0 <= nsd <= manifold.correction_encoder.out_features
            if nsd < manifold.correction_encoder.out_features:
                neg_directions = neg_directions.at[i, nsd:].set(0)

        # Handle normal basis if provided
        if normal_basis is not None:
            for i, nb in enumerate(normal_basis):
                if nb.size == 0:
                    assert normal_space_dims[i] == 0
                    continue

                assert len(nb.shape) == 2
                assert nb.shape[-1] == manifold.base_manifold.d
                assert nb.shape[0] <= manifold.base_manifold.d

                gram_mat = nb @ nb.T
                assert jnp.max(jnp.abs(gram_mat - jnp.eye(gram_mat.shape[-1]))) < 1e-6
                neg_directions = neg_directions.at[i, :gram_mat.shape[0]].set(nb)

        # Generate negative samples
        mask = normal_space_dims > 0
        nm_pts = self.generate_negatives(pts[mask], neg_directions[mask])
        nm_pts = jnp.reshape(nm_pts, (-1, nm_pts.shape[-1]))
        self.last_negatives = nm_pts

        # Process negative samples
        nm_out, nm_coords = manifold.correction_encoder(nm_pts)
        losses['non_manifold_norm'] = self.high_norm_loss(
            nm_out, self.params['non_manifold_alpha']
        )

        # Get gradients for negative samples
        nm_grads = jax.grad(lambda x: manifold.correction_encoder(x)[0])(nm_coords)
        nm_grads = jnp.reshape(nm_grads, (mask.sum(), -1, nm_grads.shape[-2], nm_grads.shape[-1]))

        # Initialize loss components
        max_nsd = jnp.max(normal_space_dims)
        assert max_nsd < pts.shape[-1]

        loss_eik = 0.
        loss_orthogonal = 0.
        loss_zero_norm = 0.
        nm_loss_eik = 0.
        nm_loss_orthogonal = 0.
        nm_loss_zero_norm = 0.
        loss_nsp = 0.

        # Compute losses for different normal space dimensions
        for d in range(max_nsd + 1):
            d_mask = normal_space_dims == d

            tangential_component = grads[d_mask, d:]
            normal_component = grads[d_mask, :d]

            if d < max_nsd:
                loss_zero_norm += self.zero_norm_loss(tangential_component)

            if d > 0:
                nm_d_mask = normal_space_dims[mask] == d

                nm_tangential_component = jnp.reshape(nm_grads[nm_d_mask, :, d:], (-1, nm_grads.shape[-1]))
                nm_normal_component = jnp.reshape(nm_grads[nm_d_mask, :, :d], (-1, d))

                loss_eik += self.eikonal_loss(normal_component)
                loss_orthogonal += self.orthogonal_loss(normal_component)

                if d < max_nsd:
                    nm_loss_zero_norm += self.zero_norm_loss(nm_tangential_component)
                nm_loss_eik += self.eikonal_loss(nm_normal_component)
                nm_loss_orthogonal += self.orthogonal_loss(nm_normal_component)

                if self.normal_pretraining_batches > 0 and normal_basis is not None:
                    normal_bases = [nb for i, nb in enumerate(normal_basis) if d_mask[i]]
                    if normal_bases:
                        normal_bases = jnp.stack(normal_bases)
                        loss_nsp += self.normal_subspace_loss(grads[d_mask, :d], normal_bases)

        # Combine losses
        losses['manifold_eikonal'] = loss_eik + loss_zero_norm
        losses['non_manifold_eikonal'] = nm_loss_eik + nm_loss_zero_norm
        losses['orthogonal'] = loss_orthogonal + nm_loss_orthogonal
        losses['manifold_normal_subspace'] = loss_nsp

        self.normal_pretraining_batches = max(0, self.normal_pretraining_batches - 1)
        return losses

    def generate_negatives(self, x: jnp.ndarray, normal_basis: jnp.ndarray,
                           n_samples: int = 5) -> jnp.ndarray:
        key = jax.random.PRNGKey(0)  # You might want to make this configurable
        weights = jax.random.normal(
            key,
            shape=(normal_basis.shape[0], n_samples, normal_basis.shape[1])
        )

        normals = jnp.einsum('Mnk,Mkd->Mnd', weights, normal_basis)
        normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)

        return x[:, None, :] + self.params['non_manifold_eps'] * normals

    @staticmethod
    def zero_norm_loss(pts: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.linalg.norm(pts, axis=-1, ord=1))

    @staticmethod
    def high_norm_loss(pts: jnp.ndarray, alpha: float) -> jnp.ndarray:
        norms = jnp.linalg.norm(pts, axis=-1, ord=1)
        return jnp.mean(jnp.exp(-alpha * norms))

    @staticmethod
    def eikonal_loss(grads: jnp.ndarray) -> jnp.ndarray:
        grad_norms = jnp.linalg.norm(grads, axis=-1, ord=2)
        return jnp.mean(jnp.abs(grad_norms - 1))

    @staticmethod
    def orthogonal_loss(grads: jnp.ndarray) -> jnp.ndarray:
        unit_grads = grads / jnp.linalg.norm(grads, axis=-1, keepdims=True)
        grad_inner = jnp.einsum('Nij,Nkj->Nik', unit_grads, unit_grads)

        # Zero out diagonal
        mask = ~jnp.eye(grad_inner.shape[-1], dtype=bool)
        grad_inner = grad_inner * mask

        return jnp.mean(jnp.abs(grad_inner))

    @staticmethod
    def normal_loss(grads: jnp.ndarray, gt_normals: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.linalg.norm(grads - gt_normals, axis=-1, ord=2))

    @staticmethod
    def normal_subspace_loss(grads: jnp.ndarray, gt_normals: jnp.ndarray) -> jnp.ndarray:
        assert grads.shape[-1] == gt_normals.shape[-1]
        assert grads.shape[0] == gt_normals.shape[0]

        gt_normals_dotprod = jnp.einsum('Mij,Mkj->Mik', gt_normals, gt_normals)
        assert jnp.allclose(
            gt_normals_dotprod,
            jnp.eye(gt_normals.shape[-2])[None],
            atol=1e-6
        )

        dot_prods = jnp.einsum('Mij,Mkj->Mik', grads, gt_normals)
        proj = jnp.einsum('Mij,Mjk->Mik', dot_prods, gt_normals)

        return jnp.mean(jnp.linalg.norm(grads - proj, axis=-1, ord=2))