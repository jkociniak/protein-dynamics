import torch
import torch.nn as nn
from .utils.tensor import gradients, validate_tensor


class Loss(nn.Module):
    def __init__(self, weights, params, normal_pretraining_batches=0):
        super().__init__()
        self.weights = weights
        self.params = params

        assert normal_pretraining_batches >= 0, 'Normal pretraining epochs must be non-negative'
        self.normal_pretraining_batches = normal_pretraining_batches

        self.last_negatives = None
        self.last_neg_directions = None
        self.last_neg_points = None

    def forward(self, manifold, batch):
        print('WE ARE IN LOSS!')
        # batch is a PyTorch Geometric Batch object or a Data object
        #DEBUG = True
        assert batch.batch.max() == 0, "We expect the batch to be a single point cloud"
        pts = batch.pos
        assert len(pts.shape) == 2, "Coordinates batch must be 2D - (num_points, ambient_dim)"
        #assert pts.shape[0] == 1, "In this implementation batch must contain only a single point cloud"
        assert pts.shape[-1] == manifold.base_manifold.d, "Batch must have the same dimension as the manifold"
        assert manifold.correction_decoder is None, 'Decoder loss is not implemented yet'

        assert self.weights['manifold_norm'] > 0, 'Manifold loss must be enabled'
        assert self.weights['non_manifold_norm'] > 0, 'Non-manifold loss must be enabled'

        if batch.normal_space_dims is not None:
            losses = self.forward_with_normals(manifold, pts, batch.normal_space_dims, normal_basis=batch.normal_basis)
        else:
            losses = self.forward_without_normals(manifold, pts)

        return losses

    def forward_without_normals(self, manifold, pts):
        losses = {}
        out, coords = manifold.correction_encoder(pts)  # dimensions: (N, latent_dim) and (N, ambient_dim)
        grads = gradients(out, coords)  # dimensions: (N, latent_dim, ambient_dim)

        ################################################
        # LATENT REPRESENTATION MAGNITUDE REGULARIZATION
        ################################################
        losses['manifold_norm'] = self.zero_norm_loss(out)
        neg_directions = grads.clone().detach()

        nm_pts = self.generate_negatives(pts, neg_directions)
        nm_pts = torch.flatten(nm_pts, 0, 1)  # dim (N*num_noisy_samples, D)
        self.last_negatives = nm_pts.detach()

        nm_out, nm_coords = manifold.correction_encoder(nm_pts)
        losses['non_manifold_norm'] = self.high_norm_loss(nm_out, self.params['non_manifold_alpha'])

        ##################################################
        # LATENT REPRESENTATION FIRST ORDER REGULARIZATION
        ##################################################
        if self.weights['manifold_eikonal'] > 0:
            losses['manifold_eikonal'] = self.eikonal_loss(grads)

        if self.weights['non_manifold_eikonal'] > 0:
            nm_grads = gradients(nm_out, nm_coords)
            losses['non_manifold_eikonal'] = self.eikonal_loss(nm_grads)

        ##################################################
        # NORMAL FRAME REGULARIZATION
        ##################################################
        if self.weights['orthogonal'] > 0:
            losses['orthogonal'] = self.orthogonal_loss(grads)

        return losses

    def forward_with_normals(self, manifold, pts, normal_space_dims, normal_basis=None):
        losses = {}
        print(f'MEAN NORM OF POSITIVES: {torch.linalg.norm(pts, dim=-1, ord=2).mean()}')
        out, coords = manifold.correction_encoder(pts)  # dimensions: (N, latent_dim) and (N, ambient_dim)
        grads = gradients(out, coords)  # dimensions: (N, latent_dim, ambient_dim)

        ################################################
        # LATENT REPRESENTATION MAGNITUDE REGULARIZATION
        ################################################
        losses['manifold_norm'] = self.zero_norm_loss(out)
        print('MANIFOLD NORM:', losses['manifold_norm'])


        ##################################################
        # NEGATIVE EXAMPLES SAMPLING
        ##################################################
        # 0. As we want to learn Signed Distance Function, we can guess the normal direction by looking at the gradients
        #    of our encoder
        neg_directions = grads.clone().detach()
        # We assume that the first k dimensions are normal space dimensions
        # 1. We zero out the tangent dimensions
        for i, nsd in enumerate(normal_space_dims):
            assert 0 <= nsd <= manifold.correction_encoder.hparams['out_features'], f"Normal space dimension must be less than ambient dimension. Normal space dimension: {nsd}, Ambient dimension: {manifold.base_manifold.d}"
            if nsd < manifold.correction_encoder.hparams['out_features']:
                neg_directions[i, nsd:] = 0

        # 2. If normal basis is provided, we validate it and use it
        if normal_basis is not None:
            for i, nb in enumerate(normal_basis):
                if nb.numel() == 0:
                    assert normal_space_dims[i] == 0, "normal space dimension must be 0 if normal basis is None"
                    continue

                assert isinstance(nb, torch.Tensor), f"Normal basis must be a tensor or None. Normal basis: {type(nb)}"
                assert len(nb.shape) == 2, f"Normal basis must be 2D - (n_normals, ambient_dim). Normal basis: {nb}"
                assert nb.shape[-1] == manifold.base_manifold.d, f"Normal basis must be 2D - (n_normals, ambient_dim). Normal basis: {nb}"
                assert nb.shape[0] <= manifold.base_manifold.d, f"Normal basis cannot contain more than ambient_dim vectors. Normal basis: {nb}"

                gram_mat = nb @ nb.T
                assert (gram_mat - torch.eye(gram_mat.shape[-1])[None, None]).abs().max() < 1e-6, "Normal basis must be orthonormal"
                neg_directions[i, :gram_mat.shape[0]] = nb

        validate_tensor(neg_directions, 'neg_directions')

        print('Number of nondegenerate normal space points: ', (normal_space_dims > 0).sum())

        nm_pts = self.generate_negatives(pts[normal_space_dims > 0], neg_directions[normal_space_dims > 0])
        nm_pts = torch.flatten(nm_pts, 0, 1)  # dim (N*num_noisy_samples, D)
        self.last_negatives = nm_pts.detach()
        self.last_neg_points = pts[normal_space_dims > 0].detach()
        self.last_neg_directions = neg_directions[normal_space_dims > 0].detach()

        ##############################################################
        # LATENT REPRESENTATION MAGNITUDE REGULARIZATION FOR NEGATIVES
        ##############################################################
        print(f'Mean norm of negatives: {torch.linalg.norm(nm_pts, dim=-1, ord=2).mean()}')
        nm_out, nm_coords = manifold.correction_encoder(nm_pts)
        losses['non_manifold_norm'] = self.high_norm_loss(nm_out, self.params['non_manifold_alpha'])
        print(f'NON-MANIFOLD NORM: {losses["non_manifold_norm"]}')

        ##################################################
        # LATENT REPRESENTATION FIRST ORDER REGULARIZATION
        ##################################################

        assert self.weights['manifold_eikonal'] > 0, 'Manifold eikonal loss must be enabled'
        assert self.weights['non_manifold_eikonal'] > 0, 'Non-manifold eikonal loss must be enabled'
        assert self.weights['orthogonal'] > 0, 'Orthogonal loss must be enabled'
        assert self.weights['manifold_normal_subspace'] > 0, 'Manifold normal subspace loss must be enabled'

        nm_grads = gradients(nm_out, nm_coords)  # dimensions: (N*num_noisy_samples, latent_dim, ambient_dim)
        nm_grads = nm_grads.reshape(((normal_space_dims > 0).sum()), -1, nm_grads.shape[-2],
                                    nm_grads.shape[-1])  # dim (N, num_noisy_samples, latent_dim, ambient_dim)

        max_nsd = normal_space_dims.max()
        assert max_nsd.item() < pts.shape[-1], "maximal normal space dimension must be less than ambient space dimension"
        loss_eik = torch.zeros(1, device=pts.device)
        loss_orthogonal = torch.zeros(1, device=pts.device)
        loss_zero_norm = torch.zeros(1, device=pts.device)

        nm_loss_eik = torch.zeros(1, device=pts.device)
        nm_loss_orthogonal = torch.zeros(1, device=pts.device)
        nm_loss_zero_norm = torch.zeros(1, device=pts.device)

        loss_nsp = torch.zeros(1, device=pts.device)

        # loss for points INSIDE the manifold (full-dimensional)

        for d in range(0, max_nsd + 1):
            d_mask = normal_space_dims == d
            if not d_mask.any():
                continue

            tangential_component = grads[d_mask, d:]
            normal_component = grads[d_mask, :d]

            # tangential components work only if normal space dimension is lower than ambient dimension
            if d < max_nsd:
                partial_loss = self.zero_norm_loss(tangential_component)
                print(partial_loss)
                validate_tensor(partial_loss, 'partial_loss')
                loss_zero_norm += partial_loss

            if d > 0:
                nm_d_mask = normal_space_dims[
                                normal_space_dims > 0] == d  # we generated non manifold points only where normal space dimensions > 0

                nm_tangential_component = nm_grads[nm_d_mask, :, d:].flatten(0, 1)
                nm_normal_component = nm_grads[nm_d_mask, :, :d].flatten(0, 1)

                loss_eik += self.eikonal_loss(normal_component)
                loss_orthogonal += self.orthogonal_loss(normal_component)

                if d < max_nsd:
                    nm_loss_zero_norm += self.zero_norm_loss(nm_tangential_component)
                nm_loss_eik += self.eikonal_loss(nm_normal_component)
                nm_loss_orthogonal += self.orthogonal_loss(nm_normal_component)

                if self.normal_pretraining_batches > 0:
                    normal_bases = [normal_basis[i] for i, mask in enumerate(d_mask) if mask.item()]
                    if normal_bases:
                        normal_bases = torch.stack(normal_bases, dim=0)
                        loss_nsp += self.normal_subspace_loss(grads[d_mask, :d], normal_bases)

        validate_tensor(loss_eik, 'loss_eik')
        validate_tensor(loss_orthogonal, 'loss_orthogonal')
        validate_tensor(loss_zero_norm, 'loss_zero_norm')
        validate_tensor(nm_loss_eik, 'loss_eik_nm')
        validate_tensor(nm_loss_orthogonal, 'loss_orthogonal_nm')
        validate_tensor(nm_loss_zero_norm, 'loss_zero_norm_nm')
        validate_tensor(loss_nsp, 'loss_nsp')

        losses['manifold_eikonal'] = loss_eik + loss_zero_norm
        losses['non_manifold_eikonal'] = nm_loss_eik + nm_loss_zero_norm
        losses['orthogonal'] = loss_orthogonal + nm_loss_orthogonal
        losses['manifold_normal_subspace'] = loss_nsp
        self.normal_pretraining_batches = max(0, self.normal_pretraining_batches - 1)

        return losses

    def generate_negatives(self, x, normal_basis, n_samples=5):
        # x shape: (M, D)
        # normal basis shape: (M, enc_dim, D)
        weights = torch.randn((normal_basis.shape[0], n_samples, normal_basis.shape[1]), device=x.device) ** 2 # dim (M, num_noisy_samples, enc_dim)
        validate_tensor(weights, 'weights')
        normals = torch.einsum('Mnk, Mkd->Mnd', weights, normal_basis)  # dim (M, num_noisy_samples, D)
        validate_tensor(normals, 'normals')
        normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)  # dim (M, num_noisy_samples, D)
        validate_tensor(normals, 'normals')

        samples = x[:, None, :] + self.params['non_manifold_eps'] * normals
        return samples

    @staticmethod
    def zero_norm_loss(pts):
        partial = torch.linalg.vector_norm(pts, dim=-1, ord=1)
        validate_tensor(partial, 'zero_norm_loss inner fun')
        return partial.mean()

    @staticmethod
    def high_norm_loss(pts, alpha):
        norms = torch.linalg.norm(pts, dim=-1, ord=1)
        return torch.exp(-alpha * norms).mean()

    @staticmethod
    def eikonal_loss(grads):
        grad_norms = torch.linalg.vector_norm(grads, dim=-1, ord=2)
        return torch.abs(grad_norms - 1).mean()

    @staticmethod
    def orthogonal_loss(grads):
        unit_grads = grads / torch.linalg.norm(grads, dim=-1, keepdim=True)
        grad_inner = torch.einsum('Nij, Nkj->Nik', unit_grads, unit_grads)
        _, ed, _ = grad_inner.shape
        grad_inner[..., range(ed), range(ed)] = 0  # set diagonal to zero to not blend with eikonal loss
        return torch.abs(grad_inner).mean()

    @staticmethod
    def normal_loss(grads, gt_normals):
        return torch.linalg.vector_norm(grads - gt_normals, dim=-1, ord=2).mean()

    @staticmethod
    def normal_subspace_loss(grads, gt_normals):
        # grads shape: (M, enc_dim, D)
        # gt_normal_frame shape: (M, normal_space_dim, D)
        assert grads.shape[-1] == gt_normals.shape[-1], "Normal frame must have the same dimension as the gradients"
        assert grads.shape[0] == gt_normals.shape[0], "Normal frame must have the same batch size as the gradients"
        gt_normals_dotprod = torch.einsum('Mij, Mkj->Mik', gt_normals, gt_normals)
        assert gt_normals_dotprod.allclose(torch.eye(gt_normals.shape[-2])[None], atol=1e-6), "Normal frame must be orthogonal"
        dot_prods = torch.einsum('Mij, Mkj->Mik', grads, gt_normals)  # shape: (M, enc_dim, normal_space_dim)
        proj = torch.einsum('Mij, Mjk->Mik', dot_prods, gt_normals)  # shape: (M, enc_dim, D)
        diff = torch.linalg.vector_norm(grads - proj, dim=-1, ord=2).mean()
        return diff

    # def regularize_second_order(self):
    #     # DIVERGENCE LOSS
    #     if self.weights['manifold_div'] > 0 and self.div_ctr % 5 == 0:
    #         if grads is None:
    #             grads = gradients(out, coords)
    #         manifold_div = 0.
    #         for out_i in range(grads.shape[1]):
    #             manifold_div += (directional_div(coords, grads[:, out_i, :]) ** 2).mean()
    #         losses['manifold_div'] = manifold_div
    #         self.div_ctr += 1
    #
    #     # DIVERGENCE LOSS FOR NEGATIVE EXAMPLES
    #     if self.weights['non_manifold_div'] > 0:
    #         if nm_grads is None:
    #             if nm_out is None:
    #                 if nm_pts is None:
    #                     nm_pts = self.generate_negatives_unsupervised(batch, grads)
    #                 nm_out, nm_coords = manifold.correction_encoder(nm_pts)
    #             nm_grads = gradients(nm_out, nm_coords)
    #         nm_div = 0.
    #         for out_i in range(nm_grads.shape[1]):
    #             nm_div += torch.abs(directional_div(nm_coords, nm_grads[:, out_i, :])).mean()
    #         losses['non_manifold_div'] = nm_div
    #     pass

    # def unused(self):
    #     if manifold.correction_decoder is not None:
    #         tangent_dims = manifold.correction_decoder.hparams['in_features']
    #     else:
    #         assert self.weights['reconstruction'] == 0, 'Reconstruction loss is enabled but no decoder is provided'
    #
    #     assert self.weights['manifold_norm'] > 0 or self.weights['reconstruction'] > 0, 'At least one of manifold_norm or reconstruction must be enabled'
    #
    #     # RECONSTRUCTION LOSS
    #     if self.weights['reconstruction'] > 0:
    #         rec, _ = manifold.correction_decoder(tangent_coords)
    #         losses['reconstruction'] = torch.linalg.vector_norm(rec - coords, dim=-1, ord=2).mean()
    #
    #     # COSINE LOSS BETWEEN CONSECUTIVE GRADIENTS
    #     if self.weights['cosine'] > 0:
    #         # grads dimensions: (N, M, enc_dim, D)
    #         # we would like to have consistent frames
    #         # frame is (enc_dim, D) matrix where rows are normal vectors in ambient space
    #         # we can compute cosine similarity between consecutive frames
    #         # we assume here that the graph is a chain
    #         # so we want to pull cosine simailiarity of nearby frames together
    #         # we want matrix (N, M-1, enc_dim)
    #         grads_dots = torch.sum(grads[:, :-1] * grads[:, 1:], dim=-1)
    #         grad_norms1 = torch.linalg.norm(grads[:, :-1], dim=-1, ord=2)
    #         grad_norms2 = torch.linalg.norm(grads[:, 1:], dim=-1, ord=2)
    #         cosines = grads_dots / grad_norms1 / grad_norms2
    #         print(f'Number of entries where cosine is negative: {torch.sum(cosines < 0)}')
    #         weights = torch.exp(-0.5 * cosines)
    #         loss = torch.abs(weights * (1 - cosines)).mean()
    #         losses['cosine'] = loss
    #
    #     # GEODESIC LOSS
    #     if self.weights['geodesic'] > 0:
    #         x1 = batch[:, :-1, :]  # dimensions: (B, N-1, D)
    #         x2 = batch[:, 1:, :]  # dimensions: (B, N-1, D)
    #
    #         corrected_dists = manifold.distance(x1, x2)  # dimensions (B, N-1)
    #         lhs = torch.sum(corrected_dists, dim=1) ** 2  # dimensions: (B)
    #         rhs = corrected_dists.shape[1] * torch.sum(corrected_dists ** 2, dim=1)  # dimensions: (B)
    #         g_loss = nn.functional.mse_loss(lhs, rhs)  # dimensions: (B)
    #         losses['geodesic'] = g_loss.mean()
    #
    #     # HESSIAN LOSS
    #     if self.weights['neg_hess_norm'] > 0:
    #         # if out is None:
    #         #     out, coords = manifold.correction_encoder(batch)
    #         # x_enc_norm = torch.linalg.norm(out, dim=2, keepdim=True)  # dimensions: (N, M)
    #         # aux = 0.5 * x_enc_norm ** 2  # dimensions: (N, M)
    #         #
    #         # aux_fun_grad = gradients(aux, coords)  # dimensions: (N, M, 1, d)
    #         #
    #         # scalar_coeff = 16 / (x_enc_norm ** 2 + 1) ** 4  # dimensions: (N, M)
    #         # grad_norm = torch.linalg.norm(aux_fun_grad.squeeze(2), dim=2) ** 4  # dimensions: (N, M)
    #         # fro_norm = scalar_coeff * grad_norm  # dimensions: (N, M)
    #         #
    #         # hess_norm = torch.clamp(fro_norm, max=self.params['max_hess_norm'])
    #         # neg_hess_norm = -hess_norm.mean()
    #         # losses['neg_hess_norm'] = neg_hess_norm
    #         mt = manifold.metric_tensor(batch)
    #         mt_loss = torch.linalg.matrix_norm(mt - torch.eye(mt.shape[-1])).mean()
    #         losses['neg_hess_norm'] = mt_loss
    #    pass
