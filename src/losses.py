import torch
import torch.nn as nn
from .utils.tensor import gradients, directional_div, gradients_pc

from torch_geometric.data import Batch


class Loss(nn.Module):
    def __init__(self, weights, params):
        super().__init__()
        self.weights = weights
        self.params = params

        self.div_ctr = 0

    def forward(self, manifold, batch):
        # THIS IS THE MOST HORRIBLE PIECE OF CODE I HAVE EVER WRITTEN. I'M SORRY.
        # A LOT OF THESE LOSSES SHARE INPUTS
        # SO REPEATING THE SAME COMPUTATION IS A WASTE OF TIME
        # THEREFORE EACH LOSS FUNCTION CHECKS IF THE INPUTS HAVE BEEN ALREADY COMPUTED

        assert len(batch.shape) == 3, "Batch must be 3D - (num_traj, num_points, dim)"
        losses = {}

        tangent_dims = 0
        if manifold.correction_decoder is not None:
            tangent_dims = manifold.correction_decoder.hparams['in_features']
        else:
            assert self.weights['reconstruction'] == 0, 'Reconstruction loss is enabled but no decoder is provided'

        assert self.weights['manifold_norm'] > 0 or self.weights['reconstruction'] > 0, 'At least one of manifold_norm or reconstruction must be enabled'
        out, coords = manifold.correction_encoder(batch)

        tangent_coords = out[..., :tangent_dims]
        normal_coords = out[..., tangent_dims:]

        # MANIFOLD POINTS LOSS
        if self.weights['manifold_norm'] > 0:
            losses['manifold_norm'] = torch.linalg.vector_norm(normal_coords, dim=-1, ord=1).mean()

        # RECONSTRUCTION LOSS
        if self.weights['reconstruction'] > 0:
            rec, _ = manifold.correction_decoder(tangent_coords)
            losses['reconstruction'] = torch.linalg.vector_norm(rec - coords, dim=-1, ord=2).mean()

        grads = gradients(out, coords)  # dimensions: (N, M, enc_dim, D)

        # EIKONAL LOSS
        if self.weights['manifold_eikonal'] > 0:
            grad_norms = torch.linalg.vector_norm(grads, dim=2, ord=2)
            losses['manifold_eikonal'] = torch.abs(grad_norms - 1).mean()

        # ORTHOGONAL GRADIENTS LOSS
        if self.weights['orthogonal'] > 0:
            unit_grads = grads / torch.linalg.norm(grads, dim=-1, keepdim=True)
            grad_inner = torch.einsum('NMij, NMkj->NMik', unit_grads, unit_grads)
            _, _, ed, _ = grad_inner.shape
            grad_inner[..., range(ed), range(ed)] = 0  # set diagonal to zero to not blend with eikonal loss
            losses['orthogonal'] = torch.abs(grad_inner).mean()

        # COSINE LOSS BETWEEN CONSECUTIVE GRADIENTS
        if self.weights['cosine'] > 0:
            # grads dimensions: (N, M, enc_dim, D)
            # we would like to have consistent frames
            # frame is (enc_dim, D) matrix where rows are normal vectors in ambient space
            # we can compute cosine similarity between consecutive frames
            # we assume here that the graph is a chain
            # so we want to pull cosine simailiarity of nearby frames together
            # we want matrix (N, M-1, enc_dim)
            grads_dots = torch.sum(grads[:, :-1] * grads[:, 1:], dim=-1)
            grad_norms1 = torch.linalg.norm(grads[:, :-1], dim=-1, ord=2)
            grad_norms2 = torch.linalg.norm(grads[:, 1:], dim=-1, ord=2)
            cosines = grads_dots / grad_norms1 / grad_norms2
            print(f'Number of entries where cosine is negative: {torch.sum(cosines < 0)}')
            weights = torch.exp(-0.5*cosines)
            loss = torch.abs(weights * (1 - cosines)).mean()
            losses['cosine'] = loss

            # DIVERGENCE LOSS
        if self.weights['manifold_div'] > 0 and self.div_ctr % 5 == 0:
            if grads is None:
                grads = gradients(out, coords)
            manifold_div = 0.
            for out_i in range(grads.shape[1]):
                manifold_div += (directional_div(coords, grads[:, out_i, :]) ** 2).mean()
            losses['manifold_div'] = manifold_div
            self.div_ctr += 1

        # NON-MANIFOLD POINTS LOSS
        nm_pts = self.generate_negatives_unsupervised(batch, grads)
        nm_out, nm_coords = manifold.correction_encoder(nm_pts)

        nm_normal_coords = nm_out[..., tangent_dims:]

        # NORM LOSS FOR NEGATIVE EXAMPLES
        if self.weights['non_manifold_norm'] > 0:
            nm_norms = torch.linalg.norm(nm_normal_coords, dim=-1, ord=1)
            losses['non_manifold_norm'] = torch.exp(-self.params['non_manifold_alpha'] * nm_norms).mean()

        # EIKONAL LOSS FOR NEGATIVE EXAMPLES
        if self.weights['non_manifold_eikonal'] > 0:
            nm_grads = gradients(nm_out, nm_coords)
            nm_grad_norms = torch.linalg.vector_norm(nm_grads, dim=2, ord=2)
            losses['non_manifold_eikonal'] = torch.abs(nm_grad_norms - 1).mean()

        # DIVERGENCE LOSS FOR NEGATIVE EXAMPLES
        if self.weights['non_manifold_div'] > 0:
            if nm_grads is None:
                if nm_out is None:
                    if nm_pts is None:
                        nm_pts = self.generate_negatives_unsupervised(batch, grads)
                    nm_out, nm_coords = manifold.correction_encoder(nm_pts)
                nm_grads = gradients(nm_out, nm_coords)
            nm_div = 0.
            for out_i in range(nm_grads.shape[1]):
                nm_div += torch.abs(directional_div(nm_coords, nm_grads[:, out_i, :])).mean()
            losses['non_manifold_div'] = nm_div

        # GEODESIC LOSS
        if self.weights['geodesic'] > 0:
            x1 = batch[:, :-1, :]  # dimensions: (B, N-1, D)
            x2 = batch[:, 1:, :]  # dimensions: (B, N-1, D)

            corrected_dists = manifold.distance(x1, x2)  # dimensions (B, N-1)
            lhs = torch.sum(corrected_dists, dim=1) ** 2  # dimensions: (B)
            rhs = corrected_dists.shape[1] * torch.sum(corrected_dists ** 2, dim=1)  # dimensions: (B)
            g_loss = nn.functional.mse_loss(lhs, rhs)  # dimensions: (B)
            losses['geodesic'] = g_loss.mean()

        # HESSIAN LOSS
        if self.weights['neg_hess_norm'] > 0:
            # if out is None:
            #     out, coords = manifold.correction_encoder(batch)
            # x_enc_norm = torch.linalg.norm(out, dim=2, keepdim=True)  # dimensions: (N, M)
            # aux = 0.5 * x_enc_norm ** 2  # dimensions: (N, M)
            #
            # aux_fun_grad = gradients(aux, coords)  # dimensions: (N, M, 1, d)
            #
            # scalar_coeff = 16 / (x_enc_norm ** 2 + 1) ** 4  # dimensions: (N, M)
            # grad_norm = torch.linalg.norm(aux_fun_grad.squeeze(2), dim=2) ** 4  # dimensions: (N, M)
            # fro_norm = scalar_coeff * grad_norm  # dimensions: (N, M)
            #
            # hess_norm = torch.clamp(fro_norm, max=self.params['max_hess_norm'])
            # neg_hess_norm = -hess_norm.mean()
            # losses['neg_hess_norm'] = neg_hess_norm
            mt = manifold.metric_tensor(batch)
            mt_loss = torch.linalg.matrix_norm(mt - torch.eye(mt.shape[-1])).mean()
            losses['neg_hess_norm'] = mt_loss

        return losses

    def generate_negatives_unsupervised(self, x, normal_basis):
        print(f'x device: {x.device}')
        print(f'normal basis device: {normal_basis.device}')
        raise Exception
        # x shape: (N, M, D)
        # normal basis shape: (N, M, enc_dim, D)
        # we want to generate two samples for each normal basis vector
        # so weights shape: (N, M, enc_dim, 2, D)
        ones = torch.ones((*normal_basis.shape[:-1], 2, normal_basis.shape[-1]), device=x.device)
        weights = self.params['non_manifold_eps'] * ones
        weights[..., 0, :] = -weights[..., 0, :]

        #to generate dirs we just need to multiply normal basis by weights and use broadcasting
        dirs = normal_basis.unsqueeze(3) * weights  # dim (N, M, C, 2, D)
        dirs = torch.flatten(dirs, 2, 3)  # dim (N, M, 2*C, D)
        samples = x[..., None, :] + dirs  # dim (N, M, 2*C, D)
        samples = torch.flatten(samples, 1, 2)  # dim (N, 2*M*C, D)

        assert x.shape[1] > 1, 'This function is only implemented for sequences of at least 2 points'
        p = x[:, [0, -1]]
        v = x[:, [1, -2]] - x[:, [0, -1]]
        tube_ends = p - self.params['non_manifold_eps'] * v # dim (N, 2, D)

        samples = torch.cat([tube_ends, samples], dim=1)  # dim (N, 2*M*C + 2, D)
        return samples

    def generate_negatives_3D(self, x, n_samples=16):
        assert len(x.shape) == 2  # dimensions: (N, D)
        assert x.shape[1] == 3  # implementation only for 3D data

        # 1. compute the start and end points of each segment
        x1 = x[:-1, :]
        x2 = x[1:, :]

        midpoints = (x1 + x2) / 2
        d = x2 - x1

        # Choose two orthogonal vectors to the direction vector
        prefix = torch.broadcast_to(torch.tensor([1, 0]), (d.shape[0], 2)).to(d.device)
        aux = -d[:, 0] / d[:, 2]
        aux = aux[:, None]
        v1 = torch.cat([prefix, aux], dim=1)
        v1 /= torch.linalg.norm(v1, axis=1, keepdims=True)  # dimensions (N-1, D)
        v2 = torch.linalg.cross(d, v1)  # dimensions (N-1, D)

        # Sample polar coordinates within the disk's radius
        theta = 2 * torch.pi * torch.rand(size=(n_samples, v1.shape[0])).to(v1.device)  # dimensions (NS, N-1)
        r = torch.rand(size=(n_samples, v1.shape[0])).to(v1.device)
        r = torch.sqrt(r)

        # Transform polar coordinates to Cartesian coordinates in the plane of the disk
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Translate the point to the actual position in 3D space
        # we multiply (NS, N-1) by (N-1, D)
        normals = self.loss_params['non_manifold_eps'] * (x[:, :, None] * v1[None] + y[:, :, None] * v2[None])
        tubular = midpoints + normals
        tubular = tubular.view(-1, 3).float()  # dimensions: (n_samples*(N-1), 3)

        return tubular


class GraphLoss(nn.Module):
    def __init__(self, weights, params):
        super().__init__()
        self.weights = weights
        self.params = params

    def forward(self, manifold, batch):
        assert isinstance(batch, Batch), "Batch must be a PyTorch Geometric Batch object"
        n_batches = batch.batch.max().item() + 1
        losses = {}

        # MANIFOLD POINTS LOSS
        assert self.weights['manifold_norm'] > 0, 'Manifold loss must be enabled'
        out, coords = manifold.correction_encoder(batch)
        losses['manifold_norm'] = torch.linalg.vector_norm(out, dim=-1, ord=1).mean()

        # EIKONAL LOSS
        grads = None
        if self.weights['manifold_eikonal'] > 0:
            # view(n_batches, -1, coords.shape[-1])
            grads = gradients_pc(out, coords)
            grad_norms = torch.linalg.vector_norm(grads, dim=2, ord=2)
            losses['manifold_eikonal'] = torch.abs(grad_norms - 1).mean()

        if self.weights['orthogonal'] > 0:
            raise NotImplementedError('Orthogonal loss for graphs is not implemented yet')
            if grads is None:
                grads = gradients_pc(out, coords.view(n_batches, -1, coords.shape[-1]))
            grads = grads[:, :3]
            grad_inner = torch.einsum('Nij, Nkj->Nik', grads, grads)
            N, D, _ = grad_inner.shape
            grad_inner[:, range(D), range(D)] = 0
            losses['orthogonal'] = torch.abs(grad_inner).sum()

        if self.weights['manifold_div'] > 0:
            raise NotImplementedError('Manifold div not implemented for graphs')
            if grads is None:
                grads = gradients_pc(out, coords.view(n_batches, -1, coords.shape[-1]))
            manifold_div = 0.
            for out_i in range(grads.shape[1]):
                manifold_div += torch.abs(directional_div(coords, grads[:, out_i, :])).mean()
            losses['manifold_div'] = manifold_div

        # NON-MANIFOLD POINTS LOSS
        nm_pts = None
        if self.weights['non_manifold_norm'] > 0 or \
                self.weights['non_manifold_eikonal'] > 0 or \
                self.weights['non_manifold_div'] > 0:
            raise NotImplementedError('Non-manifold points loss for graphs is not implemented yet')

            if grads is None:
                grads = gradients(out, coords)

            nm_pts = self.generate_negatives_unsupervised(batch.pos, grads)
            x = torch.ones(nm_pts.shape[0], 1).to(batch.pos.device)
            nm_pts = Data(x=x, pos=nm_pts, edge_index=batch.edge_index)

        nm_out = None
        nm_coords = None
        if self.weights['non_manifold_norm'] > 0:
            raise NotImplementedError('Non-manifold points loss for graphs is not implemented yet')
            assert nm_pts is not None, 'Non-manifold points must be generated'
            # self.last_negatives = non_manifold_pts.clone().detach().numpy()

            nm_out, nm_coords = manifold.correction_encoder(nm_pts)
            nm_norms = torch.linalg.norm(nm_out, dim=-1, ord=1)
            losses['non_manifold_norm'] = torch.exp(-self.loss_params['non_manifold_alpha'] * nm_norms).mean()

        nm_grads = None
        if self.weights['non_manifold_eikonal'] > 0:
            raise NotImplementedError('Non-manifold points loss for graphs is not implemented yet')
            if nm_out or nm_coords is None:
                if nm_pts is None:
                    nm_pts = self.generate_negatives_unsupervised(batch, grads)
                nm_out, nm_coords = manifold.correction_encoder(nm_pts)
            nm_grads = gradients(nm_out, nm_coords)
            nm_grad_norms = torch.linalg.vector_norm(nm_grads, dim=2, ord=2)
            losses['non_manifold_eikonal'] = torch.abs(nm_grad_norms - 1).mean()

        if self.weights['non_manifold_div'] > 0:
            raise NotImplementedError('Non-manifold points loss for graphs is not implemented yet')
            if nm_grads is None:
                if nm_out is None:
                    if nm_pts is None:
                        nm_pts = self.generate_negatives_unsupervised(batch, grads)
                    nm_out, nm_coords = manifold.correction_encoder(nm_pts)
                nm_grads = gradients(nm_out, nm_coords)
            nm_div = 0.
            for out_i in range(nm_grads.shape[1]):
                nm_div += torch.abs(directional_div(nm_coords, nm_grads[:, out_i, :])).mean()
            losses['non_manifold_div'] = nm_div

        if self.weights['geodesic'] > 0:
            # THIS PART ASSUMES THAT THE BATCH IS A SEQUENCE OF POINTS
            #raise NotImplementedError('Geodesic loss for graphs is not implemented yet')
            x1 = batch.index_select(slice(0, -1))
            x1 = Batch.from_data_list(x1)
            x2 = batch.index_select(slice(1, None))
            x2 = Batch.from_data_list(x2)

            deep_dists, base_dists = manifold.distance(x1, x2, return_base_dists=True)  # dimensions (1, N-1)
            print(f'Base dists mean: {base_dists.mean()}, max: {base_dists.max()}')
            print(f'Base dists (rescaled) mean: {base_dists.mean() * manifold.alpha}, max: {base_dists.max() * manifold.alpha}')
            print(f'Deep dists mean: {deep_dists.mean()}, max: {deep_dists.max()}')
            print(f'Deep dists (rescaled) mean: {deep_dists.mean() * manifold.beta}, max: {deep_dists.max() * manifold.beta}')

            lhs = torch.sum(base_dists, dim=1) ** 2  # dimensions: (1)
            rhs = base_dists.shape[1] * torch.sum(base_dists ** 2, dim=1)  # dimensions: (1)
            base_g_loss = nn.functional.mse_loss(lhs, rhs)  # dimensions: (1)
            print(f'Base geodesic loss: {base_g_loss}')

            corrected_dists = torch.sqrt(manifold.alpha * base_dists ** 2 + manifold.beta * deep_dists ** 2)
            lhs = torch.sum(corrected_dists, dim=1) ** 2  # dimensions: (1)
            rhs = corrected_dists.shape[1] * torch.sum(corrected_dists ** 2, dim=1)  # dimensions: (1)
            g_loss = nn.functional.mse_loss(lhs, rhs)  # dimensions: (1)
            print(f'Corrected geodesic loss: {g_loss}')
            losses['geodesic'] = g_loss

        return losses

    def generate_negatives_unsupervised(self, x, normal_basis):
        weights = self.params['non_manifold_eps'] * torch.ones((x.shape[0], 2, normal_basis.shape[1]))
        weights[:, 0, :] = -weights[:, 0, :]
        # weights (N, S, M)
        # normal basis (N, M, D)
        dirs = (normal_basis.unsqueeze(1) * weights.unsqueeze(-1)).sum(2)  # dim (N, S, D)
        samples = x.unsqueeze(1) + dirs  # dim (N, S, D)
        samples = samples.view(-1, samples.shape[-1])  # dim (N*S, D)

        return samples