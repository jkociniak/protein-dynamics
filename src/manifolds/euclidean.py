import torch
from torch import nn
from torch.func import vmap

from ..utils.utils import validate_tensor
from .base import Manifold


class Euclidean(Manifold):
    """ Base class describing Euclidean space of dimension `d` """

    def __init__(self, d, a=1.):
        super().__init__(d)
        self.a = a

    def inner(self, p, X, Y):
        """
        :param p: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        assert (len(p.shape) + 1) == len(X.shape) == len(Y.shape)
        assert p.shape[-1] == X.shape[-1] == Y.shape[-1] == self.d and p.shape[:-1] == X.shape[:-2] == Y.shape[:-2]
        if len(p.shape) > 2:  # so N is a tensor
            pp = p.reshape(-1, self.d)
            XX = X.reshape(-1, X.shape[-2], X.shape[-1])
            YY = Y.reshape(-1, Y.shape[-2], Y.shape[-1])
            return self.inner(pp, XX, YY).reshape(p.shape[:-1], X.shape[-2], Y.shape[-2])
        else:
            return self.a * torch.einsum("NMi,NLi->NML", X, Y)

    def pairwise_distance(self, p, q):
        """
        :param p: N x M x d
        :param q: N x M' x d
        :return: N x M x M'
        """
        assert len(p.shape) == len(q.shape)
        assert p.shape[0] == q.shape[0] and p.shape[-1] == q.shape[-1] == self.d
        return torch.sqrt(self.a * torch.sum((p.unsqueeze(-2) - q.unsqueeze(-3)) ** 2, -1) + 1e-8)

    def distance(self, p, q):
        """

        :param p: N x M x d
        :param q: N x M x d
        :return: N x M
        """
        assert p.shape == q.shape
        assert p.shape[-1] == self.d

        out = torch.sqrt(self.a * torch.sum((p - q) ** 2, -1) + 1e-8)
        assert out.shape == p.shape[:-1]
        return out

    def geodesic(self, p, q, t):
        """
        :param p: N x 1 x d tensor
        :param q: N x 1 x d tensor
        :param t: M tensor
        :return: N x M x d tensor
        """

        def single_example_geodesic(p, q, t):
            """
            :param p: 1 x d tensor
            :param q: 1 x d tensor
            :param t: M tensor
            :return: M x d tensor
            """
            (1 - t[:, None]) * p + t[:, None] * q

        batched_geodesic_fn = vmap(single_example_geodesic, in_dims=(0, 0, None), out_dims=0)
        return batched_geodesic_fn(p, q, t)

    def log(self, p, q):
        """

        :param p: N x M x d
        :param q: N x M' x d
        :return: N x M x M' x d
        """
        assert len(p.shape) == len(q.shape)
        assert p.shape[0] == q.shape[0]
        assert p.shape[-1] == q.shape[-1] == self.d
        res = q.unsqueeze(1) - p.unsqueeze(2)
        assert res.shape == (p.shape[0], p.shape[1], q.shape[1], self.d)
        return res

    def exp(self, p, X):
        """
        :param p: N x d
        :param X: N x M x d
        :return: N x M x d
        """
        assert (len(p.shape) + 1) == len(X.shape)
        return p.unsqueeze(-2) + X

    def parallel_transport(self, p, X, q):
        """

        :param p: N x d
        :param X: N x M x d
        :param q: N x d
        :return: N x M x d
        """
        return X

    def manifold_dimension(self):
        return self.d

    # def sharp(self, p, Xi):
    #     return Xi

    # def flat(self, p, X):
    #     return X


class CorrectedEuclideanManifold(Manifold):
    def __init__(self, base_manifold_params, correction_encoder, beta, learnable_beta):
        assert 'd' in base_manifold_params
        super().__init__(d=base_manifold_params['d'])

        self.base_manifold = Euclidean(**base_manifold_params)
        self.correction_encoder = correction_encoder
        self.disable_correction = False

        assert isinstance(learnable_beta, bool)
        self.learnable = learnable_beta

        self.beta = nn.Parameter(torch.tensor([1.], dtype=torch.float32), requires_grad=learnable_beta)
        if beta is not None:
            assert isinstance(beta, float) and beta >= 0.
            beta_val = beta
        else:
            with torch.no_grad():
                beta_val = self.init_correction_coeff()
        self.beta.data.fill_(beta_val)

    def init_correction_coeff(self):
        # generate mesh data
        x = torch.linspace(0, torch.pi, 30)
        y = torch.linspace(0, 1, 30)
        xv, yv = torch.meshgrid(x, y)
        xy = torch.vstack([xv.ravel(), yv.ravel()]).T
        ps = xy[None]
        target = torch.tensor([torch.pi, 0])[None, None].float()

        # compute base and corrected prelogs
        base_prelogs = self.base_manifold.log(ps, target).squeeze([0, 2])
        correction_prelogs, _, _, _ = self.prelog(ps, target)
        correction_prelogs = correction_prelogs.squeeze([0, 2])

        # set beta so the correction prelogs norm mean is the same magnitude as the base prelogs norm mean
        base_prelogs_norm_mean = torch.norm(base_prelogs, dim=1).mean()
        correction_prelogs_norm_mean = torch.norm(correction_prelogs, dim=1).mean()
        correction_coeff = 100 * (base_prelogs_norm_mean / correction_prelogs_norm_mean).item()

        return correction_coeff

    def pairwise_distance(self, x, y):
        """
        Manifold distance between points x and y
        :param x: N x M x d tensor
        :param y: N x M' x d tensor
        :return: N x M x M' tensor
        """

        base_dists = self.base_manifold.pairwise_distance(x, y)  # dimensions: (N, M, M')
        if self.disable_correction:
            return base_dists

        x_enc = self.correction_encoder.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
        y_enc = self.correction_encoder.forward_2d_batch(y)  # dimensions: (N, M', enc_dim)
        x_enc = x_enc.unsqueeze(2)  # dimensions: (N, M, 1, enc_dim)
        y_enc = y_enc.unsqueeze(1)  # dimensions: (N, 1, M', enc_dim)

        x_enc_norm = torch.norm(x_enc, dim=3)**2 + 1  # dimensions: (N, M, 1)
        y_enc_norm = torch.norm(y_enc, dim=3)**2 + 1  # dimensions: (N, 1, M')
        deep_dists = torch.log(x_enc_norm / y_enc_norm)  # dimensions: (N, M, M')

        corrected_dists = torch.sqrt(base_dists ** 2 + self.beta * deep_dists ** 2)  # dimensions: (N, M, M')
        return corrected_dists

    def distance(self, x, y):
        """
        :param x: N x M x d tensor
        :param y: N x M x d tensor
        :return: N x M tensor
        """
        base_dists = self.base_manifold.distance(x, y)  # dimensions: (N, M)
        if self.disable_correction:
            return base_dists

        x_enc = self.correction_encoder.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
        y_enc = self.correction_encoder.forward_2d_batch(y)  # dimensions: (N, M, enc_dim)

        x_enc_norm = torch.norm(x_enc, dim=2) ** 2 + 1  # dimensions: (N, M)
        y_enc_norm = torch.norm(y_enc, dim=2) ** 2 + 1  # dimensions: (N, M)
        deep_dists = torch.log(x_enc_norm / y_enc_norm)

        corrected_dists = torch.sqrt(base_dists ** 2 + self.beta * deep_dists ** 2)
        return corrected_dists

    def forward(self, x, y):
        return self.distance(x, y)

    def norm(self, x, X):
        """

        :param x: N x M x d tensor
        :param X: N x M x L x d tensor
        :return: N x M x L tensor
        """
        assert x.shape[0] == X.shape[0]

        N = x.shape[0]
        M = x.shape[1]
        L = X.shape[2]

        norm = torch.zeros(N, M, L)
        for l in range(L):
            norm[:, :, l] = torch.sqrt(
                self.inner(x, X[:, :, l, :][:, :, None, :], X[:, :, l, :][:, :, None, :])[:, :, 0])

        return norm

    def inner(self, x, X, Y):
        """

        :param x: N x M x d tensor
        :param X: N x M x L x d tensor
        :param Y: N x M x K x d tensor
        :return: N x M x L x K tensor
        """
        assert x.shape[0] == X.shape[0] == Y.shape[0]

        H = self.metric_tensor(x)  # dimensions: (N, M, d, d)
        inner = torch.einsum("NMij,NMLi,NMKj->NMLK", H, X, Y)

        return inner

    def metric_tensor(self, x):
        """

        :param x: N x M x d tensor
        :param asmatrix:
        :return: N x M x d x d
        """

        x_enc = self.correction_encoder.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
        x_enc_norm = torch.linalg.vector_norm(x_enc, dim=2, keepdim=True).unsqueeze(2)  # dimensions: (N, M, 1, 1)
        aux_fun_grad_val = self.correction_encoder.aux_fun_grad(x, twodim_batch=True)  # dimensions: (N, M, d)
        hessian = self.log_corr_hessian(x_enc_norm, aux_fun_grad_val)  # dimensions: (N, M, d, d)

        return hessian

    def geodesic(self, x, y, tau, step_size=1., max_iter=100, tol=1e-3, debug=False):
        """

        :param x: N x 1 x d tensor
        :param y: N x 1 x d tensor
        :param tau: M tensor

        :param step_size: float
        :param max_iter: int
        :param tol: float
        :param debug: bool
        :return: N x M x d tensor
        """

        if self.disable_correction:
            return self.base_manifold.geodesic(x, y, tau)

        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == 1

        error0 = self.distance(x, y).max() + 1e-6
        relerror = 1.
        k = 1
        z = torch.ones(len(tau))[None, :, None].to(x.device) * y
        while relerror > tol and k <= max_iter:
            # compute grad
            grad_Wzx = - self.log(z, x)[:, :, 0]
            validate_tensor(grad_Wzx, "grad_Wzx")

            grad_Wzy = - self.log(z, y)[:, :, 0]
            validate_tensor(grad_Wzy, "grad_Wzy")

            grad_Wz = (1 - tau[None, :, None]) * grad_Wzx + tau[None, :, None] * grad_Wzy
            # update z
            z = z - step_size * grad_Wz
            validate_tensor(z, "z")

            # compute new error
            error = self.norm(z, grad_Wz[:, None]).max()
            relerror = error / error0
            if debug:
                print(f"{k} | relerror = {relerror}")

            k = k + 1

        return z

    def log_corr_hessian(self, x_enc_norm, aux_fun_grad_val):
        # compute hessian for log^2(||x||^2 + 1 / ||.||^2 + 1)
        scalar_coeff = 8 / (x_enc_norm ** 2 + 1) ** 2  # dimensions: (N, M, 1, 1)
        scalar_coeff = scalar_coeff.squeeze((2, 3))  # dimensions: (N, M)
        correction_hessian = self.beta * 0.5 * torch.einsum("NM,NMa,NMb->NMab", scalar_coeff, aux_fun_grad_val, aux_fun_grad_val)  # dimensions: (N, M, d, d)
        return correction_hessian

    def prelog(self, x, y):
        """
        :param x: N x M x d tensor
        :param y: N x M' x d tensor
        :return: N x M x M' x d tensor
        """
        prelog = self.base_manifold.log(x, y)  # dimensions: (N, M, M', d)
        validate_tensor(prelog, 'prelog')
        if self.disable_correction:
            return prelog

        # 1. compute scalar coefficient in the correction grad
        x_enc = self.correction_encoder.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
        y_enc = self.correction_encoder.forward_2d_batch(y)  # dimensions: (N, M', enc_dim)

        x_enc_norm = torch.linalg.vector_norm(x_enc, dim=2, keepdim=True).unsqueeze(2)  # dimensions: (N, M, 1, 1)
        y_enc_norm = torch.linalg.vector_norm(y_enc, dim=2, keepdim=True).unsqueeze(1)  # dimensions: (N, 1, M', 1)
        scalar_coeff = 4 * torch.log((x_enc_norm ** 2 + 1) / (y_enc_norm ** 2 + 1)) / (
                x_enc_norm ** 2 + 1)  # dimensions: (N, M, M', 1)

        # 2. compute gradient of the auxiliary function
        aux_fun_grad_val = self.correction_encoder.aux_fun_grad(x, twodim_batch=True)  # dimensions: (N, M, d)

        # 3. combine to get correction gradient
        deep_corr_grad = scalar_coeff * aux_fun_grad_val.unsqueeze(2)  # dimensions: (N, M, M', d)
        validate_tensor(deep_corr_grad, 'deep_corr_grad')
        deep_corr_grad = - 0.5 * deep_corr_grad

        corr_prelog = self.beta * deep_corr_grad

        out = prelog + corr_prelog
        validate_tensor(out, 'out')

        return out, x_enc_norm, aux_fun_grad_val, corr_prelog

    def log(self, x, y):
        """
        :param x: N x M x d tensor
        :param y: N x M' x d tensor
        :return: N x M x M' x d tensor
        """
        assert len(x.shape) == len(y.shape)
        assert x.shape[0] == y.shape[0]
        assert x.shape[-1] == y.shape[-1] == self.base_manifold.d

        log = self.base_manifold.log(x, y)
        validate_tensor(log, 'log')
        if self.disable_correction:
            return log

        prelog, x_enc_norm, aux_fun_grad_val, _ = self.prelog(x, y)  # prelog dimensions: (N, M, M', d)

        base_hessian = torch.eye(self.base_manifold.d).to(x.device)
        base_hessian = base_hessian.unsqueeze(0).unsqueeze(0)  # dimensions: (1, 1, d, d)
        correction_hessian = self.log_corr_hessian(x_enc_norm, aux_fun_grad_val)  # dimensions: (N, M, d, d)
        hessian = base_hessian + correction_hessian  # dimensions: (N, M, d, d)
        hessian_inv = torch.linalg.inv(hessian)

        log = torch.einsum("NMab,NMLb->NMLa", hessian_inv, prelog)  # dimensions: (N, M, d)

        return log