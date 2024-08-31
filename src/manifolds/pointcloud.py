import torch
from torch.func import vmap, jacfwd, functional_call
import logging
from torch_geometric.data import Data, Batch

from src.models.metric_correction import PCSingleExampleWrapper
from .base import Manifold
from ..utils.tensor import gradients, validate_tensor, gradients_pc


class PointCloudManifold(torch.nn.Module):
    """
    Manifold of point clouds
    """
    def __init__(self, n, d, base=None, alpha=0.1):
        """

        :param d: integer d
        :param n: integer n
        :param base: n x d tensor
        :param alpha: float32
        """
        super().__init__()
        #assert n >= d + 1
        self.d = d
        self.n = n

        self.manifold_dimension = int(self.d * self.n - self.d * (self.d + 1) / 2)
        self.vert_space_dimension = int(self.d * (self.d + 1) / 2)

        if base is None:
            self.has_base_point = False
        else:
            print(base.shape)
            print('wooho')
            assert len(base.shape) == 2
            print(base.shape[0], self.n)
            print(base.shape[1], self.d)
            assert base.shape[0] == self.n and base.shape[1] == self.d
            self.has_base_point = True
            self.base_point = self.center_mpoint(base[None, None]).squeeze()

        self.alpha = alpha

    @staticmethod
    def translate_mpoint(x, t):
        """
        Compute t . (x_1, ..., x_n) := (x_1 - t, ..., x_n - t)
        :param x: N x M x n x d tensor
        :param t: N x M x d tensor
        :return: N x M x n x d tensor
        """
        return x + t[:, :, None, :]

    @classmethod
    def center_mpoint(cls, x):
        """
        :param x: N x M x n x d tensor
        :return: N x M x n x d tensor
        """
        t = torch.mean(x, 2)
        return cls.translate_mpoint(x, -t)

    @classmethod
    def gyration_matrix(cls, x):
        """
        :param x: N x M x n x d
        :return: N x M x d x d
        """
        xc = cls.center_mpoint(x)
        return torch.einsum("NMia,NMib->NMab", xc, xc)

    @staticmethod
    def pairwise_distance_internal(x):
        """
        Function that computes the pairwise distance matrix of a point cloud x
        :param x: N x M x n x d tensor
        :return: N x M x n x n tensor with values [(\|x_i - x_j\|^2)_ij]_NM
        """
        x_gram_mat = torch.einsum("NMia,NMja->NMij", x, x)

        x_gram_diag = torch.diagonal(x_gram_mat, dim1=2, dim2=3)
        x_gram_diag_mat = torch.einsum("NMi,j->NMij", x_gram_diag, torch.ones((x.shape[2],), device=x.device))

        return x_gram_diag_mat - 2 * x_gram_mat + torch.transpose(x_gram_diag_mat, 2, 3)

    @staticmethod
    def orthogonal_transform_mpoint(x, O):
        """
        Compute O . (x_1, ..., x_n) := (O x_1, ..., O x_n)
        :param x: N x M x n x d tensor
        :param O: N x M x d x d tensor
        :return:
        """
        return torch.einsum("NMba,NMia->NMib", O, x)

    def least_orthogonal(self, x, base=None):
        """
        Solve inf_O \sum_i \|y_i - Ox_i\|^2, i.e., how to rotate x such that it's closest to y:=base
        :param x: N x M x n x d tensor
        :param base: n x d tensor
        :return: N x M x d x d tensor
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point

        inertia_tensor = torch.einsum("NMia,ib->NMab", x, base) / self.n
        svd = torch.svd(inertia_tensor)

        O = torch.einsum("NMcb,NMab->NMca", svd.V, svd.U)
        return O

    def align_mpoint(self, x, base=None):
        """
        :param x: N x M x n x d tensor
        :param base: n x d tensor
        :return: N x M x n x d tensor
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point

        base_ = self.center_mpoint(base[None, None]).squeeze()
        xc = self.center_mpoint(x)
        O = self.least_orthogonal(xc, base=base_)
        return self.orthogonal_transform_mpoint(xc, O)

    def pairwise_distance(self, x, y):
        """
        Manifold distance between points x and y
        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :return: N x M x M' tensor
        """
        assert x.shape[0] == y.shape[0]  # batch size must be equal

        x_pairwise_distances = self.pairwise_distance_internal(x)
        x_pairwise_distances += torch.eye(self.n, device=x.device)  # for numerical stability in log

        y_pairwise_distances = self.pairwise_distance_internal(y)
        y_pairwise_distances += torch.eye(self.n, device=y.device)  # for numerical stability in log

        predists = (1 / 2 * torch.log(
            x_pairwise_distances[:, :, None, :, :] / y_pairwise_distances[:, None, :, :, :])) ** 2  # dimensions N x M x M' x n x n

        # alpha * correction term
        x_gyration = self.gyration_matrix(x)
        y_gyration = self.gyration_matrix(y)

        corrections = torch.log(torch.det(x_gyration[:, :, None, :, :]) / torch.det(y_gyration[:, None, :, :, :])) ** 2

        return torch.sqrt(1 / 2 * torch.sum(predists, [3, 4]) + self.alpha * corrections)  # factor 1/2 in first term because we count everything double

    def distance(self, x, y):
        """
        Manifold distance between points x and y
        :param x: N x M x n x d tensor
        :param y: N x M x n x d tensor
        :return: N x M tensor
        """
        assert x.shape[0] == y.shape[0]  # batch size must be equal

        x_pairwise_distances = self.pairwise_distance_internal(x)
        x_pairwise_distances += torch.eye(self.n, device=x.device)  # for numerical stability in log

        y_pairwise_distances = self.pairwise_distance_internal(y)
        y_pairwise_distances += torch.eye(self.n, device=y.device)  # for numerical stability in log

        predists = (1 / 2 * torch.log(x_pairwise_distances / y_pairwise_distances)) ** 2  # dimension N x M x n x n

        # alpha * correction term
        x_gyration = self.gyration_matrix(x)
        y_gyration = self.gyration_matrix(y)

        corrections = torch.log(torch.det(x_gyration) / torch.det(y_gyration)) ** 2

        return torch.sqrt(1 / 2 * torch.sum(predists, [2, 3]) + self.alpha * corrections)  # factor 1/2 in first term because we count everything double

    def metric_tensor(self, x, asmatrix=False):
        """

        :param x: N x M x n x d tensor
        :param asmatrix:
        :return: N x M x n x n x d x d tensor or N x M x nd x nd tensor if asmatrix==True
        """
        N = x.shape[0]
        M = x.shape[1]

        x_pairwise_distances = self.pairwise_distance_internal(x)
        x_pairwise_distances += torch.eye(self.n)  # for numerical stability in log

        xixj = x[:, :, :, None] - x[:, :, None, :]
        xij = torch.einsum("NMija,NMijb->NMijab", xixj, xixj)
        A = - xij \
            / x_pairwise_distances[:, :, :, :, None, None] ** 2
        # fix diagonal
        Adiag = - torch.sum(A, 3).permute(0, 1, 3, 4, 2)
        A += torch.diag_embed(Adiag).permute(0, 1, 4, 5, 2, 3)

        # alpha * correction term
        xc = self.center_mpoint(x)
        x_gyration = self.gyration_matrix(x)

        L, Q = torch.linalg.eigh(x_gyration)
        yi = torch.einsum("NMab,NMb,NMcb,NMic->NMia", Q, 1 / L, Q, xc)

        B = 4 * torch.einsum("NMia,NMjb->NMijab", yi, yi)

        if asmatrix:
            return (A + self.alpha * B).permute(0, 1, 2, 5, 3, 4).reshape(N, M, self.n * self.d, self.n * self.d)
        else:
            return A + self.alpha * B

    def inner(self, x, X, Y):
        """

        :param x: N x M x n x d tensor
        :param X: N x M x L x n x d tensor
        :param Y: N x M x K x n x d tensor
        :return: N x M x L x K tensor
        """
        assert x.shape[0] == X.shape[0] == Y.shape[0]

        H = self.metric_tensor(x)  # dimensions N x M x n x n x d x d
        inner = torch.einsum("NMijab,NMLia,NMKjb->NMLK", H, X, Y)

        return inner

    def norm(self, x, X):
        """

        :param x: N x M x n x d tensor
        :param X: N x M x L x n x d tensor
        :return: N x M x L tensor
        """
        assert x.shape[0] == X.shape[0]

        N = x.shape[0]
        M = x.shape[1]
        L = X.shape[2]

        norm = torch.zeros(N, M, L)
        for l in range(L):
            norm[:, :, l] = torch.sqrt(
                self.inner(x, X[:, :, l, :, :][:, :, None, :, :], X[:, :, l, :, :][:, :, None, :, :])[:, :, 0, 0])

        return norm

    def prelog(self, x, y, asvector=False):
        """
        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :param asvector:
        :return: N x M x M' x n x d tensor
        """
        assert x.shape[0] == y.shape[0]
        N = x.shape[0]
        M = x.shape[1]
        MM = y.shape[1]

        x_pairwise_distances = self.pairwise_distance_internal(x)
        x_pairwise_distances += torch.eye(self.n)  # for numerical stability in log
        #validate_tensor(x_pairwise_distances, 'x_pairwise_distances')

        y_pairwise_distances = self.pairwise_distance_internal(y)
        y_pairwise_distances += torch.eye(self.n)  # for numerical stability in log
        validate_tensor(y_pairwise_distances, 'y_pairwise_distances')

        predists = 1 / 2 * torch.log(x_pairwise_distances[:, :, None, :, :] / y_pairwise_distances[:, None, :, :, :])
        validate_tensor(predists, 'predists')

        xixj = x[:, :, None, :, None] - x[:, :, None, None, :]

        prelogs = - predists[:, :, :, :, :, None] \
                  * xixj / x_pairwise_distances[:, :, None, :, :, None]
        validate_tensor(prelogs, 'prelog')

        prelog = torch.sum(prelogs, 4)


        # alpha * correction term
        x_gyration = self.gyration_matrix(x)
        y_gyration = self.gyration_matrix(y)

        precorrections = torch.log(torch.det(x_gyration[:, :, None, :, :]) / torch.det(y_gyration[:, None, :, :, :]))

        xc = self.center_mpoint(x)
        L, Q = torch.linalg.eigh(x_gyration)
        gxi = torch.einsum("NMab,NMb,NMcb,NMic->NMia", Q, 1 / L, Q, xc)

        prelogcorrections = - 2 * gxi[:, :, None] * precorrections[:, :, :, None, None]

        if torch.isnan(prelogcorrections).any() or torch.isinf(prelogcorrections).any():
            raise Exception("prelogcorrections has nans/infs")

        if asvector:
            return (prelog + self.alpha * prelogcorrections).reshape(N, M, MM, self.n * self.d)
        else:
            return prelog + self.alpha * prelogcorrections

    def log(self, x, y, asvector=False):
        """

        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :param asvector:
        :return: N x M x M' x n x d tensor
        """
        assert x.shape[0] == y.shape[0]
        N = x.shape[0]
        M = x.shape[1]
        MM = y.shape[1]

        prelog = self.prelog(x, y, asvector=True)
        #print('prelog max: ', prelog.max())
        if torch.isnan(prelog).any() or torch.isinf(prelog).any():
            raise Exception("prelog has nans/infs")

        H = self.metric_tensor(x, asmatrix=True)
        L, Q = torch.linalg.eigh(H)

        vertical_dim = self.vert_space_dimension
        log = torch.einsum("NMxy,NMy,NMzy,NMLz->NMLx",
                           Q[:, :, :, vertical_dim:], 1/L[:, :, vertical_dim:], Q[:, :, :, vertical_dim:], prelog)

        # log = torch.zeros((N, M, MM, self.n * self.d))
        # for m in range(M):
        #     for mm in range(MM):
        #         log[:, m, mm] = torch.linalg.lstsq(H[:, m], prelog[:, m, mm]).solution

        if asvector:
            return log
        else:
            return log.reshape(N, M, MM, self.n, self.d)

    def s_mean(self, x, x0=None, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False):
        """
        :param x: N x M x n x d tensor
        :param x0: N x 1 x n x d tensor
        :return: N x 1 x n x d tensor
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point

        if x0 is not None:
            z = x0
            pws_mat = self.pairwise_distance(x, z) ** 2
            error0 = torch.sqrt(torch.sum(pws_mat, 1).min(1).values.max()) + 1e-6
        else:  # not recommended for combinations of large M and n
            pws_mat = self.pairwise_distance(x, x) ** 2
            error0 = torch.sqrt(torch.sum(pws_mat, 1).min(1).values.max()) + 1e-6
            z = x[:, torch.argmin(torch.sum(pws_mat, 1), 1)]   # pick conformation with least distance squared to every other point
        relerror = 1.
        k = 1
        while relerror > tol and k <= max_iter:
            # compute grad
            grad_Wz = - torch.mean(self.log(z, x), 2)
            z = z - step_size * grad_Wz
            error = self.norm(z, grad_Wz[:, :, None]).max()
            relerror = error / error0
            if debug:
                print(f"{k} | relerror = {relerror}")

            k = k + 1

        return self.align_mpoint(z, base=base)

    def s_geodesic(self, x, y, tau, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False, print_iterations=False):
        """

        :param x: N x 1 x n x d tensor
        :param y: N x 1 x n x d tensor
        :param tau: M tensor
        :return: N x M x n x d tensor
        """

        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == 1

        error0 = self.distance(x, y).max() + 1e-6
        relerror = 1.
        k = 1
        z = torch.ones(len(tau))[None, :, None, None] * y
        while relerror > tol and k <= max_iter:
            # compute grad
            grad_Wzx = - self.log(z, x)[:, :, 0]
            if torch.isnan(grad_Wzx).any():
                raise Exception(f"grad_Wzx has nans after {k} iterations")
            #print(f'grad_Wzx max: {grad_Wzx.max()}')


            grad_Wzy = - self.log(z, y)[:, :, 0]
            if torch.isnan(grad_Wzy).any():
                raise Exception(f"grad_Wzy has nans after {k} iterations")
            #print(f'grad_Wzy max: {grad_Wzy.max()}')

            grad_Wz = (1 - tau[None, :, None, None]) * grad_Wzx + tau[None, :, None, None] * grad_Wzy
            #print(f'grad_Wz max: {grad_Wz.max()}')
            # update z
            z = z - step_size * grad_Wz
            # check if z has nans
            if torch.isnan(z).any():
                raise Exception(f"z has nans after {k} iterations")

            # compute new error
            error = self.norm(z, grad_Wz[:, None]).max()
            relerror = error / error0
            if print_iterations:
                print(f"{k} | relerror = {relerror}")

            k = k + 1

        base = self.base_point if base is None else base
        final = self.align_mpoint(z, base=base)
        return final

    def s_exp(self, x, X, c=1 / 4, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False):
        """

        :param x: N x 1 x n x d tensor
        :param X: N x 1 x n x d tensor
        :return: N x 1 x n x d tensor
        """
        if base is not None:
            K = int(c * self.norm(x, X[:, :, None]).max()) + 1
            print(f"computing exp in {K} steps")
            x0 = x
            x1 = x + 1 / K * X

            k = 1
            xkk = x0
            xk = x1
            while k < K:
                x_new = self.s_geodesic(xkk, xk, torch.tensor([2.]), step_size=step_size, max_iter=max_iter, tol=tol,
                                        debug=debug)
                xkk = xk
                xk = x_new
                k = k + 1

            return self.align_mpoint(xk, base=base)
        else:
            return self.s_exp(x, X, c=c, base=self.base_point, step_size=step_size, max_iter=max_iter, tol=tol,
                              debug=debug)

    # def orthonormal_basis(self, x, asvector=False):
    #     """
    #
    #     :param x: N x M x n x d tensor
    #     :param asvector:
    #     :return: N x M x L x n x d tensor with L:= manifold_dimension
    #     """
    #     N = x.shape[0]
    #     M = x.shape[1]
    #
    #     H = self.metric_tensor(x, asmatrix=True)
    #     L, Q = torch.linalg.eigh(H)
    #
    #     vertical_dim = self.vert_space_dimension
    #     horizontal_vectors = Q.permute(0, 1, 3, 2)[:, :, vertical_dim:]
    #     rescaling_factors = 1 / torch.sqrt(L[:, :, vertical_dim:])
    #     horizontal_vectors = rescaling_factors[:, :, :, None] * horizontal_vectors
    #     if asvector:
    #         return horizontal_vectors
    #     else:
    #         return horizontal_vectors.reshape(N, M, self.manifold_dimension, self.n, self.d)
    #
    # def coordinates_in_basis(self, x, X, Xi):
    #     """
    #     compute coefficients c of X in basis \Xi, i.e., X = c^i \Xi_i
    #     :param x: N x M x n x d tensor
    #     :param X: N x M x n x d tensor
    #     :param Xi: N x M x L x n x d
    #     :return: N x M x L tensor with L:= manifold_dimension
    #     """
    #     return self.inner(x, X[:, :, None], Xi)[:, :, 0]
    #
    # def tvector_in_basis(self, c, Xi):
    #     """
    #     compute tvector X from coordinates c in basis \Xi, i.e., X = c^i \Xi_i
    #     :param c: N x M x L tensor with L:= manifold_dimension
    #     :param Xi: N x M x L x n x d tensor
    #     :return: N x M x n x d tensor
    #     """
    #     return torch.einsum("NML,NMLia->NMia", c, Xi)
    #
    # def horizontal_projection_tvector(self, x, X):
    #     """
    #
    #     :param x: N x M x n x d tensor
    #     :param X: N x M x M' x n x d tensor
    #     :return: N x M x M' x n x d tensor
    #     """
    #     assert x.shape[0] == X.shape[0] and x.shape[1] == X.shape[1]
    #     N = x.shape[0]
    #     M = x.shape[1]
    #
    #     xc = self.center_mpoint(x)
    #     x_gyration = self.gyration_matrix(x)
    #     L, Q = torch.linalg.eigh(x_gyration)
    #
    #     vertical_basis = torch.zeros((N, M, self.vert_space_dimension, self.n, self.d))
    #
    #     for i in range(self.d):
    #         ei = torch.zeros(self.d)
    #         ei[i] = 1.
    #         vi = ei[None] * torch.ones(N, M, self.n)[:, :, :, None]
    #         vertical_basis[:, :, i] = 1 / self.n ** (1 / 2) * vi
    #         for j in range(self.d):
    #             if j > i:
    #                 Gij = torch.zeros((self.d, self.d))
    #                 Gij[i, j] = 1
    #                 Gij[j, i] = -1
    #                 Ld = torch.einsum('ab,NMa->NMab', torch.eye(self.d), L ** (-1 / 2))
    #                 QLGijLQt = Q @ Ld @ Gij[None, None] @ Ld @ Q.transpose(2, 3)
    #                 normalisation = torch.sqrt((L[:, :, i] * L[:, :, j]) / (L[:, :, i] + L[:, :, j]))[:, :, None, None]
    #                 ind = self.d + int((self.d * (self.d - 1) / 2) - (self.d - i) * ((self.d - i) - 1) / 2 + j - i - 1)
    #                 vij = torch.einsum("NMab,NMib->NMia", normalisation * QLGijLQt, xc)
    #                 vertical_basis[:, :, ind] = vij
    #
    #                 # project X onto vertical space
    #     VX_inner = torch.einsum("NMVia,NMLia->NMVL", vertical_basis, X)
    #     Vproj_X = torch.einsum("NMVL,NMVia->NMLia", VX_inner, vertical_basis)
    #
    #     return X - Vproj_X


class CorrectedPointCloudManifold(PointCloudManifold):
    def __init__(self, correction_encoder, *args, metric='l2', beta=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.align_base_point()

        self.correction_encoder = correction_encoder
        self.disable_correction = False

        assert metric in ['l2', 'log']
        self.metric = metric

        assert isinstance(beta, float)
        assert beta >= 0.
        self.beta = beta

    def align_base_point(self):
        # constuct rotation matrix
        rot_xz = torch.zeros(3, 3)
        rot_xz[2, 0] = 1.
        rot_xz[1, 1] = 1.
        rot_xz[0, 2] = -1.
        self.base_point = torch.einsum("ba,ia->ib", rot_xz, self.base_point)

        rot_xy = torch.zeros(3, 3)
        theta = torch.tensor([- torch.pi * 1 / 3])
        rot_xy[0, 0] = torch.cos(theta)
        rot_xy[0, 1] = - torch.sin(theta)
        rot_xy[1, 0] = torch.sin(theta)
        rot_xy[1, 1] = torch.cos(theta)
        rot_xy[2, 2] = 1.
        self.base_point = torch.einsum("ba,ia->ib", rot_xy, self.base_point)

    def s_distance(self, x, y):
        """
        Manifold distance between points x and y
        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :return: N x M x M' tensor
        """

        base_dists = super().s_distance(x, y)  # dimensions: (N, M, M')

        if self.disable_correction:
            return base_dists

        corr1 = self.correction_encoder.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
        corr2 = self.correction_encoder.forward_2d_batch(y)  # dimensions: (N, M', enc_dim)
        corr1 = corr1.unsqueeze(2)  # dimensions: (N, M, 1, enc_dim)
        corr2 = corr2.unsqueeze(1)  # dimensions: (N, 1, M', enc_dim)

        if self.metric == 'l2':
            corrs = torch.norm(corr1 - corr2, dim=3)  # dimensions: (N, M, M')
        elif self.metric == 'log':
            corrnorm1 = torch.norm(corr1, dim=3) + 1  # dimensions: (N, M, 1)
            corrnorm2 = torch.norm(corr2, dim=3) + 1  # dimensions: (N, 1, M')
            corrs = torch.log(corrnorm1 / corrnorm2) ** 2  # dimensions: (N, M, M')
        else:
            raise ValueError(f'Unknown metric: {self.metric}')

        corrected_dists = torch.sqrt(base_dists ** 2 + self.beta * corrs ** 2)  # dimensions: (N, M, M')
        return corrected_dists

    def s_prelog(self, x, y, asvector=False):
        """
        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :param asvector:
        :return: N x M x M' x n x d tensor
        """
        prelog = super().s_prelog(x, y, asvector=asvector)

        if torch.isnan(prelog).any() or torch.isinf(prelog).any():
            raise ValueError("NaN or Inf in prelog")

        if self.disable_correction:
            return prelog

        N = x.shape[0]
        M = x.shape[1]
        MM = y.shape[1]
        # 1. Compute representations of x and y
        x_rep = self.correction_encoder.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
        y_rep = self.correction_encoder.forward_2d_batch(y)  # dimensions: (N, M', enc_dim)

        # 2. Compute jacobian of correction encoder at x
        # to use vmap we need a function that does not expect batch dimension
        single_example_encoder = PCSingleExampleWrapper(self.correction_encoder)

        def encoder_call(params, x):
            return functional_call(single_example_encoder, params, x)

        # compute jacobian using functional operators
        jacfun = jacfwd(encoder_call, argnums=1)
        batch_jacfun = vmap(jacfun, in_dims=(None, 0))  # we don't vectorize over param dimension
        batch_2d_jacfun = vmap(batch_jacfun, in_dims=(None, 0))

        model_params = dict(single_example_encoder.named_parameters())
        enc_jacobian = batch_2d_jacfun(model_params, x)  # dimensions: (N, M, enc_dim, n, d)
        enc_jacobian = enc_jacobian.unsqueeze(2)  # dimensions: (N, M, 1, enc_dim, n, d)

        #print(f'max of enc_jacobian: {enc_jacobian.max()}')

        if self.metric == 'l2':
            diffs = x_rep[:, :, None, :] - y_rep[:, None, :, :]  # dimensions: (N, M, M', enc_dim)
            deep_corr_grad = torch.einsum("NMLa,NMLabc->NMLbc", 2 * diffs, enc_jacobian)
        elif self.metric == 'log':
            x_rep_norm = torch.norm(x_rep, dim=3, keepdim=True)   # dimensions: (N, M, 1)
            y_rep_norm = torch.norm(y_rep, dim=3).unsqueeze(1)  # dimensions: (N, 1, M')
            scalar_coeff = 2 * torch.log((x_rep_norm + 1) / (y_rep_norm + 1)) / (x_rep_norm ** 2 + x_rep_norm)  # dimensions: (N, M, M')
            deep_corr_grad = torch.einsum("NML,NMLa,NMLabc->NMLbc", scalar_coeff, x_rep.unsqueeze(2), enc_jacobian)
        else:
            raise ValueError(f'Unknown metric: {self.metric}')

        if torch.isnan(deep_corr_grad).any() or torch.isinf(deep_corr_grad).any():
            raise ValueError("NaN or Inf in deep_corr_grad")

        if asvector:
            out = prelog + (self.beta * deep_corr_grad).reshape(N, M, MM, self.n * self.d)
        else:
            out = prelog + self.beta * deep_corr_grad

        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("NaN or Inf in out")

        return out


class L2CorrectedPointCloudManifold(Manifold):
    """
    Manifold of point clouds
    """
    def __init__(self, correction_encoder, alpha, beta, base_manifold_params):
        """
        :param dim: integer d
        :param numpoints: integer n
        :param base: n x d tensor
        :param alpha: float32
        """
        n, d = base_manifold_params['n'], base_manifold_params['d']
        #assert n >= d + 1, "There must be more nodes than node dimension"
        super().__init__(d=n*d)

        self.base_manifold = PointCloudManifold(**base_manifold_params)
        self.correction_encoder = correction_encoder

        assert isinstance(beta, float) and beta >= 0.
        self.beta = torch.tensor([beta], dtype=torch.float32)

        assert isinstance(alpha, float) and alpha >= 0.
        self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def distance(self, x, y, input_as_vector=False):
        """
        Manifold distance between points x and y
        :param x: N x M x n x d tensor
        :param y: N x M x n x d tensor
        :return: N x M tensor
        """
        assert x.shape[0] == y.shape[0]  # batch size must be equal
        assert x.shape[1] == y.shape[1]  # number of points must be equal

        N = x.shape[0]
        M = x.shape[1]

        if input_as_vector:
            x = x.view(N, M, self.base_manifold.n, self.base_manifold.d)
            y = y.view(N, M, self.base_manifold.n, self.base_manifold.d)

        base_dists = self.base_manifold.distance(x, y)  # dimensions: (N, M)

        # correction encoder is from R^(n x d) to R^encdim
        # print(f'x shape: {x.shape}')
        # print(f'N: {N}, M: {M}')
        # print(f'base manifold n: {self.base_manifold.n}, d: {self.base_manifold.d}')
        x_enc, _ = self.correction_encoder(x.reshape(N * M, self.base_manifold.n * self.base_manifold.d))
        x_enc = x_enc.reshape(N, M, -1)

        y_enc, _ = self.correction_encoder(y.reshape(N * M, self.base_manifold.n * self.base_manifold.d))
        y_enc = y_enc.reshape(N, M, -1)

        deep_dists = torch.linalg.norm(x_enc - y_enc, dim=2)  # dimensions: (N, M)

        corrected_dists = torch.sqrt(self.alpha * base_dists ** 2 + self.beta * deep_dists ** 2)
        return corrected_dists

    def forward(self, x, y):
        return self.distance(x, y)

    def metric_tensor(self, x, enc_grad=None, asmatrix=False):
        """
        :param x: N x M x n x d tensor
        :param asmatrix:
        :return: N x M x n x n x d x d tensor or N x M x nd x nd tensor if asmatrix==True
        """
        N, M, n, d = x.shape
        base_mt = self.base_manifold.metric_tensor(x, asmatrix=False)  # dimensions N x M x n x n x d x d

        if enc_grad is None:
            x_enc, coords = self.correction_encoder(x.view(N * M, n * d))
            enc_grad = gradients(x_enc, coords)  # dimensions (N * M, encdim, n * d)
            enc_grad = enc_grad.reshape(N, M, x_enc.shape[-1], n, d)

        correction_mt = 2 * torch.einsum('NMija,NMikb->NMjkab', enc_grad, enc_grad)  # dimensions: (N, M, n, n, d, d)
        mt = self.alpha * base_mt + self.beta * correction_mt  # dimensions: (N, M, n, n, d, d)

        if asmatrix:
            return mt.permute(0, 1, 2, 5, 3, 4).reshape(N, M, n * d, n * d)
        else:
            return mt

    def norm(self, x, X):
        """

        :param x: N x M x n x d tensor
        :param X: N x M x L x n x d tensor
        :return: N x M x L tensor
        """
        assert x.shape[0] == X.shape[0]

        N = x.shape[0]
        M = x.shape[1]
        L = X.shape[2]

        norm = torch.zeros(N, M, L)
        for l in range(L):
            norm[:, :, l] = torch.sqrt(
                self.inner(x, X[:, :, l, :, :][:, :, None, :, :], X[:, :, l, :, :][:, :, None, :, :])[:, :, 0, 0])

        return norm

    def inner(self, x, X, Y):
        """

        :param x: N x M x n x d tensor
        :param X: N x M x L x n x d tensor
        :param Y: N x M x K x n x d tensor
        :return: N x M x L x K tensor
        """
        assert x.shape[0] == X.shape[0] == Y.shape[0]

        H = self.metric_tensor(x)
        inner = torch.einsum("NMijab,NMLia,NMKjb->NMLK", H, X, Y)

        return inner

    def prelog(self, x, y, asvector=False):
        """
        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :param asvector:
        :return: N x M x M' x n x d tensor
        """
        assert x.shape[0] == y.shape[0]
        N, M, n, d = x.shape
        MM = y.shape[1]

        prelog = self.base_manifold.prelog(x, y, asvector=False)  # dimensions: (N, M, M', n, d)

        x_enc, coords = self.correction_encoder(x.reshape(-1, n * d))  # dimensions: (N*M, enc_dim)
        y_enc, _ = self.correction_encoder(y.reshape(-1, n * d))  # dimensions: (N*M', enc_dim)

        validate_tensor(x_enc, "x_enc")
        validate_tensor(coords, "coords")

        enc_grad = gradients(x_enc, coords)  # dimension (N*M, enc_dim, n * d)

        validate_tensor(enc_grad, "enc_grad")

        enc_grad = enc_grad.reshape(N, M, x_enc.shape[-1], -1)  # dimensions: (N, M, enc_dim, n * d)

        x_enc = x_enc.reshape(N, M, 1, -1)
        y_enc = y_enc.reshape(N, 1, M, -1)
        diffs = x_enc - y_enc  # dimensions: (N, M, M', enc_dim)
        enc_grad = enc_grad.unsqueeze(2)  # dimensions: (N, M, 1, enc_dim, n*d)

        validate_tensor(diffs, "diffs")

        # 3. combine to get correction gradient
        deep_corr_grad = 2 * torch.einsum('NMKij,NMKi->NMKj', enc_grad, diffs)  # dimensions: (N, M, M', n*d)

        corr_prelog = - 0.5 * deep_corr_grad.reshape(N, M, MM, n, d)

        out = self.alpha * prelog + self.beta * corr_prelog  # dimensions: (N, M, M', n, d)

        enc_grad = enc_grad.reshape(N, M, -1, n, d)
        if asvector:
            return out.reshape(N, M, MM, -1), None, enc_grad, corr_prelog
        else:
            return out, None, enc_grad, corr_prelog

    def log(self, x, y, asvector=False):
        """

        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :param asvector:
        :return: N x M x M' x n x d tensor
        """
        assert x.shape[0] == y.shape[0]
        N = x.shape[0]
        M = x.shape[1]
        MM = y.shape[1]

        prelog, _, enc_grad, _ = self.prelog(x, y, asvector=True)  # dimensions: (N, M, M', n*d)
        #print('prelog max: ', prelog.max())
        if torch.isnan(prelog).any() or torch.isinf(prelog).any():
            raise Exception("prelog has nans/infs")

        H = self.metric_tensor(x, enc_grad=enc_grad, asmatrix=True)  # dimensions: (N, M, n*d, n*d)
        L, Q = torch.linalg.eigh(H)  # dimensions: (N, M, n*d), (N, M, n*d, n*d)

        vertical_dim = self.base_manifold.vert_space_dimension
        log = torch.einsum("NMxy,NMy,NMzy,NMLz->NMLx",
                           Q[:, :, :, vertical_dim:], 1/L[:, :, vertical_dim:], Q[:, :, :, vertical_dim:], prelog)

        # log = torch.zeros((N, M, MM, self.n * self.d))
        # for m in range(M):
        #     for mm in range(MM):
        #         log[:, m, mm] = torch.linalg.lstsq(H[:, m], prelog[:, m, mm]).solution

        if asvector:
            return log
        else:
            return log.reshape(N, M, MM, self.base_manifold.n, self.base_manifold.d)

    def s_geodesic(self, x, y, tau, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False, print_iterations=False):
        """

        :param x: N x 1 x n x d tensor
        :param y: N x 1 x n x d tensor
        :param tau: M tensor
        :return: N x M x n x d tensor
        """

        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == 1

        #print(f'x shape: {x.shape}, y shape: {y.shape}, tau shape: {tau.shape}')
        error0 = self.distance(x, y).max() + 1e-6
        relerror = 1.
        k = 1
        z = torch.ones(len(tau))[None, :, None, None] * y
        while relerror > tol and k <= max_iter:
            # compute grad
            grad_Wzx = - self.log(z, x)[:, :, 0]
            if torch.isnan(grad_Wzx).any():
                raise Exception(f"grad_Wzx has nans after {k} iterations")
            #print(f'grad_Wzx max: {grad_Wzx.max()}')


            grad_Wzy = - self.log(z, y)[:, :, 0]
            if torch.isnan(grad_Wzy).any():
                raise Exception(f"grad_Wzy has nans after {k} iterations")
            #print(f'grad_Wzy max: {grad_Wzy.max()}')

            grad_Wz = (1 - tau[None, :, None, None]) * grad_Wzx + tau[None, :, None, None] * grad_Wzy
            #print(f'grad_Wz max: {grad_Wz.max()}')
            # update z
            z = z - step_size * grad_Wz
            # check if z has nans
            if torch.isnan(z).any():
                raise Exception(f"z has nans after {k} iterations")

            # compute new error
            error = self.norm(z, grad_Wz[:, None]).max()
            relerror = error / error0
            if print_iterations:
                print(f"{k} | relerror = {relerror}")

            k = k + 1

        base = self.base_manifold.base_point if base is None else base
        final = self.base_manifold.align_mpoint(z, base=base)
        return final


class L2CorrectedPointCloudManifoldWE(Manifold):
    """
    Manifold of point clouds
    """
    def __init__(self, correction_encoder, alpha, beta, base_manifold_params):
        """
        :param dim: integer d
        :param numpoints: integer n
        :param base: n x d tensor
        :param alpha: float32
        """
        n, d = base_manifold_params['n'], base_manifold_params['d']
        #assert n >= d + 1, "There must be more nodes than node dimension"
        super().__init__(d=n*d)

        self.base_manifold = PointCloudManifold(**base_manifold_params)
        self.correction_encoder = correction_encoder

        assert isinstance(beta, float) and beta >= 0.
        self.beta = torch.tensor([beta], dtype=torch.float32)

        assert isinstance(alpha, float) and alpha >= 0.
        self.alpha = torch.tensor([alpha], dtype=torch.float32)

        self.edge_index = None

        #DEBUG
        self.printed = False
        self.z_pos_history = []

    def distance(self, graphs_x, graphs_y, return_base_dists=False):
        """
        Manifold distance between points x and y
        :param graphs_x: Batch from PyTorch Geometric (N elements)
        :param graphs_y: Batch from PyTorch Geometric (N elements)
        :return: 1 x N tensor
        """
        assert isinstance(graphs_x, Batch) and isinstance(graphs_y, Batch)
        assert graphs_x.num_graphs == graphs_y.num_graphs
        n_graphs = graphs_x.num_graphs

        x = graphs_x.pos.reshape(1, n_graphs, self.base_manifold.n, self.base_manifold.d)
        y = graphs_y.pos.reshape(1, n_graphs, self.base_manifold.n, self.base_manifold.d)

        print(x.shape)
        base_dists = self.base_manifold.distance(x, y)  # dimensions: (1, n_graphs)

        # correction encoder is from R^(n x d) to R^encdim
        x_enc, _ = self.correction_encoder(graphs_x)
        x_enc = x_enc[None]  # dimensions (1, n_graphs, enc_dim)

        y_enc, _ = self.correction_encoder(graphs_y)
        y_enc = y_enc[None]  # dimensions (1, n_graphs, enc_dim)

        deep_dists = torch.linalg.norm(x_enc - y_enc, dim=2)  # dimensions: (1, n_graphs)

        if return_base_dists:
            return deep_dists, base_dists
        else:
            corrected_dists = torch.sqrt(self.alpha * base_dists ** 2 + self.beta * deep_dists ** 2)
            return corrected_dists

    def forward(self, x, y):
        return self.distance_graph(x, y)

    def metric_tensor(self, graphs, enc_grad=None, asmatrix=False):
        """
        :param graphs: Batch from PyTorch Geometric (M elements)
        :param asmatrix:
        :return: M x n x n x d x d tensor or M x nd x nd tensor if asmatrix==True
        """
        assert isinstance(graphs, Batch)
        x = graphs.pos.reshape(1, graphs.num_graphs, self.base_manifold.n, self.base_manifold.d)
        base_mt = self.base_manifold.metric_tensor(x, asmatrix=False)  # dimensions 1 x M x n x n x d x d
        base_mt = base_mt.squeeze(0)

        if enc_grad is None:
            x_enc, coords = self.correction_encoder(graphs)
            enc_grad = gradients_pc(x_enc, coords)  # dimensions (n_graphs, encdim, n, d)

        correction_mt = 2 * torch.einsum('Mija,Mikb->Mjkab', enc_grad, enc_grad)  # dimensions: (M, n, n, d, d)

        # print(f'Rescaled base_mt: {self.alpha * base_mt}')
        # print(f'Rescaled correction_mt: {self.beta * correction_mt}')
        if not self.printed:
            print(f'Difference: {self.alpha * base_mt - self.beta * correction_mt}')
            self.printed = True
        mt = self.alpha * base_mt + self.beta * correction_mt  # dimensions: (M, n, n, d, d)

        if asmatrix:
            return mt.permute(0, 1, 3, 2, 4).reshape(graphs.num_graphs, self.base_manifold.n * self.base_manifold.d,
                                                     self.base_manifold.n * self.base_manifold.d)
        else:
            return mt

    def norm(self, graph_x, X):
        """

        :param x: Batch from PyTorch Geometric (1 element)
        :param X: n x d tensor
        :return: scalar tensor
        """
        assert isinstance(graph_x, Batch)
        assert graph_x.num_graphs == 1

        inner = self.inner(graph_x, X[None], X[None])
        norm = torch.sqrt(inner)

        return norm

    def inner(self, x, X, Y):
        """

        :param x: Batch from PyTorch Geometric (M elements)
        :param X: M x n x d tensor
        :param Y: M x n x d tensor
        :return: M tensor
        """
        assert isinstance(x, Batch)
        assert x.num_graphs == X.shape[0] == Y.shape[0], f'{x.num_graphs}, {X.shape[0]}, {Y.shape[0]}'

        H = self.metric_tensor(x)  # dim (M, n, n, d, d)
        inner = torch.einsum("Mijab,Mia,Mjb->M", H, X, Y)

        return inner

    def prelog(self, graphs_x, graphs_y, asvector=False):
        """
        :param x: Batch from PyTorch Geometric (N elements)
        :param y: Batch from PyTorch Geometric (N' elements)
        :param asvector:
        :return: 1 x N x N' x n x d tensor
        """
        assert isinstance(graphs_x, Batch) and isinstance(graphs_y, Batch)
        n_graphs = graphs_x.batch.max().item() + 1
        nn_graphs = graphs_y.batch.max().item() + 1

        x = graphs_x.pos.reshape(1, n_graphs, self.base_manifold.n, self.base_manifold.d)
        y = graphs_y.pos.reshape(1, nn_graphs, self.base_manifold.n, self.base_manifold.d)

        prelog = self.base_manifold.prelog(x, y, asvector=False)  # dimensions: (1, N, N', n, d)

        x_enc, coords = self.correction_encoder(graphs_x)  # dimensions: (N*M, enc_dim)
        y_enc, _ = self.correction_encoder(graphs_y)  # dimensions: (N*M', enc_dim)

        validate_tensor(x_enc, "x_enc")
        validate_tensor(coords, "coords")

        enc_grad = gradients_pc(x_enc, coords)  # dimension (n_graphs, enc_dim, n, d)

        validate_tensor(enc_grad, "enc_grad")

        enc_grad = enc_grad.reshape(1, n_graphs, x_enc.shape[-1], -1)  # dimensions: (N, M, enc_dim, n * d)

        x_enc = x_enc.reshape(1, n_graphs, 1, -1)
        y_enc = y_enc.reshape(1, 1, nn_graphs, -1)
        diffs = x_enc - y_enc  # dimensions: (1, N, N', enc_dim)
        enc_grad = enc_grad.unsqueeze(2)  # dimensions: (N, M, 1, enc_dim, n*d)

        validate_tensor(diffs, "diffs")

        # 3. combine to get correction gradient
        deep_corr_grad = 2 * torch.einsum('NMKij,NMKi->NMKj', enc_grad, diffs)  # dimensions: (N, M, M', n*d)

        corr_prelog = - 0.5 * deep_corr_grad.reshape(1, n_graphs, nn_graphs, self.base_manifold.n, self.base_manifold.d)

        out = self.alpha * prelog + self.beta * corr_prelog  # dimensions: (N, M, M', n, d)

        enc_grad = enc_grad.reshape(1, n_graphs, -1, self.base_manifold.n, self.base_manifold.d)
        if asvector:
            return out.reshape(1, n_graphs, nn_graphs, -1), None, enc_grad, corr_prelog
        else:
            return out, None, enc_grad, corr_prelog

    def log(self, graphs_x, graphs_y, asvector=False):
        """

        :param x: Batch from PyTorch Geometric (N elements)
        :param y: Batch from PyTorch Geometric (N' elements)
        :param asvector:
        :return: 1 x M x M' x n x d tensor
        """
        assert isinstance(graphs_x, Batch) and isinstance(graphs_y, Batch)
        n_graphs = graphs_x.batch.max().item() + 1
        nn_graphs = graphs_y.batch.max().item() + 1
        assert n_graphs == 1 and nn_graphs == 1

        prelog, _, enc_grad, _ = self.prelog(graphs_x, graphs_y, asvector=True)  # dimensions: (N, M, M', n*d)
        #print('prelog max: ', prelog.max())
        if torch.isnan(prelog).any() or torch.isinf(prelog).any():
            raise Exception("prelog has nans/infs")

        H = self.metric_tensor(graphs_x, enc_grad=enc_grad.squeeze(0), asmatrix=True)  # dimensions: (N, M, n*d, n*d)
        H = H[None]
        L, Q = torch.linalg.eigh(H)  # dimensions: (N, M, n*d), (N, M, n*d, n*d)

        vertical_dim = self.base_manifold.vert_space_dimension
        log = torch.einsum("NMxy,NMy,NMzy,NMLz->NMLx",
                           Q[:, :, :, vertical_dim:], 1/L[:, :, vertical_dim:], Q[:, :, :, vertical_dim:], prelog)

        # log = torch.zeros((N, M, MM, self.n * self.d))
        # for m in range(M):
        #     for mm in range(MM):
        #         log[:, m, mm] = torch.linalg.lstsq(H[:, m], prelog[:, m, mm]).solution

        if asvector:
            return log
        else:
            return log.reshape(1, n_graphs, nn_graphs, self.base_manifold.n, self.base_manifold.d)

    def s_geodesic_graph(self, graphs_x, graphs_y, tau, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False, print_iterations=False):
        """

        :param x: Batch from PyTorch Geometric (1 element)
        :param y: Batch from PyTorch Geometric (1 element)
        :param tau: float
        :return: n x d tensor
        """

        assert isinstance(graphs_x, Batch) and isinstance(graphs_y, Batch)
        if self.edge_index is None:
            self.edge_index = graphs_x.edge_index
        else:
            assert torch.equal(self.edge_index, graphs_x.edge_index), "Manifold has different edge index than provided graph. Set to None or provide the same edge index."
        n_graphs = graphs_x.num_graphs
        nn_graphs = graphs_y.num_graphs
        num_nodes = graphs_x.num_nodes
        assert n_graphs == 1 and nn_graphs == 1

        #print(f'x shape: {x.shape}, y shape: {y.shape}, tau shape: {tau.shape}')
        error0 = self.distance(graphs_x, graphs_y).max() + 1e-6
        relerror = 1.
        k = 1
        z_pos = graphs_x.pos * (1 - tau) + graphs_y.pos * tau
        z = Data(pos=z_pos, x=torch.ones(num_nodes, 1), edge_index=self.edge_index)
        z = Batch.from_data_list([z])

        z_pos_hist = []
        while relerror > tol and k <= max_iter:
            z_pos_hist.append(z.pos.detach().clone())
            # compute grad
            grad_Wzx = - self.log(z, graphs_x).squeeze((0, 1, 2))
            if torch.isnan(grad_Wzx).any():
                raise Exception(f"grad_Wzx has nans after {k} iterations")
            #print(f'grad_Wzx max: {grad_Wzx.max()}')

            grad_Wzy = - self.log(z, graphs_y).squeeze((0, 1, 2))
            if torch.isnan(grad_Wzy).any():
                raise Exception(f"grad_Wzy has nans after {k} iterations")
            #print(f'grad_Wzy max: {grad_Wzy.max()}')

            grad_Wz = (1 - tau) * grad_Wzx + tau * grad_Wzy  # dim (n, d)
            #print(f'grad_Wz max: {grad_Wz.max()}')
            # update z positions
            z.pos = z.pos - step_size * grad_Wz

           # check if z has nans
            if torch.isnan(z.pos).any():
                raise Exception(f"z has nans after {k} iterations")

            # compute new pos error
            # gWz = Data(pos=, x=torch.ones(num_nodes), edge_index=self.edge_index)
            # gWz = Batch.from_data_list([gWz])
            error = self.norm(z, grad_Wz).max()
            relerror = error / error0
            if print_iterations:
                print(f"{k} | relerror = {relerror}")

            k = k + 1

        self.z_pos_history.append(z_pos_hist)

        base = self.base_manifold.base_point if base is None else base
        final = self.base_manifold.align_mpoint(z.pos[None, None], base=base)
        final = final.squeeze((0, 1))
        return final
