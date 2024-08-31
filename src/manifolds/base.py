import torch
from ..utils.tensor import validate_tensor


class Manifold(torch.nn.Module):
    """ Base class describing a manifold of dimension `ndim` """
    def __init__(self, d):
        super().__init__()
        self.d = d

    def barycentre(self, x, tol=1e-3, max_iter=20):
        """

        :param x: N x M x Mpoint
        :return: N x Mpoint
        """
        k = 0
        rel_error = 1.
        y = x[:, 0]
        while k <= max_iter and rel_error >= tol:
            y = self.exp(y, torch.mean(self.log(y, x), 1).unsqueeze(-2)).squeeze(-2)
            k += 1

        return y

    def inner(self, p, X, Y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def norm(self, p, X):
        """

        :param p: N x Mpoint
        :param X: N x M x Mpoint
        :return: N x M
        """
        return torch.sqrt(self.inner(p.unsqueeze(-2) * torch.ones((1, X.shape[-2], 1)),
                                     X.unsqueeze(-2), X.unsqueeze(-2)).squeeze(-2))

    def distance(self, p, q):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def log(self, p, q):
        """

        :param p: N x Mpoint
        :param q: N x M x Mpoint
        :return: N x M x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def exp(self, p, X):
        """

        :param p: N x Mpoint
        :param X: N x M x Mpoint
        :return: N x M x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def geodesic(self, x, y, tau, step_size=1., max_iter=100, tol=1e-3, debug=False, print_iterations=False):
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
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == 1

        error0 = self.distance(x, y).max() + 1e-6
        relerror = 1.
        k = 1
        z = torch.ones(len(tau))[None, :, None].to(x.device) * y
        mt_history = []
        z_history = []
        grads_Wzx = []
        grads_Wzy = []
        grads_Wz = []
        while relerror > tol and k <= max_iter:
            # compute grad
            grad_Wzx = - self.log(z, x)[:, :, 0]
            if debug:
                validate_tensor(grad_Wzx, "grad_Wzx")
                grads_Wzx.append(grad_Wzx)

            grad_Wzy = - self.log(z, y)[:, :, 0]
            if debug:
                grads_Wzy.append(grad_Wzy)
                validate_tensor(grad_Wzy, "grad_Wzy")

            grad_Wz = (1 - tau[None, :, None]) * grad_Wzx + tau[None, :, None] * grad_Wzy
            if debug:
                grads_Wz.append(grad_Wz)
                validate_tensor(grad_Wz, "grad_Wz")

            # update z
            z = z - step_size * grad_Wz
            if debug:
                metric_tensor = self.metric_tensor(z)
                mt_history.append(metric_tensor)
                z_history.append(z)
                validate_tensor(z, "z")

            # compute new error
            error = self.norm(z, grad_Wz[:, None]).max()
            relerror = error / error0

            if print_iterations:
                print(f"{k} | relerror = {relerror}")

            k = k + 1
        if debug:
            return z, mt_history, z_history, grads_Wzx, grads_Wzy, grads_Wz
        return z

    def parallel_transport(self, p, X, q):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def manifold_dimension(self):
        return self.d