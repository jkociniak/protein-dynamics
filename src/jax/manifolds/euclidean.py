import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass
from typing import Optional
from functools import partial

from .base import Manifold

@dataclass
class Euclidean(Manifold):
    """Class describing Euclidean space of dimension `d`"""
    a: float = 1.0

    def inner(self, p: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x d array
            X: N x M x d array
            Y: N x L x d array
        Returns:
            N x M x L array
        """
        assert (len(p.shape) + 1) == len(X.shape) == len(Y.shape)
        assert p.shape[-1] == X.shape[-1] == Y.shape[-1] == self.d
        assert p.shape[:-1] == X.shape[:-2] == Y.shape[:-2]

        if len(p.shape) > 2:  # N is a tensor
            pp = p.reshape(-1, self.d)
            XX = X.reshape(-1, X.shape[-2], X.shape[-1])
            YY = Y.reshape(-1, Y.shape[-2], Y.shape[-1])
            result = self.inner(pp, XX, YY)
            return result.reshape(p.shape[:-1], X.shape[-2], Y.shape[-2])

        return self.a * jnp.einsum("NMi,NLi->NML", X, Y)

    def pairwise_distance(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x M x d array
            q: N x M' x d array
        Returns:
            N x M x M' array
        """
        assert len(p.shape) == len(q.shape)
        assert p.shape[0] == q.shape[0] and p.shape[-1] == q.shape[-1] == self.d

        diff = jnp.expand_dims(p, -2) - jnp.expand_dims(q, -3)
        return jnp.sqrt(self.a * jnp.sum(diff ** 2, -1) + 1e-8)

    def distance(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x M x d array
            q: N x M x d array
        Returns:
            N x M array
        """
        assert p.shape == q.shape, f'p.shape = {p.shape}, q.shape = {q.shape}'
        assert p.shape[-1] == self.d

        out = jnp.sqrt(self.a * jnp.sum((p - q) ** 2, -1) + 1e-8)
        assert out.shape == p.shape[:-1]
        return out

    def geodesic(self, p: jnp.ndarray, q: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x 1 x d array
            q: N x 1 x d array
            t: M array
        Returns:
            N x M x d array
        """

        def single_example_geodesic(p, q, t):
            """
            Args:
                p: 1 x d array
                q: 1 x d array
                t: M array
            Returns:
                M x d array
            """
            return (1 - t[:, None]) * p + t[:, None] * q

        return jax.vmap(single_example_geodesic, in_axes=(0, 0, None), out_axes=0)(p, q, t)

    def log(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x M x d array
            q: N x M' x d array
        Returns:
            N x M x M' x d array
        """
        assert len(p.shape) == len(q.shape)
        assert p.shape[0] == q.shape[0]
        assert p.shape[-1] == q.shape[-1] == self.d

        res = jnp.expand_dims(q, 1) - jnp.expand_dims(p, 2)
        assert res.shape == (p.shape[0], p.shape[1], q.shape[1], self.d)
        return res

    def exp(self, p: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x d array
            X: N x M x d array
        Returns:
            N x M x d array
        """
        assert (len(p.shape) + 1) == len(X.shape)
        return jnp.expand_dims(p, -2) + X

    def parallel_transport(self, p: jnp.ndarray, X: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x d array
            X: N x M x d array
            q: N x d array
        Returns:
            N x M x d array
        """
        return X

    def metric_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        mt = jnp.eye(self.d)
        mt = mt[None, None]  # dimensions: (1, 1, d, d)
        return mt

    def s_geodesic(self, x: jnp.ndarray, y: jnp.ndarray, tau: jnp.ndarray,
                   base: Optional[jnp.ndarray] = None, step_size: float = 1.,
                   max_iter: int = 100, tol: float = 1e-3,
                   debug: bool = False, print_iterations: bool = False) -> jnp.ndarray:
        """
        Args:
            x: N x 1 x n x d array
            y: N x 1 x n x d array
            tau: M array
            base: Optional base point
        Returns:
            N x M x n x d array
        """
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == 1

        error0 = self.distance(x, y).max() + 1e-6
        z = jnp.ones((1, len(tau), 1, 1)) * y

        def body_fun(carry, _):
            z, relerror = carry

            grad_Wzx = -self.log(z, x)[:, :, 0]
            validate_tensor(grad_Wzx, "grad_Wzx")

            grad_Wzy = -self.log(z, y)[:, :, 0]
            validate_tensor(grad_Wzy, "grad_Wzy")

            grad_Wz = (1 - tau[None, :, None, None]) * grad_Wzx + tau[None, :, None, None] * grad_Wzy

            z_new = z - step_size * grad_Wz
            validate_tensor(z_new, "z_new")

            error = self.norm(z_new, grad_Wz[:, None]).max()
            relerror = error / error0

            if print_iterations:
                print(f"iteration | relerror = {relerror}")

            return (z_new, relerror), None

        init_carry = (z, 1.0)
        (z_final, _), _ = jax.lax.while_loop(
            lambda carry: (carry[1] > tol),
            lambda carry: body_fun(carry, None)[0],
            init_carry
        )

        if base is None:
            base = self.base_point
        final = self.align_mpoint(z_final, base=base)
        return final