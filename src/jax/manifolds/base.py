import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from dataclasses import dataclass


def validate_tensor(tensor, name: str):
    if jnp.isnan(tensor).any():
        raise ValueError(f"NaN in {name}")
    elif jnp.isinf(tensor).any():
        raise ValueError(f"Inf in {name}")


@dataclass
class Manifold(eqx.Module):
    """Base class describing a manifold of dimension `ndim`"""
    d: int

    def barycentre(self, x: jnp.ndarray, tol: float = 1e-3, max_iter: int = 20) -> jnp.ndarray:
        """
        Args:
            x: N x M x Mpoint array
            tol: convergence tolerance
            max_iter: maximum number of iterations
        Returns:
            N x Mpoint array
        """

        def body_fun(carry, _):
            y, _ = carry
            log_mean = jnp.mean(self.log(y, x), axis=1)
            y_new = self.exp(y, log_mean[:, None, :]).squeeze(-2)
            return (y_new, None), None

        y_init = x[:, 0]
        (y_final, _), _ = jax.lax.scan(
            body_fun,
            (y_init, None),
            None,
            length=max_iter
        )
        return y_final

    def inner(self, p: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses should implement this")

    def norm(self, p: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x Mpoint array
            X: N x M x Mpoint array
        Returns:
            N x M array
        """
        p_expanded = p[:, None, :] * jnp.ones((1, X.shape[1], 1))
        return jnp.sqrt(self.inner(p_expanded, X[:, None], X[:, None]).squeeze(-2))

    def distance(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses should implement this")

    def log(self, p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x Mpoint array
            q: N x M x Mpoint array
        Returns:
            N x M x Mpoint array
        """
        raise NotImplementedError("Subclasses should implement this")

    def exp(self, p: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            p: N x Mpoint array
            X: N x M x Mpoint array
        Returns:
            N x M x Mpoint array
        """
        raise NotImplementedError("Subclasses should implement this")

    def geodesic(self,
                 x: jnp.ndarray,
                 y: jnp.ndarray,
                 tau: jnp.ndarray,
                 step_size: float = 1.0,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 debug: bool = False,
                 print_iterations: bool = False) -> Union[jnp.ndarray, Tuple[
        jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]]:
        """
        Args:
            x: N x 1 x d array
            y: N x 1 x d array
            tau: M array
            step_size: float
            max_iter: int
            tol: float
            debug: bool
        Returns:
            N x M x d array or tuple of arrays if debug=True
        """
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == 1

        error0 = self.distance(x, y).max() + 1e-6
        z = jnp.ones((1, len(tau), 1)) * y

        if not debug:
            def body_fun(carry, _):
                z, relerror = carry

                grad_Wzx = -self.log(z, x)[:, :, 0]
                grad_Wzy = -self.log(z, y)[:, :, 0]
                grad_Wz = (1 - tau[None, :, None]) * grad_Wzx + tau[None, :, None] * grad_Wzy

                z_new = z - step_size * grad_Wz
                error = self.norm(z_new, grad_Wz[:, None]).max()
                relerror = error / error0

                return (z_new, relerror), None

            init_carry = (z, 1.0)
            (z_final, _), _ = jax.lax.while_loop(
                lambda carry: (carry[1] > tol),
                lambda carry: body_fun(carry, None)[0],
                init_carry
            )
            return z_final
        else:
            mt_history = []
            z_history = []
            grads_Wzx = []
            grads_Wzy = []
            grads_Wz = []

            relerror = 1.0
            k = 1

            while relerror > tol and k <= max_iter:
                grad_Wzx = -self.log(z, x)[:, :, 0]
                validate_tensor(grad_Wzx, "grad_Wzx")
                grads_Wzx.append(grad_Wzx)

                grad_Wzy = -self.log(z, y)[:, :, 0]
                validate_tensor(grad_Wzy, "grad_Wzy")
                grads_Wzy.append(grad_Wzy)

                grad_Wz = (1 - tau[None, :, None]) * grad_Wzx + tau[None, :, None] * grad_Wzy
                validate_tensor(grad_Wz, "grad_Wz")
                grads_Wz.append(grad_Wz)

                z = z - step_size * grad_Wz

                metric_tensor = self.metric_tensor(z)
                mt_history.append(metric_tensor)
                z_history.append(z)
                validate_tensor(z, "z")

                error = self.norm(z, grad_Wz[:, None]).max()
                relerror = error / error0

                if print_iterations:
                    print(f"{k} | relerror = {relerror}")

                k += 1

            return z, mt_history, z_history, grads_Wzx, grads_Wzy, grads_Wz

    def parallel_transport(self, p: jnp.ndarray, X: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses should implement this")

    def manifold_dimension(self) -> int:
        return self.d