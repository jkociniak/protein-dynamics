import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ReLULayer(eqx.Module):
    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]

    def __init__(self, in_features: int, out_features: int, *, use_bias: bool = True,
                 key: Optional[jax.random.PRNGKey] = None):
        key_w, key_b = jax.random.split(key) if key is not None else (None, None)
        scale = 1.0 / in_features
        self.weight = jax.random.uniform(key_w, (out_features, in_features), minval=-scale, maxval=scale)
        self.bias = jax.random.uniform(key_b, (out_features,), minval=-scale, maxval=scale) if use_bias else None

    def __call__(self, x):
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return jax.nn.relu(x)


class GeLULayer(eqx.Module):
    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]

    def __init__(self, in_features: int, out_features: int, *, use_bias: bool = True,
                 key: Optional[jax.random.PRNGKey] = None):
        key_w, key_b = jax.random.split(key) if key is not None else (None, None)
        scale = 1.0 / in_features
        self.weight = jax.random.uniform(key_w, (out_features, in_features), minval=-scale, maxval=scale)
        self.bias = jax.random.uniform(key_b, (out_features,), minval=-scale, maxval=scale) if use_bias else None

    def __call__(self, x):
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return jax.nn.gelu(x)


class SineLayer(eqx.Module):
    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    omega_0: float

    def __init__(self, in_features: int, out_features: int, *, is_first: bool = False,
                 omega_0: float = 30.0, use_bias: bool = True, key: Optional[jax.random.PRNGKey] = None):
        key_w, key_b = jax.random.split(key) if key is not None else (None, None)
        self.omega_0 = omega_0

        if is_first:
            scale = 1.0 / in_features
        else:
            scale = jnp.sqrt(6.0 / in_features) / omega_0

        self.weight = jax.random.uniform(key_w, (out_features, in_features), minval=-scale, maxval=scale)
        self.bias = jax.random.uniform(key_b, (out_features,), minval=-scale, maxval=scale) if use_bias else None

    def __call__(self, x):
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return jnp.sin(self.omega_0 * x)


class FourierEmbedding(eqx.Module):
    B: jnp.ndarray
    sigma: float

    def __init__(self, input_dim: int, output_dim: int, *, sigma: float = 1.0,
                 key: Optional[jax.random.PRNGKey] = None):
        self.sigma = sigma
        self.B = self.sigma * jax.random.normal(key, (output_dim, input_dim))

    def __call__(self, x):
        Bx = x @ self.B.T
        return jnp.concatenate([jnp.cos(2 * jnp.pi * Bx), jnp.sin(2 * jnp.pi * Bx)], axis=-1)


@dataclass
class CoordNetSmall(eqx.Module):
    embedding: eqx.Module
    hidden1: eqx.Module
    hidden2: eqx.Module
    output: eqx.Module
    emb_features: int

    def __init__(self, in_features: int, out_features: int, emb_features: int, hidden_features: int, *,
                 embedding_type: str = 'relu',
                 embedding_params: Optional[Dict[str, Any]] = None,
                 first_hidden_type: str = 'relu',
                 first_hidden_params: Optional[Dict[str, Any]] = None,
                 second_hidden_type: str = 'relu',
                 second_hidden_params: Optional[Dict[str, Any]] = None,
                 key: Optional[jax.random.PRNGKey] = None):

        keys = jax.random.split(key, 4) if key is not None else (None,) * 4
        embedding_params = embedding_params or {}
        first_hidden_params = first_hidden_params or {}
        second_hidden_params = second_hidden_params or {}

        # Embedding layer
        if embedding_type == 'relu':
            self.embedding = ReLULayer(in_features, emb_features, key=keys[0])
            self.emb_features = emb_features
        elif embedding_type == 'gelu':
            self.embedding = GeLULayer(in_features, emb_features, key=keys[0])
            self.emb_features = emb_features
        elif embedding_type == 'sine':
            omega_0 = embedding_params.get('omega_0', 30.0)
            self.embedding = SineLayer(in_features, emb_features, is_first=True,
                                       omega_0=omega_0, key=keys[0])
            self.emb_features = emb_features
        elif embedding_type == 'fourier':
            sigma = embedding_params.get('sigma', 1.0)
            self.embedding = FourierEmbedding(in_features, emb_features, sigma=sigma, key=keys[0])
            self.emb_features = 2 * emb_features
        else:
            raise ValueError(f'Invalid embedding type: {embedding_type}')

        # First hidden layer
        if first_hidden_type == 'relu':
            self.hidden1 = ReLULayer(self.emb_features, hidden_features, key=keys[1])
        elif first_hidden_type == 'gelu':
            self.hidden1 = GeLULayer(self.emb_features, hidden_features, key=keys[1])
        elif first_hidden_type == 'sine':
            omega_0 = first_hidden_params.get('omega_0', 30.0)
            self.hidden1 = SineLayer(self.emb_features, hidden_features, omega_0=omega_0, key=keys[1])
        else:
            raise ValueError(f'Invalid first hidden type: {first_hidden_type}')

        # Second hidden layer
        if second_hidden_type == 'relu':
            self.hidden2 = ReLULayer(hidden_features, hidden_features, key=keys[2])
        elif second_hidden_type == 'gelu':
            self.hidden2 = GeLULayer(hidden_features, hidden_features, key=keys[2])
        elif second_hidden_type == 'sine':
            omega_0 = second_hidden_params.get('omega_0', 30.0)
            self.hidden2 = SineLayer(hidden_features, hidden_features, omega_0=omega_0, key=keys[2])
        else:
            raise ValueError(f'Invalid second hidden type: {second_hidden_type}')

        # Output layer
        self.output = eqx.nn.Linear(hidden_features, out_features, key=keys[3])

    def __call__(self, coords):
        x = self.embedding(coords)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x, coords


# Example usage
def create_model(config):
    key = jax.random.PRNGKey(0)  # You might want to make this configurable
    return CoordNetSmall(
        in_features=config['in_features'],
        out_features=config['out_features'],
        emb_features=config['emb_features'],
        hidden_features=config['hidden_features'],
        embedding_type=config['embedding_type'],
        embedding_params=config['embedding_params'],
        first_hidden_type=config['first_hidden_type'],
        first_hidden_params=config['first_hidden_params'],
        second_hidden_type=config['second_hidden_type'],
        second_hidden_params=config['second_hidden_params'],
        key=key
    )