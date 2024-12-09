import jax.numpy as jnp

def validate_tensor(tensor, name):
    if jnp.isnan(tensor).any():
        raise ValueError(f"NaN in {name}")
    elif jnp.isinf(tensor).any():
        raise ValueError(f"Inf in {name}")