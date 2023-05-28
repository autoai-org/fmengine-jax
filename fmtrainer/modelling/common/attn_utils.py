import jax.numpy as jnp

def split_heads(x, n_head):
    # [sequence, features] to [heads, sequence, features]
    m = x.shape[-1]
    split = jnp.reshape(x, x.shape[:-1] + (n_head, m // n_head))
    return jnp.transpose(split, [1, 0, 2])

def merge_heads(x):
    x = jnp.transpose(x, [1, 0, 2])
    *shape, a, b = x.shape
    return jnp.reshape(x, shape + [a * b])

def mask_attn_weights(w, n_past):
    _, nd, ns = w.shape

    i = jnp.arange(nd)[:, None]
    j = jnp.arange(ns)
    b = (i >= j - n_past).astype(w.dtype)
    b = jnp.reshape(b, [1, nd, ns])
    w = w * b - jnp.float32(1e10).astype(w.dtype) * (1 - b)
    return w