import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Any

class Conv1D(nn.Module):
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", jax.nn.initializers.normal(stddev=0.02), (self.features, inputs.shape[-1]))
        kernel = jnp.asarray(kernel.transpose(), self.dtype)
        y = jax.lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        if self.use_bias:
            bias = self.param("bias", jax.nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y