import chex
import jax.numpy as jnp

@chex.dataclass
class HyperParams:
    name: str
    lr: float
    steps: int
    batch_size: int
    warmup_steps: int
    seq_len: int
    seed: int
    ckpt_dir: str
    ckpt_step: int = 100
    ckpt_max_to_keep: int = 3
    mesh_dims: str = '1,-1,1'
    dtype: jnp.dtype = jnp.float32

