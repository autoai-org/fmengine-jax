import chex
import jax.numpy as jnp

@chex.dataclass
class HyperParams:
    name: str
    lr: float
    steps: int
    batch_size: int
    lr_warmup_steps: int
    weight_decay: float
    accumulate_gradient_steps: int
    seq_len: int
    seed: int
    ckpt_dir: str
    ckpt_step: int = 100
    ckpt_max_to_keep: int = 3
    # order dp, fsdp, mp
    # default strategy is 1 dp-degree, ${device} fsdp-degree, 1 mp-degree
    mesh_dims: str = '1,-1,1'
    dtype: jnp.dtype = jnp.float32

