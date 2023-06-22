import jax
import jax.numpy as jnp
from fmengine.utils.rng import RNGGen
from fmengine.modelling.language.gpt2.gpt2_config import GPT2Config
from fmengine.modelling.language.gpt2.gpt2_model import FlaxGPT2ForCausalLMModule

model_config = GPT2Config()
model = FlaxGPT2ForCausalLMModule(config=model_config)
batch_size = 16
seq_len = 1024
params = model.init(
    input_ids=jnp.zeros(
        (batch_size, seq_len),
        dtype=jnp.float32,
    ),
    position_ids=jnp.zeros(
        (batch_size, seq_len),
        dtype=jnp.float32,
    ),
    attention_mask=jnp.ones(
        (batch_size, seq_len),
        dtype=jnp.float32,
    ),
    rngs=RNGGen.from_seed(42)(model_config.rng_keys()),
)
x = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)
