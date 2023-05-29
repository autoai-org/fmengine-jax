import optax
import jax.numpy as jnp
from fmtrainer.trainer.trainer import Trainer, HyperParams
from fmtrainer.nn.losses import cross_entropy_loss_and_accuracy
from fmtrainer.modelling.language.gpt2.gpt2_config import GPT2Config
from fmtrainer.modelling.language.gpt2.gpt2_model import FlaxGPT2ForCausalLMModule

model_config = GPT2Config()

# Create a hyperparameters
hyper_params = HyperParams(
    lr=1e-4,
    steps=10,
    batch_size=8,
    warmup_steps=0,
    seq_len=1024,
    seed=42,
    dtype=jnp.float32,
)

# create optimizer
optimizer = optax.sgd(1e-4)

trainer = Trainer(
    model=FlaxGPT2ForCausalLMModule(config=model_config),
    optimizer=optimizer,
    loss_fn=cross_entropy_loss_and_accuracy,
    hyperparams=hyper_params,
)