import optax
from fmengine.trainer.hyperparams import HyperParams
import jax.numpy as jnp

def adamw(hyper_params: HyperParams):
    optimizer = optax.MultiSteps(
        optax.adamw(
            learning_rate = hyper_params.lr,
            weight_decay = hyper_params.weight_decay,
        ),
        hyper_params.accumulate_gradient_steps
    )

    def learning_rate_schedule(step):
        multiplier = hyper_params.lr / 0.01
        return multiplier / jnp.sqrt(jnp.maximum(step, hyper_params.lr_warmup_steps))

    def weight_decay_schedule(step):
        multiplier = hyper_params.weight_decay / 1e-4
        return -multiplier * jnp.square(learning_rate_schedule(step))

    optimizer_info = dict(
        learning_rate_schedule=learning_rate_schedule,
        weight_decay_schedule=weight_decay_schedule,
    )
    return optimizer, optimizer_info