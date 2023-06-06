import jax
import haiku as hk
from jax import numpy as jnp
from typing import Callable, TypedDict
from optax import GradientTransformation
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from jax.experimental.pjit import with_sharding_constraint

from fmtrainer.utils.rng import RNGGen
from fmtrainer.trainer._base import BaseTrainer
from fmtrainer.utils.global_norm import global_norm
from fmtrainer.trainer.hyperparams import HyperParams
from fmtrainer.modelling._base import FlaxPreTrainedModel

class DistributedTrainer(BaseTrainer):
    def __init__(self, 
                 model: FlaxPreTrainedModel, 
                 optimizer: GradientTransformation, 
                 loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray], hyperparams: HyperParams, 
                 scheduler: dict | None = None) -> None:
        super().__init__(model, optimizer, loss_fn, hyperparams, scheduler)

    def _train_step(
        self,
        train_state: TrainState,
        rng: any,
        batch: TypedDict,
    ):
        rng_gen = RNGGen(rng)

        sharded_batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def loss_and_accuracy(params):
            logits = self.model.apply(
                params,
                sharded_batch["input_tokens"],
                deterministic=False,
                rngs=rng_gen(self.model.config.rng_keys()),
            ).logits
            return self.loss_fn(
                logits,
                sharded_batch["target_tokens"],
                None
            )

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=self.hyperparams.lr,
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_gen(), metrics