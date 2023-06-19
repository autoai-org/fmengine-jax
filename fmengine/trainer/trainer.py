import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as PS
from typing import Callable, TypedDict, Optional
import haiku as hk
from optax import GradientTransformation
from flax.training.train_state import TrainState

from fmengine.utils.rng import RNGGen
from fmengine.trainer._base import BaseTrainer
from fmengine.utils.global_norm import global_norm
from fmengine.trainer.hyperparams import HyperParams
from fmengine.modelling._base import FlaxPreTrainedModel
from fmengine.parallelism.partition import with_sharding_constraint

class ShardedLMTrainer(BaseTrainer):
    def __init__(
        self,
        model: FlaxPreTrainedModel,
        optimizer: GradientTransformation,
        optimizer_info: dict,
        loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray],
        hyperparams: HyperParams,
    ) -> None:

        super().__init__(
            model,
            optimizer,
            optimizer_info,
            loss_fn,
            hyperparams,
        )

    def _init_params(self, rng):
        rng_gen = RNGGen(rng)
        params = self.model.init(
            input_ids=jnp.zeros(
                (self.hyperparams.batch_size, self.hyperparams.seq_len),
                dtype=self.hyperparams.dtype,
            ),
            position_ids=jnp.zeros(
                (self.hyperparams.batch_size, self.hyperparams.seq_len),
                dtype=self.hyperparams.dtype,
            ),
            attention_mask=jnp.ones(
                (self.hyperparams.batch_size, self.hyperparams.seq_len),
                dtype=self.hyperparams.dtype,
            ),
            rngs=rng_gen(self.model.config.rng_keys()),
        )
        return params

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
            lr=self.optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_gen(), metrics