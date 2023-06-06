
import os
import jax
import chex
from jax import numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from typing import Callable, TypedDict, Optional
from jax.experimental.pjit import with_sharding_constraint

import haiku as hk
from loguru import logger
from optax import GradientTransformation
from flax.training.train_state import TrainState

from fmtrainer.utils.rng import RNGGen
from fmtrainer.utils.global_norm import global_norm
from fmtrainer.trainer.hyperparams import HyperParams
from fmtrainer.modelling._base import FlaxPreTrainedModel
from fmtrainer.parallelism.partition import match_partition_rules, make_shard_and_gather_fns
from fmtrainer.trainer._base import BaseTrainer


class LMTrainer(BaseTrainer):
    def __init__(
        self,
        model: FlaxPreTrainedModel,
        optimizer: GradientTransformation,
        loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray],
        hyperparams: HyperParams,
        scheduler: Optional[dict] = None,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            loss_fn,
            hyperparams,
            scheduler
        )

    def _train_step(
        self,
        train_state: TrainState,
        rng: any,
        batch: TypedDict,
    ):
        rng_gen = RNGGen(rng)

        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        
        def loss_and_accuracy(params):
            logits = self.model.apply(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=rng_gen(self.model.config.rng_keys()),
            ).logits
            return self.loss_fn(
                logits,
                batch["target_tokens"],
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

    def _init_train_state(self):
        self.params = self.model.init(
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
            rngs=self.jax_rng(self.model.config.rng_keys()),
        )
        self.optimizer_state = self.optimizer.init(self.params)
        self.train_state: TrainState = TrainState.create(
            params=self.params,
            tx=self.optimizer,
            apply_fn=None,
        )
        return self.train_state

    def restore(self, step=-1):
        empty_state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(
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
                rngs=self.jax_rng(self.model.config.rng_keys()),
            ),
            tx=self.optimizer,
        )
        if step == -1:
            step = self.ckpt_manager.latest_step()

        logger.info(
            f"Restoring from checkpoint {self.hyperparams.ckpt_dir}/{step}...")

        restored = self.ckpt_manager.restore(
            step, items={'train_state': empty_state, 'meta': None})

        self.train_state = restored['train_state']
        self.meta = restored['meta']
        self.params = self.train_state.params
        self.optimizer_state = self.train_state.opt_state
