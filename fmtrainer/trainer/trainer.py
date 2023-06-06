from jax import numpy as jnp
from typing import Callable, TypedDict, Optional

import haiku as hk
from loguru import logger
from optax import GradientTransformation
from flax.training.train_state import TrainState

from fmtrainer.utils.rng import RNGGen
from fmtrainer.trainer.hyperparams import HyperParams
from fmtrainer.modelling._base import FlaxPreTrainedModel
from fmtrainer.trainer.distributed import DistributedTrainer

class LMTrainer(DistributedTrainer):
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

    def _init_train_state(self, rng):
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
        # optimizer = self.optimizer.init(params)
        train_state: TrainState = TrainState.create(
            params=params,
            tx=self.optimizer,
            apply_fn=None,
        )
        return train_state

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
