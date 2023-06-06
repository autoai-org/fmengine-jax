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