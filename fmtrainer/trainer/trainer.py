import jax
import chex
import haiku as hk
from tqdm import tqdm
from jax import numpy as jnp
from typing import Callable, TypedDict
from optax import GradientTransformation
from flax.training.train_state import TrainState
from fmtrainer.modelling._base import FlaxPreTrainedModel

@chex.dataclass
class HyperParams:
    lr: float
    steps: int
    batch_size: int
    warmup_steps: int

class Trainer:
    def __init__(
            self,
            model: FlaxPreTrainedModel,
            optimizer: GradientTransformation,
            loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray],
            hyperparams: HyperParams
        ) -> None:
        self.model: FlaxPreTrainedModel = model
        self.optimizer: GradientTransformation = optimizer
        self.loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray] = loss_fn
        self.hyperparams: HyperParams = hyperparams
        self.train_state: TrainState = None
        self.params = self.model.init()

    def _train_step(
        self,
        batch: TypedDict,
    ):
        pass

    def fit(
            self,
            batch_gen: Callable[[], TypedDict],
        ):
        for i in range(self.hyperparams.steps):
            step_loss = self._train_step(batch_gen())