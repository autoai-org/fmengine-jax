"""
Trainer class
"""
import os
import jax
import chex
from jax import numpy as jnp
from typing import Callable, TypedDict, Optional

import haiku as hk
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import orbax.checkpoint as checkpoint
from optax import GradientTransformation
from flax.training.train_state import TrainState

from fmtrainer.utils.rng import RNGGen
from fmtrainer.utils.global_norm import global_norm
from fmtrainer.modelling._base import FlaxPreTrainedModel
from fmtrainer.dataloader._base import FMTrainerDataset

@chex.dataclass
class HyperParams:
    lr: float
    steps: int
    batch_size: int
    warmup_steps: int
    seq_len: int
    seed: int
    ckpt_dir: str
    ckpt_step: int = 100
    ckpt_max_to_keep: int = 3
    dtype: jnp.dtype = jnp.float32

class Trainer:
    def __init__(
        self,
        model: FlaxPreTrainedModel,
        optimizer: GradientTransformation,
        loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray],
        hyperparams: HyperParams,
        scheduler: Optional[dict]=None,
    ) -> None:
        self.model: FlaxPreTrainedModel = model
        self.optimizer: GradientTransformation = optimizer
        self.optimizer_args: dict = scheduler
        self.loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray] = loss_fn
        self.hyperparams: HyperParams = hyperparams
        self.jax_rng: RNGGen = RNGGen.from_seed(self.hyperparams.seed)
        options = checkpoint.CheckpointManagerOptions(save_interval_steps=self.hyperparams.ckpt_step, max_to_keep=self.hyperparams.ckpt_max_to_keep)
        # if ckpt_dir is not empty, raise error
        if os.path.exists(self.hyperparams.ckpt_dir) and os.listdir(self.hyperparams.ckpt_dir):
            raise ValueError(
                f"Checkpoint directory {self.hyperparams.ckpt_dir} is not empty."
            )
        os.makedirs(self.hyperparams.ckpt_dir, exist_ok=True)
        
        self.ckpt_manager = checkpoint.CheckpointManager(
            self.hyperparams.ckpt_dir,
            checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler()),
            options
        )

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
        self.train_state: TrainState = TrainState.create(
            params=self.params, tx=optimizer, apply_fn=None
        )

    def _train_step(
        self,
        batch: TypedDict,
    ):
        def loss_and_accuracy(params):
            logits = self.model.apply(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=self.jax_rng(self.model.config.rng_keys()),
            ).logits
            return self.loss_fn(
                logits, batch["target_tokens"], None
            )

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(self.train_state.params)
        self.train_state = self.train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=self.hyperparams.lr,
            gradient_norm=global_norm(grads),
            param_norm=global_norm(self.train_state.params),
        )
        return metrics
    
    def fit(
        self,
        dataset: FMTrainerDataset,
    ):
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        )
        with progress:
            train_task = progress.add_task("[blue]Training...", total=self.hyperparams.steps)
            for i in range(self.hyperparams.steps):
                batch = next(iter(dataset))
                metrics = self._train_step(batch)
                if i>0:
                    self.ckpt_manager.save(i, self.train_state)
                progress.update(
                    task_id=train_task,
                    description=f"Training [loss={metrics['loss']:.4f}]",
                    advance=1,
                )