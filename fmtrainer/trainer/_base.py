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
import orbax.checkpoint as checkpoint
from optax import GradientTransformation
from flax.training.train_state import TrainState
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from typing import Any
from fmtrainer.utils.rng import RNGGen
from fmtrainer.utils.global_norm import global_norm
from fmtrainer.trainer.hyperparams import HyperParams
from fmtrainer.parallelism.device import get_jax_mesh
from fmtrainer.dataloader._base import FMTrainerDataset
from fmtrainer.modelling._base import FlaxPreTrainedModel
from fmtrainer.parallelism.partition import match_partition_rules, make_shard_and_gather_fns


class FMTrainer():
    def __init__(self,
                 model: FlaxPreTrainedModel,
                 optimizer: GradientTransformation,
                 loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray],
                 hyperparams: HyperParams,
                 scheduler: Optional[dict] = None,
                 ) -> None:

        self.model: FlaxPreTrainedModel = model
        self.optimizer: GradientTransformation = optimizer
        self.optimizer_args: dict = scheduler
        self.loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray] = loss_fn
        self.hyperparams: HyperParams = hyperparams
        self.hyperparams.ckpt_dir = os.path.abspath(self.hyperparams.ckpt_dir)
        self.jax_rng: RNGGen = RNGGen.from_seed(self.hyperparams.seed)

        options = checkpoint.CheckpointManagerOptions(
            save_interval_steps=self.hyperparams.ckpt_step,
            max_to_keep=self.hyperparams.ckpt_max_to_keep,
            create=True,
        )
        self.ckpt_manager = checkpoint.CheckpointManager(
            self.hyperparams.ckpt_dir, {
                'train_state': checkpoint.AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler()),
                'meta': checkpoint.Checkpointer(checkpoint.JsonCheckpointHandler()),
            },
            options
        )
        self.meta = {
            'current_step': -1,
            'current_loss': -1
        }

    def fit(self, dataset: FMTrainerDataset):
        mesh = self.model.config.get_jax_mesh(self.hyperparams.mesh_dims)
        with mesh:
            if os.path.exists(self.hyperparams.ckpt_dir) and os.listdir(self.hyperparams.ckpt_dir):
                self.restore()
            else:
                self.initialize()
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        )
        with progress:
            task = f"[blue] Training <step={self.meta['current_step']}, loss={self.meta['current_loss']:.4f}>"
            train_task = progress.add_task(
                task,
                total=self.hyperparams.steps,
                start=self.meta['current_step']
            )
            for i in range(self.meta['current_step'], self.meta['current_step']+self.hyperparams.steps):
                batch = next(iter(dataset))
                metrics = self.sharded_train_step(batch)
                self.meta = {
                    'current_step': i,
                    'current_loss': float(metrics['loss'])
                }
                if i > 0:
                    self.ckpt_manager.save(i, items={
                        'train_state': self.train_state,
                        'meta': self.meta
                    })
                task_description = f"[blue] Training <step={self.meta['current_step']}, loss={self.meta['current_loss']:.4f}>"
                progress.update(
                    task_id=train_task,
                    description=task_description,
                    advance=1,
                )
        logger.info(f"Training finished at step {self.meta['current_step']}")
        logger.info(f"Final loss: {self.meta['current_loss']:.4f}")
        logger.info(f"Waiting for background processes to finish...")
        self.ckpt_manager.wait_until_finished()

    def restore(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def _train_step(self, batch: TypedDict) -> dict:
        raise NotImplementedError

    def _init_train_state(self) -> TrainState:
        raise NotImplementedError