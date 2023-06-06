"""
Trainer class
"""

import os
import jax
import wandb
from jax import numpy as jnp
from typing import Callable, TypedDict

import haiku as hk
from loguru import logger
import orbax.checkpoint as checkpoint
from jax.experimental.pjit import pjit
from optax import GradientTransformation
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from fmtrainer.utils.rng import RNGGen
from fmtrainer.trainer.hyperparams import HyperParams
from fmtrainer.dataloader._base import FMTrainerDataset
from fmtrainer.modelling._base import FlaxPreTrainedModel
from fmtrainer.parallelism.partition import match_partition_rules, make_shard_and_gather_fns

class BaseTrainer():
    def __init__(self,
                 model: FlaxPreTrainedModel,
                 optimizer: GradientTransformation,
                 loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray],
                 hyperparams: HyperParams,
                 scheduler: dict | None = None,
                ) -> None:
        self.model: FlaxPreTrainedModel = model
        self.optimizer: GradientTransformation = optimizer
        self.optimizer_args: dict = scheduler
        self.loss_fn: Callable[[hk.Params, TypedDict], jnp.ndarray] = loss_fn
        self.hyperparams: HyperParams = hyperparams
        self.hyperparams.ckpt_dir = os.path.abspath(self.hyperparams.ckpt_dir)

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
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        )
        run = wandb.init(
            # Set the project where this run will be logged
            project=self.hyperparams.name,
            # Track hyperparameters and run metadata
            config=self.hyperparams)
        mesh = self.model.config.get_jax_mesh(self.hyperparams.mesh_dims)
        with mesh:
            rng = RNGGen.from_seed(self.hyperparams.seed)()
            
            if os.path.exists(self.hyperparams.ckpt_dir) and os.listdir(self.hyperparams.ckpt_dir):
                train_state = self.restore(rng)
            else:
                train_state = self.initialize(rng)
            with progress:
                task = f"[blue] Training <step={self.meta['current_step']}, loss={self.meta['current_loss']:.4f}>"
                train_task = progress.add_task(
                    task,
                    total=self.hyperparams.steps,
                    start=self.meta['current_step']
                )
                for i in range(self.meta['current_step'], self.meta['current_step']+self.hyperparams.steps):
                    batch = next(iter(dataset))
                    train_state, rng, metrics = self.sharded_train_step(train_state, rng, batch)
                    self.meta = {
                        'current_step': i,
                        'current_loss': float(metrics['loss'])
                    }
                    wandb.log({
                        "loss": float(metrics['loss'])
                    })
                    if i > 0:
                        self.ckpt_manager.save(i, items={
                            'train_state': train_state,
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

    def _sharding(self, rng):
        train_state_shapes = jax.eval_shape(self._init_train_state, rng)
        
        self.train_state_partition = match_partition_rules(
            self.model.config.get_partition_rules(), train_state_shapes
        )
        self.shard_fns, self.gather_fns = make_shard_and_gather_fns(
            self.train_state_partition, train_state_shapes
        )
        self.sharded_init_fn = pjit(
            self._init_train_state,
            in_shardings=PS(),
            out_shardings=self.train_state_partition
        )
        self.sharded_train_step = pjit(
            self._train_step,
            in_shardings=(self.train_state_partition, PS(), PS()),
            out_shardings=(self.train_state_partition, PS(), PS()),
            donate_argnums=(0, 1),
        )

    def initialize(self, rng):
        self.meta = {
            'current_step': 0,
            'current_loss': -1
        }
        logger.info(
            f"Cannot load train state from checkpoint, initializing from scratch...")
        self._sharding(rng)
        return self.sharded_init_fn(rng)

    def restore(self, rng, step=-1):
        empty_state = self._init_train_state(rng)
        if step == -1:
            step = self.ckpt_manager.latest_step()
        logger.info(
            f"Restoring from checkpoint {self.hyperparams.ckpt_dir}/{step}...")
        restored = self.ckpt_manager.restore(
            step, items={'train_state': empty_state, 'meta': None})
        self.meta = restored['meta']
        train_state = restored['train_state']
        self._sharding(rng)
        return train_state
    
    def _init_train_state(self, rng):
        # optimizer = self.optimizer.init(params)
        train_state: TrainState = TrainState.create(
            params=self._init_params(rng),
            tx=self.optimizer,
            apply_fn=None,
        )
        return train_state
    
    def _train_step(train_state: TrainState, rng: any, batch: TypedDict) -> dict:
        """
        This function is pjit-ed - it should be stateless
        """
        raise NotImplementedError