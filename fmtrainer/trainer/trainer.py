"""
Trainer class
"""
import os
import jax
import chex
from jax import numpy as jnp
from typing import Callable, TypedDict, Optional

import haiku as hk
from loguru import logger
import orbax.checkpoint as checkpoint
from optax import GradientTransformation
from flax.training.train_state import TrainState
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from fmtrainer.utils.rng import RNGGen
from fmtrainer.utils.global_norm import global_norm
from fmtrainer.dataloader._base import FMTrainerDataset
from fmtrainer.modelling._base import FlaxPreTrainedModel

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
        if os.path.exists(self.hyperparams.ckpt_dir) and os.listdir(self.hyperparams.ckpt_dir):
            self.restore()
        else:
            self.initialize()
        
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
            task_description = f"[blue] Training <step={self.meta['current_step']}, loss={self.meta['current_loss']:.4f}>"
            train_task = progress.add_task(
                task_description,
                total=self.hyperparams.steps,
                start=self.meta['current_step']
            )
            for i in range(self.meta['current_step'], self.meta['current_step']+self.hyperparams.steps):
                batch = next(iter(dataset))
                metrics = self._train_step(batch)
                self.meta = {
                    'current_step': i,
                    'current_loss': float(metrics['loss'])
                }
                if i>0:
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
                
        logger.info("Training Finished! Waiting for background processes to finish...")
        self.ckpt_manager.wait_until_finished()
    
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
        
        logger.info(f"Restoring from checkpoint {self.hyperparams.ckpt_dir}/{step}...")
        
        restored = self.ckpt_manager.restore(step, items={'train_state': empty_state, 'meta': None})
        
        self.train_state = restored['train_state']
        self.meta = restored['meta']
        self.params = self.train_state.params
        self.optimizer_state = self.train_state.opt_state

    def initialize(self):
        self.meta = {
            'current_step': 0,
            'current_loss': -1
        }
        logger.info(f"Cannot load train state from checkpoint, initializing from scratch...")
        
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