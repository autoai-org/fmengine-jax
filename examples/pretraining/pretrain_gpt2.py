import jax
import optax
import portpicker
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

from fmengine.trainer.trainer import ShardedLMTrainer, HyperParams
from fmengine.nn.losses import cross_entropy_loss_and_accuracy
from fmengine.nn.optimizers import adamw
from fmengine.modelling.language.gpt2.gpt2_config import GPT2Config
from fmengine.dataloader.jsonl_reader import JSONLDatasetForAutoRegressiveModel
from fmengine.modelling.language.gpt2.gpt2_model import FlaxGPT2ForCausalLMModule

model_config = GPT2Config()
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

# Create a HyperParams object
hyper_params = HyperParams(
    name="gpt2-falcon-refinedweb",
    lr=1e-4,
    steps=50000,
    batch_size=2,
    lr_warmup_steps=1000,
    weight_decay=0.0001,
    accumulate_gradient_steps=4,
    seq_len=1024,
    seed=42,
    ckpt_dir=".cache/gpt2/",
    ckpt_step=5000,
    ckpt_max_to_keep=3,
    dtype=jnp.int32,
)

# create optimizer
optimizer, optimizer_info = adamw(hyper_params)

port = portpicker.pick_unused_port()
jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

trainer = ShardedLMTrainer(
    model=FlaxGPT2ForCausalLMModule(config=model_config),
    optimizer=optimizer,
    optimizer_info=optimizer_info,
    loss_fn=cross_entropy_loss_and_accuracy,
    hyperparams=hyper_params,
)

# create dataset
dataset = load_dataset("tiiuae/falcon-refinedweb",
                       split="train",
                       streaming=True).shuffle(buffer_size=10_000, seed=42)

dataset = JSONLDatasetForAutoRegressiveModel(
    dataset=dataset,
    seq_len=1024,
    doc_separator="",
    batch_size=2,
    tokenizer=tokenizer,
    field='content',
)

# fit the model
trainer.fit(dataset)
