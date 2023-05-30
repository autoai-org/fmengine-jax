import optax
import jax.numpy as jnp

from transformers import AutoTokenizer
from datasets import load_dataset

from fmtrainer.trainer.trainer import Trainer, HyperParams
from fmtrainer.nn.losses import cross_entropy_loss_and_accuracy
from fmtrainer.modelling.language.gpt2.gpt2_config import GPT2Config
from fmtrainer.dataloader.jsonl_reader import JSONLDatasetForAutoRegressiveModel
from fmtrainer.modelling.language.gpt2.gpt2_model import FlaxGPT2ForCausalLMModule

model_config = GPT2Config()
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

# Create a HyperParams object
hyper_params = HyperParams(
    lr=1e-4,
    steps=5000,
    batch_size=2,
    warmup_steps=0,
    seq_len=1024,
    seed=42,
    ckpt_dir=".cache/checkpoints",
    ckpt_step=100,
    ckpt_max_to_keep=3,
    dtype=jnp.float32,
)

# create optimizer
optimizer = optax.adam(hyper_params.lr)

trainer = Trainer(
    model=FlaxGPT2ForCausalLMModule(config=model_config),
    optimizer=optimizer,
    loss_fn=cross_entropy_loss_and_accuracy,
    hyperparams=hyper_params,
)

# create dataset
dataset = load_dataset(
    "openwebtext", split="train", streaming=True
).shuffle(buffer_size=10_000, seed=42)

dataset = JSONLDatasetForAutoRegressiveModel(
    dataset=dataset,
    seq_len=1024,
    doc_separator="",
    batch_size=2,
    tokenizer=tokenizer,
)

# fit the model
trainer.fit(dataset)