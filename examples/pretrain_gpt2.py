import optax
import jax.numpy as jnp

from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

from fmtrainer.trainer.trainer import Trainer, HyperParams
from fmtrainer.nn.losses import cross_entropy_loss_and_accuracy
from fmtrainer.modelling.language.gpt2.gpt2_config import GPT2Config
from fmtrainer.dataloader.jsonl_reader import JSONLDatasetForAutoRegressiveModel
from fmtrainer.modelling.language.gpt2.gpt2_model import FlaxGPT2ForCausalLMModule

model_config = GPT2Config()
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

# Create a hyperparameters
hyper_params = HyperParams(
    lr=1e-4,
    steps=10,
    batch_size=8,
    warmup_steps=0,
    seq_len=1024,
    seed=42,
    dtype=jnp.float32,
)

# create optimizer
optimizer = optax.sgd(hyper_params.lr)

trainer = Trainer(
    model=FlaxGPT2ForCausalLMModule(config=model_config),
    optimizer=optimizer,
    loss_fn=cross_entropy_loss_and_accuracy,
    hyperparams=hyper_params,
)

# create dataset
data_files = {"train": ".cache/ft_data/train.jsonl"}
dataset = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
).shuffle(buffer_size=10_000, seed=42)
dataset = JSONLDatasetForAutoRegressiveModel(
    dataset=dataset,
    seq_len=1024,
    doc_separator="",
    batch_size=2,
    tokenizer=tokenizer,
)

trainer.fit(dataset)