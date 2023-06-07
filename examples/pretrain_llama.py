import jax
import optax
import portpicker
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

from fmtrainer.trainer.trainer import ShardedLMTrainer, HyperParams
from fmtrainer.nn.losses import cross_entropy_loss_and_accuracy
from fmtrainer.nn.optimizers import adamw
from fmtrainer.modelling.language.llama.llama_config import LLaMAConfig, LLAMA_STANDARD_CONFIGS
from fmtrainer.dataloader.jsonl_reader import JSONLDatasetForAutoRegressiveModel
from fmtrainer.modelling.language.llama.llama_model import FlaxLLaMAForCausalLMModule

model_config = LLaMAConfig(**LLAMA_STANDARD_CONFIGS['debug'])
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

# Create a HyperParams object
hyper_params = HyperParams(
    name="llama-falcon-refinedweb",
    lr=1e-4,
    steps=50000,
    batch_size=2,
    lr_warmup_steps=5000,
    weight_decay=0.0001,
    accumulate_gradient_steps=4,
    seq_len=1024,
    seed=42,
    ckpt_dir=".cache/llama_checkpoints/",
    ckpt_step=5000,
    ckpt_max_to_keep=3,
    dtype=jnp.int32,
)

# create optimizer
optimizer, optimizer_info = adamw(hyper_params)

port = portpicker.pick_unused_port()
jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

trainer = ShardedLMTrainer(
    model=FlaxLLaMAForCausalLMModule(config=model_config),
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
    seq_len=2048,
    doc_separator="",
    batch_size=2,
    tokenizer=tokenizer,
    field='content',
)

# fit the model
trainer.fit(dataset)
