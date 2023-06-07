import jax
import optax
import portpicker
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

from fmtrainer.trainer.trainer import ShardedLMTrainer, HyperParams
from fmtrainer.nn.losses import cross_entropy_loss_and_accuracy
from fmtrainer.nn.optimizers import adamw
from fmtrainer.modelling.language.gptj.gptj_config import GPTJConfig, GPTJ_STANDARD_CONFIGS
from fmtrainer.dataloader.jsonl_reader import JSONLDatasetForAutoRegressiveModel
from fmtrainer.modelling.language.gptj.gptj_model import FlaxGPTJForCausalLMModule

model_config = GPTJConfig.from_dict(GPTJ_STANDARD_CONFIGS["6b"])

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-j-6B",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|extratoken_40|>",
    cls_token="<|extratoken_41|>",
    mask_token="<|extratoken_42|>",
    padding_side="left",
    truncation_side="right",
)

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
    ckpt_dir=".cache/gptj/",
    ckpt_step=5000,
    ckpt_max_to_keep=3,
    dtype=jnp.int32,
)

# create optimizer
optimizer, optimizer_info = adamw(hyper_params)

port = portpicker.pick_unused_port()
jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

trainer = ShardedLMTrainer(
    model=FlaxGPTJForCausalLMModule(config=model_config, dtype=jnp.float32),
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
