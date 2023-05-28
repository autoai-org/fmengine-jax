from fmtrainer.trainer.trainer import Trainer, HyperParams
from fmtrainer.modelling.language.gpt2.gpt2_model import FlaxGPT2ForCausalLMModule
from fmtrainer.modelling.language.gpt2.gpt2_config import GPT2Config

model_config = GPT2Config()

# Create a trainer object
hyper_params = HyperParams(
    lr=1e-4,
    steps=10,
    batch_size=8,
    warmup_steps=0,
)

trainer = Trainer(
    model=FlaxGPT2ForCausalLMModule(config=model_config),
    optimizer=None,
    loss_fn=None,
    hyperparams=hyper_params,
)