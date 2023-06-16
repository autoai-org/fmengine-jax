import jax.numpy as jnp
from datasets import Dataset
from transformers import AutoTokenizer
from fmengine.dataloader._base import FMTrainerDataset

class JSONLDatasetForAutoRegressiveModel(FMTrainerDataset):
    def __init__(
        self,
        dataset: Dataset,
        seq_len: int,
        doc_separator: str = "",
        batch_size: int = 1,
        tokenizer: AutoTokenizer = None,
        field:str = "text"
    ) -> None:
        self.field = field
        self.dataset = dataset.with_format("jax")
        self.seq_len = seq_len
        self.doc_separator = doc_separator
        self.tokenizer = tokenizer
        self.it = None
        self.iter_count = 0
        self.token_buffer = []
        self.batch_size = batch_size
        self.chunk_size = seq_len * batch_size

    def state_dict(self):
        return {
            "iter_count": self.iter_count,
            "token_buffer": self.token_buffer,
        }

    def load_state_dict(self, state_dict):
        self.iter_count = state_dict["iter_count"]
        self.token_buffer = state_dict["token_buffer"]
        self.dataset = self.dataset.skip(self.iter_count)

    def get_sequence(self):
        while True:
            try:
                for x in iter(self.dataset.skip(self.iter_count)):
                    self.iter_count += 1
                    curr_tokens = self.tokenizer(self.doc_separator + x[self.field])[
                        "input_ids"
                    ]
                    self.token_buffer += curr_tokens
                    while len(self.token_buffer) >= self.chunk_size+1:
                        batch = {
                            "input_tokens": jnp.array(
                                self.token_buffer[: self.chunk_size]
                            ).reshape((self.batch_size, -1)),
                            "target_tokens": jnp.array(
                                self.token_buffer[1: self.chunk_size + 1]
                            ).reshape(self.batch_size, -1),
                        }
                        self.token_buffer = self.token_buffer[self.chunk_size:]
                        yield batch
            except Exception as e:
                raise e

