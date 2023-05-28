import haiku as hk
from typing import Optional

class FMTrainerModel(hk.Module):
    def __init__(self, name: str | None = None):
        super().__init__(name)