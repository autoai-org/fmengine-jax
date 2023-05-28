import haiku as hk

class ConfigModule(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config