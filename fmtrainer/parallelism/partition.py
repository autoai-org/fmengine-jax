import re
import numpy as np
from jax.sharding import PartitionSpec as PS
from fmtrainer.utils.trees import named_tree_map

def match_partition_rules(rules, params):
    """
    Returns a pytree of PartitionSpec according to rules.
    Supports handling Flax TrainState and Optax optimizer state.
    """
    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')
    return named_tree_map(get_partition_spec, params, sep='/')
