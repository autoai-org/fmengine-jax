import re
import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from fmengine.utils.trees import named_tree_map
from jax.interpreters import pxla
from jax.experimental.pjit import with_sharding_constraint as _with_sharding_constraint

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

def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """ Create pytree of sharding and gathering functions from pytree of partition specs.
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                # Convert all float tensors to the same dtype
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return tensor.astype(dtype_spec.dtype)
            return tensor
        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=None,
            out_shardings=partition_spec
        )
        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()
        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=partition_spec,
            out_shardings=None
        )
        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))
        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(
            make_shard_fn, partition_specs, dtype_specs
        )
        gather_fns = jax.tree_util.tree_map(
            make_gather_fn, partition_specs, dtype_specs
        )
    return shard_fns, gather_fns

def get_names_from_parition_spec(partition_specs):
    """ Return axis names from partition specs. """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)

def names_in_current_mesh(*names):
    """ Check if current mesh axes contain these names. """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def with_sharding_constraint(x, partition_specs):
    """ A smarter version of with_sharding_constraint that only applies the
        constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        x = _with_sharding_constraint(x, partition_specs)
    return x