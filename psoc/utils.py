import math
from typing import Dict, Callable

import jax
from jax import numpy as jnp
from jax import random as jr

import numpy as onp

from flax import linen as nn
from flax.training.train_state import TrainState

import optax


def create_pairs(
    samples: jnp.ndarray
):
    nb_samples, nb_steps, dim = samples.shape
    states, next_states = samples[:, :-1, :], samples[:, 1:, :]
    states = jnp.reshape(states, (nb_samples * (nb_steps - 1), dim))
    next_states = jnp.reshape(next_states, (nb_samples * (nb_steps - 1), dim))
    return states, next_states


def batcher(
    key: jax.Array,
    samples: jnp.ndarray,
    batch_size: int,
    skip_last: bool = False,
):
    states, next_states = create_pairs(samples)
    batch_idx = jr.permutation(key, len(states))
    batch_idx = onp.asarray(batch_idx)

    if skip_last:
        # Skip incomplete batch
        steps_per_epoch = len(states) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        # include incomplete batch
        steps_per_epoch = math.ceil(len(states) / batch_size)
        batch_idx = onp.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        yield states[idx], next_states[idx]


def create_train_state(
    key: jax.Array,
    module: nn.Module,
    init_data: jnp.ndarray,
    learning_rate: float,
    optimizer: Callable = optax.adam
):
    params = module.init(key, init_data)["params"]
    tx = optimizer(learning_rate)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx
    )


def positivity_constraint(params: Dict):
    return jax.tree_map(lambda _x: jnp.log1p(jnp.exp(_x)), params)


def identity_constraint(params: Dict):
    return jax.tree_map(lambda _x: _x, params)
