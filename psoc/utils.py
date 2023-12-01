import math
from typing import Dict

import jax
from jax import numpy as jnp, random as jr

import optax
from flax import linen as nn
from flax.training.train_state import TrainState


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
):
    states, next_states = create_pairs(samples)

    batch_idx = jr.permutation(key, len(states))
    batch_idx = onp.asarray(batch_idx)

    # include incomplete batch
    steps_per_epoch = math.ceil(len(states) / batch_size)
    batch_idx = onp.array_split(batch_idx, steps_per_epoch)

    # # Skip incomplete batch
    # steps_per_epoch = len(states) // batch_size
    # batch_idx = batch_idx[: steps_per_epoch * batch_size]
    # batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        yield states[idx], next_states[idx]


def create_train_state(
    key: jax.Array,
    module: nn.Module,
    init_data: jnp.ndarray,
    learning_rate: float
):
    params = module.init(key, init_data)["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx
    )


def constrain(params: Dict):
    return jax.tree_map(lambda _x: jnp.log1p(jnp.exp(_x)), params)
