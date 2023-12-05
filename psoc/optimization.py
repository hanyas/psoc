from functools import partial
from typing import Callable, Any

import jax
from flax.training.train_state import TrainState

from jax import numpy as jnp
from jax import random as jr
from jax import lax as jl

from jax.experimental.host_callback import id_tap

from psoc.sampling import csmc_sampling
from psoc.common import maximization, compute_score
from psoc.utils import batcher


def score_optimization(
    key: jax.Array,
    nb_iter: int,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
    init_state: jnp.ndarray,
    opt_state: TrainState,
    tempering: float,
    batch_size: int,
    make_env: Callable
):
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, loss: {z[1]:.4f}", end="\n"
    )

    for i in range(nb_iter):
        key, sample_key, batch_key = jr.split(key, 3)

        # sampling step
        samples = csmc_sampling(
            sample_key,
            nb_steps,
            nb_particles,
            nb_samples,
            reference,
            init_state,
            opt_state.params,
            tempering,
            make_env
        )
        reference = samples[-1]

        # maximization step
        loss = 0.0
        batches = batcher(batch_key, samples, batch_size)
        for batch in batches:
            states, next_states = batch
            opt_state, batch_loss = maximization(
                states,
                next_states,
                init_state,
                opt_state,
                tempering,
                make_env
            )
            loss += batch_loss

        id_tap(print_func, (i, loss))

    return opt_state


@partial(jax.jit, static_argnums=(1, 2, 3, 4, -1))
def rao_blackwell_score_optimization(
    key: jax.Array,
    nb_iter: int,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
    init_state: jnp.ndarray,
    opt_state: TrainState,
    tempering: float,
    make_env: Callable,
):
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, loss: {z[1]:.4f}", end="\n"
    )

    def iteration(carry, args):
        i, key, reference, opt_state = carry

        key, sub_key = jr.split(key, 2)
        reference, loss, score = compute_score(
            sub_key,
            nb_steps,
            nb_particles,
            nb_samples,
            reference,
            init_state,
            opt_state.params,
            tempering,
            make_env,
        )
        opt_state = opt_state.apply_gradients(grads=score)

        id_tap(print_func, (i, loss))
        return (i+1, key, reference, opt_state), loss

    key, sub_key = jr.split(key, 2)
    (_, _, reference, opt_state), loss = \
        jl.scan(iteration, (0, key, reference, opt_state), (), length=nb_iter)

    return opt_state, reference, loss
