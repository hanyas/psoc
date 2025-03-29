from functools import partial
from typing import Callable

import jax
from flax.training.train_state import TrainState

from jax import numpy as jnp
from jax import random as jr
from jax import lax as jl

from jax.experimental.host_callback import id_tap

from psoc.sampling import smc_sampling, csmc_sampling
from psoc.sampling import rb_csmc_sampling
from psoc.common import maximization
from psoc.common import compute_score
from psoc.common import compute_markovian_score
from psoc.common import rollout

from psoc.utils import batcher


@partial(jax.jit, static_argnums=(1, 2, 3, 4, -2, -1))
def score_optimization(
    key: jax.Array,
    nb_iter: int,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    opt_state: TrainState,
    tempering: float,
    make_env: Callable,
    verbose: bool = False,
):
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, reward: {z[1]:.4f}", end="\n"
    )

    def iteration(carry, args):
        i, key, opt_state = carry

        key, sub_key = jr.split(key, 2)
        _, loss, score = compute_score(
            sub_key,
            nb_steps,
            nb_particles,
            nb_samples,
            init_state,
            opt_state.params,
            tempering,
            make_env
        )
        opt_state = opt_state.apply_gradients(grads=score)

        key, sub_key = jr.split(key, 2)
        _, reward = rollout(
            sub_key,
            nb_steps,
            nb_samples,
            init_state,
            opt_state.params,
            1.0,
            make_env,
        )

        if verbose:
            id_tap(print_func, (i, loss))
        return (i+1, key, opt_state), (loss, reward)

    key, sub_key = jr.split(key, 2)
    (_, _, opt_state), (loss, reward) = \
        jl.scan(iteration, (0, sub_key, opt_state), (), length=nb_iter)

    return opt_state, reward


def batched_markovian_score_optimization(
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
    make_env: Callable,
    verbose: bool = False,
):
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, reward: {z[1]:.4f}", end="\n"
    )

    reward_list = []
    for i in range(nb_iter):
        key, sample_key, batch_key = jr.split(key, 3)

        # sampling step
        samples, reference = csmc_sampling(
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

        key, sub_key = jr.split(key, 2)
        _, reward = rollout(
            sub_key,
            nb_steps,
            int(10 * nb_samples),
            init_state,
            opt_state.params,
            1.0,
            make_env,
        )
        reward_list.append(reward)

        if verbose:
            id_tap(print_func, (i, reward))

    return opt_state, jnp.asarray(reward_list)


def batched_rao_blackwell_markovian_score_optimization(
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
    make_env: Callable,
    verbose: bool = False,
):
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, reward: {z[1]:.4f}", end="\n"
    )

    reward_list = []
    for i in range(nb_iter):
        key, sample_key, batch_key = jr.split(key, 3)

        # sampling step
        samples, reference = rb_csmc_sampling(
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

        key, sub_key = jr.split(key, 2)
        _, reward = rollout(
            sub_key,
            nb_steps,
            nb_samples,
            init_state,
            opt_state.params,
            1.0,
            make_env,
        )
        reward_list.append(reward)

        if verbose:
            id_tap(print_func, (i, reward))

    return opt_state, jnp.asarray(reward_list)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, -2, -1))
def rao_blackwell_markovian_score_optimization(
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
    verbose: bool = False,
):
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, reward: {z[1]:.4f}", end="\n"
    )

    def iteration(carry, args):
        i, key, reference, opt_state = carry

        key, sub_key = jr.split(key, 2)
        reference, loss, score = compute_markovian_score(
            sub_key,
            nb_steps,
            nb_particles,
            nb_samples,
            reference,
            init_state,
            opt_state.params,
            tempering,
            make_env
        )
        opt_state = opt_state.apply_gradients(grads=score)

        key, sub_key = jr.split(key, 2)
        _, reward = rollout(
            sub_key,
            nb_steps,
            nb_samples,
            init_state,
            opt_state.params,
            1.0,
            make_env,
        )

        if verbose:
            id_tap(print_func, (i, reward))
        return (i+1, key, reference, opt_state), (loss, reward)

    key, sub_key = jr.split(key, 2)
    (_, _, reference, opt_state), (loss, reward) = \
        jl.scan(iteration, (0, sub_key, reference, opt_state), (), length=nb_iter)

    return opt_state, reward
