from typing import Callable

import jax
from jax import random as jr
from jax import numpy as jnp

import distrax

from psoc.abstract import StochasticDynamics
from psoc.abstract import Network

from psoc.common import rollout
from psoc.utils import create_train_state
from psoc.sampling import smc_sampling

from psoc.optimization import score_optimization
from psoc.optimization import batched_markovian_score_optimization
from psoc.optimization import batched_rao_blackwell_markovian_score_optimization


def smc_experiment(
    key: jax.Array,
    state_dim: int,
    action_dim: int,
    dynamics: StochasticDynamics,
    network: Network,
    bijector: distrax.Chain,
    make_env: Callable,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    tempering: float,
    nb_iter: int,
    learning_rate: float,
):
    key, sub_key = jr.split(key, 2)
    opt_state = create_train_state(
        key=sub_key,
        module=network,
        init_data=jnp.zeros((state_dim,)),
        learning_rate=learning_rate
    )

    key, sub_key = jr.split(key, 2)
    _, init_reward = rollout(
        sub_key,
        nb_steps,
        int(10 * nb_samples),
        init_state,
        opt_state.params,
        1.0,
        make_env,
    )

    key, sub_key = jr.split(key, 2)
    opt_state, reward = \
        score_optimization(
            sub_key,
            nb_iter,
            nb_steps,
            nb_particles,
            nb_samples,
            init_state,
            opt_state,
            tempering,
            make_env
        )

    reward = jnp.hstack((init_reward, reward))
    return opt_state, reward


def csmc_experiment(
    key: jax.Array,
    state_dim: int,
    action_dim: int,
    dynamics: StochasticDynamics,
    network: Network,
    bijector: distrax.Chain,
    make_env: Callable,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    tempering: float,
    nb_iter: int,
    learning_rate: float,
    batch_size=64,
):
    key, sub_key = jr.split(key, 2)
    opt_state = create_train_state(
        key=sub_key,
        module=network,
        init_data=jnp.zeros((state_dim,)),
        learning_rate=learning_rate
    )

    key, sub_key = jr.split(key, 2)
    _, init_reward = rollout(
        sub_key,
        nb_steps,
        int(10 * nb_samples),
        init_state,
        opt_state.params,
        1.0,
        make_env,
    )

    key, sub_key = jr.split(key, 2)
    reference = smc_sampling(
        sub_key,
        nb_steps,
        int(10 * nb_particles),
        int(10 * nb_particles),
        init_state,
        opt_state.params,
        tempering,
        make_env
    )[0]

    key, sub_key = jr.split(key, 2)
    opt_state, reward = \
        batched_markovian_score_optimization(
            sub_key,
            nb_iter,
            nb_steps,
            nb_particles,
            nb_samples,
            reference,
            init_state,
            opt_state,
            tempering,
            batch_size,
            make_env
        )

    reward = jnp.hstack((init_reward, reward))
    return opt_state, reward


def rb_csmc_experiment(
    key: jax.Array,
    state_dim: int,
    action_dim: int,
    dynamics: StochasticDynamics,
    network: Network,
    bijector: distrax.Chain,
    make_env: Callable,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    tempering: float,
    nb_iter: int,
    learning_rate: float,
    batch_size: int = 64,
):
    key, sub_key = jr.split(key, 2)
    opt_state = create_train_state(
        key=sub_key,
        module=network,
        init_data=jnp.zeros((state_dim,)),
        learning_rate=learning_rate
    )

    key, sub_key = jr.split(key, 2)
    _, init_reward = rollout(
        sub_key,
        nb_steps,
        int(10 * nb_samples),
        init_state,
        opt_state.params,
        1.0,
        make_env,
    )

    key, sub_key = jr.split(key, 2)
    reference = smc_sampling(
        sub_key,
        nb_steps,
        int(10 * nb_particles),
        int(10 * nb_particles),
        init_state,
        opt_state.params,
        tempering,
        make_env
    )[0]

    key, sub_key = jr.split(key, 2)
    opt_state, reward = \
        batched_rao_blackwell_markovian_score_optimization(
            sub_key,
            nb_iter,
            nb_steps,
            nb_particles,
            nb_samples,
            reference,
            init_state,
            opt_state,
            tempering,
            batch_size,
            make_env
        )

    reward = jnp.hstack((init_reward, reward))
    return opt_state, reward
