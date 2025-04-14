from typing import Callable

import jax
from jax import random as jr
from jax import numpy as jnp
from jax import lax as jl

import distrax
import optax

from psoc.abstract import StochasticDynamics
from psoc.abstract import GaussMarkov, Gaussian

from psoc.common import rollout, compute_cost
from psoc.utils import create_train_state

from psoc.optimization import score_optimization
from psoc.optimization import rao_blackwell_markovian_score_optimization

from psoc.algorithms import mppi


def smc_experiment(
    key: jax.Array,
    state_dim: int,
    action_dim: int,
    dynamics: StochasticDynamics,
    proposal: GaussMarkov,
    bijector: distrax.Chain,
    make_env: Callable,
    nb_steps: int,
    horizon: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    tempering: float,
    nb_iter: int,
    learning_rate: float,
):
    def body(carry, args):
        key, state, optim = carry

        key, sub_key = jr.split(key, 2)
        optim, sample, _ = \
            score_optimization(
                sub_key,
                nb_iter,
                horizon,
                nb_particles,
                nb_samples,
                state,
                optim,
                tempering,
                make_env
            )
        x = sample[0, :state_dim]
        u = sample[1, -action_dim:]

        key, sub_key = jr.split(key, 2)
        xn = dynamics.sample(sub_key, x, u)

        next_state = jnp.hstack((xn, u))
        return (key, next_state, optim), state

    key, sub_key = jr.split(key, 2)
    opt_state = create_train_state(
        key=sub_key,
        module=proposal,
        init_data=jnp.zeros((action_dim,)),
        learning_rate=learning_rate,
        optimizer=optax.sgd
    )

    (_, last_state, opt_state), trajectory = \
        jl.scan(body, (key, init_state, opt_state), (), length=nb_steps - 1)

    trajectory = jnp.vstack((trajectory, last_state))
    cost = compute_cost(
        trajectory,
        init_state,
        opt_state.params,
        tempering,
        make_env
    )
    return trajectory, cost


def csmc_experiment(
    key: jax.Array,
    state_dim: int,
    action_dim: int,
    dynamics: StochasticDynamics,
    proposal: GaussMarkov,
    bijector: distrax.Chain,
    make_env: Callable,
    nb_steps: int,
    horizon: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    tempering: float,
    nb_iter: int,
    learning_rate: float,
):
    def body(carry, args):
        key, state, optim = carry

        key, sub_key = jr.split(key, 2)
        reference = rollout(
            sub_key,
            horizon,
            state,
            optim.params,
            tempering,
            make_env,
        )

        key, sub_key = jr.split(key, 2)
        optim, sample, _ = \
            rao_blackwell_markovian_score_optimization(
                sub_key,
                nb_iter,
                horizon,
                nb_particles,
                nb_samples,
                reference,
                state,
                optim,
                tempering,
                make_env
            )
        x = sample[0, :state_dim]
        u = sample[1, -action_dim:]

        key, sub_key = jr.split(key, 2)
        xn = dynamics.sample(sub_key, x, u)

        next_state = jnp.hstack((xn, u))
        return (key, next_state, optim), state

    key, sub_key = jr.split(key, 2)
    opt_state = create_train_state(
        key=sub_key,
        module=proposal,
        init_data=jnp.zeros((action_dim,)),
        learning_rate=learning_rate,
        optimizer=optax.sgd
    )

    (_, last_state, opt_state), trajectory = \
        jl.scan(body, (key, init_state, opt_state), (), length=nb_steps - 1)

    trajectory = jnp.vstack((trajectory, last_state))
    cost = compute_cost(
        trajectory,
        init_state,
        opt_state.params,
        tempering,
        make_env
    )
    return trajectory, cost


def mppi_experiment(
    key: jax.Array,
    state_dim: int,
    action_dim: int,
    dynamics: StochasticDynamics,
    proposal: Gaussian,
    bijector: distrax.Chain,
    make_env: Callable,
    nb_steps: int,
    horizon: int,
    nb_particles: int,
    init_state: jnp.ndarray,
    tempering: float,
):
    key, sub_key = jr.split(key, 2)
    params = proposal.init(
        sub_key, jnp.zeros((action_dim,))
    )["params"]

    def body(carry, args):
        key, state = carry

        prior_dist, loop_obj, reward_fn = \
            make_env(state, params, tempering)

        key, sub_key = jr.split(key, 2)
        samples, weights = \
            mppi(
                sub_key,
                horizon,
                nb_particles,
                prior_dist,
                loop_obj,
                reward_fn
            )
        key, sub_key = jr.split(key, 2)
        idx = jr.choice(sub_key, a=len(weights), p=weights)
        sample = samples[:, idx, :]

        x = sample[0, :state_dim]
        u = sample[1, -action_dim:]

        key, sub_key = jr.split(key, 2)
        xn = dynamics.sample(sub_key, x, u)

        next_state = jnp.hstack((xn, u))
        return (key, next_state), state

    (_, last_state), trajectory = \
        jl.scan(body, (key, init_state), (), length=nb_steps - 1)

    trajectory = jnp.vstack((trajectory, last_state))
    cost = compute_cost(
        trajectory,
        init_state,
        params,
        tempering,
        make_env
    )
    return trajectory, cost
