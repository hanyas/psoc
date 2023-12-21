from typing import Callable, Dict, Union
from functools import partial

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import lax as jl

from flax.training.train_state import TrainState

from psoc.abstract import OpenLoop, FeedbackLoop
from psoc.algorithms import smc_with_score
from psoc.algorithms import rao_blackwell_csmc_with_score


def log_complete_likelihood(
    state: jnp.ndarray,
    next_state: jnp.ndarray,
    transition_model: Union[OpenLoop, FeedbackLoop],
    log_observation: Callable,
):
    ll = transition_model.logpdf(state, next_state) \
         + log_observation(next_state)
    return ll


@partial(jax.jit, static_argnums=(-1,))
def maximization(
    states: jnp.ndarray,
    next_states: jnp.ndarray,
    init_state: jnp.ndarray,
    opt_state: TrainState,
    tempering: float,
    make_env: Callable
):
    vmap_ll = jax.vmap(log_complete_likelihood, in_axes=(0, 0, None, None))

    def loss_fn(params):
        _, loop_obj, reward_fn = \
            make_env(init_state, params, tempering)
        lls = vmap_ll(states, next_states, loop_obj, reward_fn)
        return - 1.0 * jnp.mean(lls)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(opt_state.params)
    return opt_state.apply_gradients(grads=grads), loss


def compute_score(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    make_env: Callable,
):
    def loss_fn(
        state: jnp.ndarray,
        next_state: jnp.ndarray,
        _parameters: Dict,
    ):
        _, loop, reward_fn = make_env(init_state, _parameters, tempering)
        loss = log_complete_likelihood(state, next_state, loop, reward_fn)
        return - 1.0 * loss

    loss_fn_ = lambda x, xn, p: loss_fn(x, xn, p)
    score_fn_ = lambda x, xn, p: jax.grad(loss_fn, 2)(x, xn, p)

    prior_dist, loop_obj, reward_fn = \
        make_env(init_state, parameters, tempering)

    key, sub_key = jr.split(key, 2)
    samples, loss, score = smc_with_score(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        prior_dist,
        loop_obj,
        reward_fn,
        loss_fn_,
        score_fn_,
        parameters
    )
    return samples, loss, score


def compute_markovian_score(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    make_env: Callable,
):
    def loss_fn(
        state: jnp.ndarray,
        next_state: jnp.ndarray,
        _parameters: Dict,
    ):
        _, loop, reward_fn = make_env(init_state, _parameters, tempering)
        loss = log_complete_likelihood(state, next_state, loop, reward_fn)
        return - 1.0 * loss

    loss_fn_ = lambda x, xn, p: loss_fn(x, xn, p)
    score_fn_ = lambda x, xn, p: jax.grad(loss_fn, 2)(x, xn, p)

    prior_dist, loop_obj, reward_fn = \
        make_env(init_state, parameters, tempering)

    key, sub_key = jr.split(key, 2)
    reference, loss, score = rao_blackwell_csmc_with_score(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        prior_dist,
        loop_obj,
        reward_fn,
        loss_fn_,
        score_fn_,
        parameters
    )
    return reference, loss, score


def compute_cost(
    samples: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    make_env: Callable
):
    _, _, reward_fn = \
        make_env(init_state, parameters, tempering)
    return - jnp.mean(jnp.sum(reward_fn(samples), axis=0))


@partial(jax.jit, static_argnums=(1, 2, -1))
def rollout(
    key: jax.Array,
    nb_steps: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    make_env: Callable,
):
    prior_dist, loop_obj, reward_fn = \
        make_env(init_state, parameters, tempering)

    def body(carry, args):
        key, prev_states = carry
        key, sub_key = jr.split(key, 2)
        next_states = loop_obj.forward(sub_key, prev_states)
        return (key, next_states), next_states

    key, sub_key = jr.split(key, 2)
    init_states = prior_dist.sample(seed=sub_key, sample_shape=(nb_samples,))

    key, sub_key = jr.split(key, 2)
    _, samples = \
        jl.scan(body, (sub_key, init_states), (), length=nb_steps - 1)

    samples = jnp.insert(samples, 0, init_states, 0)
    samples = jnp.swapaxes(samples, 0, 1)
    rewards = jnp.mean(jnp.sum(reward_fn(samples), axis=1))
    return samples, rewards
