from typing import Callable, Dict, Union
from functools import partial

import jax
from jax import numpy as jnp
from jax import lax as jl
from jax import random as jr

from flax.training.train_state import TrainState
from psoc.abstract import OpenLoop, FeedbackLoop


def log_complete_likelihood(
    state: jnp.ndarray,
    next_state: jnp.ndarray,
    transition_model: Union[OpenLoop, FeedbackLoop],
    log_observation: Callable,
):
    ll = transition_model.logpdf(state, next_state) \
         + log_observation(next_state)
    return ll


def loss_fn(
    states: jnp.ndarray,
    next_states: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    environment,
):
    _, loop, reward_fn = \
        environment.create_env(init_state, parameters, tempering)
    vmap_ll = jax.vmap(log_complete_likelihood, in_axes=(0, 0, None, None))
    lls = vmap_ll(states, next_states, loop, reward_fn)
    return - jnp.mean(lls)


@partial(jax.jit, static_argnums=(-1,))
def maximization(
    states: jnp.ndarray,
    next_states: jnp.ndarray,
    init_state: jnp.ndarray,
    opt_state: TrainState,
    tempering: float,
    environment,
):
    def _loss_fn(params):
        loss = loss_fn(
            states,
            next_states,
            init_state,
            params,
            tempering,
            environment
        )
        return loss

    grad_fn = jax.value_and_grad(_loss_fn)
    loss, grads = grad_fn(opt_state.params)
    return opt_state.apply_gradients(grads=grads), loss


def compute_cost(
    samples: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    environment,
):
    _, _, reward_fn = \
        environment.create_env(init_state, parameters, tempering)
    return - jnp.mean(jnp.sum(reward_fn(samples), axis=0))


def rollout(
    key: jax.Array,
    nb_steps: int,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    environment,
):
    prior, loop, _ = \
        environment.create_env(init_state, parameters, tempering)

    def body(carry, args):
        key, prev_state = carry
        key, sub_key = jr.split(key, 2)
        next_state = loop.forward(sub_key, prev_state)
        return (key, next_state), next_state

    _, states = \
        jl.scan(body, (key, init_state), (), length=nb_steps - 1)

    states = jnp.vstack((init_state, states))
    return states
