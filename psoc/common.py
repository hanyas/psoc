from typing import Callable, Dict, Union
from functools import partial

import jax
from jax import numpy as jnp
from jax import lax as jl
from jax import random as jr

from flax.training.train_state import TrainState

from psoc.abstract import OpenLoop, FeedbackLoop
from psoc.sampling import smc_sampling
from psoc.algorithms import rao_blackwell_csmc
from psoc.utils import create_train_state


def initialize(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    init_state: jnp.ndarray,
    tempering: float,
    input_dim: int,
    learning_rate: float,
    environment,
):
    key, sub_key = jr.split(key, 2)
    opt_state = create_train_state(
        key=sub_key,
        module=environment.module,
        init_data=jnp.zeros((input_dim,)),
        learning_rate=learning_rate
    )

    key, sub_key = jr.split(key, 2)
    reference = smc_sampling(
        sub_key,
        nb_steps,
        int(10 * nb_particles),
        1,
        init_state,
        opt_state.params,
        tempering,
        environment
    )[0]
    return opt_state, reference


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
    environment,
):
    vmap_ll = jax.vmap(log_complete_likelihood, in_axes=(0, 0, None, None))

    def loss_fn(params):
        _, loop, reward_fn = \
            environment.create_env(init_state, params, tempering)
        lls = vmap_ll(states, next_states, loop, reward_fn)
        return - 1.0 * jnp.mean(lls)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(opt_state.params)
    return opt_state.apply_gradients(grads=grads), loss


def compute_score(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    environment,
):
    def loss_fn(
        state: jnp.ndarray,
        next_state: jnp.ndarray,
        parameters: Dict,
    ):
        _, loop, reward_fn = \
            environment.create_env(init_state, parameters, tempering)
        loss = log_complete_likelihood(state, next_state, loop, reward_fn)
        return - 1.0 * loss

    loss_fn_ = lambda x, xn, p: loss_fn(x, xn, p)
    score_fn_ = lambda x, xn, p: jax.grad(loss_fn, 2)(x, xn, p)

    prior, loop, reward_fn = \
        environment.create_env(init_state, parameters, tempering)

    key, sub_key = jr.split(key, 2)
    reference, loss, score = rao_blackwell_csmc(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        prior,
        loop,
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
