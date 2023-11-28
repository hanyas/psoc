from typing import Callable, Dict
from functools import partial

import jax
from jax import numpy as jnp
from jax import lax as jl
from jax import random as jr

import numpy as onp

from flax import linen as nn
from flax.training.train_state import TrainState

import distrax
import optax

from psoc.abstract import FeedbackLoop
from psoc.algorithms import smc, csmc


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

    # # include incomplete batch
    # steps_per_epoch = math.ceil(len(states) / batch_size)
    # batch_idx = onp.array_split(batch_idx, steps_per_epoch)

    # Skip incomplete batch
    steps_per_epoch = len(states) // batch_size
    batch_idx = batch_idx[: steps_per_epoch * batch_size]
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        yield states[idx], next_states[idx]


def lower_bound(
    states: jnp.ndarray,
    next_states: jnp.ndarray,
    params: Dict,
    eta: float,
    env,
):
    _, closedloop, reward = env.create_env(params, eta)
    lls = log_complete_likelihood(states, next_states, closedloop, reward)
    return - jnp.sum(lls)


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


@partial(jax.jit, static_argnums=(-1,))
def maximization(
    opt_state: TrainState,
    states: jnp.ndarray,
    next_states: jnp.ndarray,
    eta: float,
    env,
):
    def loss_fn(params):
        loss = lower_bound(states, next_states, params, eta, env)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(opt_state.params)
    return opt_state.apply_gradients(grads=grads), loss


def compute_cost(
    samples: jnp.ndarray,
    params: Dict,
    eta: float,
    env,
):
    _, _, reward = env.create_env(params, eta)
    _, next_states = create_pairs(samples)
    return - jnp.mean(jax.vmap(reward)(next_states))


@partial(jax.jit, static_argnums=(1, 2, 3, -1))
def csmc_sampling(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
    params: Dict,
    eta: float,
    env,
):
    prior, closedloop, reward = \
        env.create_env(params, eta)

    def body(carry, args):
        key, reference = carry
        key, sub_key = jr.split(key, 2)
        sample = csmc(
            sub_key,
            nb_steps,
            nb_particles,
            reference,
            prior,
            closedloop,
            reward,
        )
        return (key, sample), sample

    _, samples = \
        jax.lax.scan(body, (key, reference), (), length=nb_samples)
    return samples


@partial(jax.jit, static_argnums=(1, 2, 3, -1))
def smc_sampling(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    params: Dict,
    eta: float,
    env,
):
    prior, closedloop, reward = \
        env.create_env(params, eta)

    key, sub_key = jr.split(key, 2)
    samples, weights = smc(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        prior,
        closedloop,
        reward,
    )

    key, sub_key = jr.split(key, 2)
    idx = jr.choice(sub_key, a=nb_samples, p=weights, shape=(nb_samples,))
    return samples[idx, ...]


def rollout(
    key: jax.Array,
    prior: distrax.Distribution,
    transition_model: FeedbackLoop,
    nb_steps: int,
    nb_samples: int,
):
    def body(carry, args):
        key, prev_state = carry
        key, sub_key = jr.split(key, 2)
        next_state = transition_model.sample(sub_key, prev_state)
        return (key, next_state), next_state

    key, sub_key = jr.split(key, 2)
    init_state = prior.sample(seed=sub_key, sample_shape=(nb_samples,))
    _, states = \
        jl.scan(body, (key, init_state), (), length=nb_steps - 1)

    states = jnp.insert(states, 0, init_state, 0)
    return states.swapaxes(0, 1)


def simulate(
    prior: distrax.Distribution,
    transition_model: FeedbackLoop,
    nb_steps: int,
):
    def body(carry, args):
        prev_state = carry
        next_state = transition_model.mean(prev_state)
        return next_state, next_state

    init_state = prior.mean()
    _, states = \
        jl.scan(body, init_state, (), length=nb_steps - 1)

    states = jnp.insert(states, 0, init_state, 0)
    return states


@partial(jax.vmap, in_axes=(0, 0, None, None))
def log_complete_likelihood(
    state: jnp.ndarray,
    next_state: jnp.ndarray,
    transition_model: FeedbackLoop,
    log_observation: Callable,
):
    ll = transition_model.logpdf(state, next_state) \
         + log_observation(next_state)
    return ll


