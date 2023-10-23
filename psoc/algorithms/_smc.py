from typing import Callable

import jax
from jax import random as jr
from jax import numpy as jnp
from jax import lax as jl

from jax.scipy.special import logsumexp

import distrax

from psoc.abstract import ClosedLoop


def _backward_tracing(
    key: jax.Array,
    filter_particles: jnp.ndarray,
    filter_ancestors: jnp.ndarray,
    filter_weights: jnp.ndarray,
):
    nb_particles = filter_particles.shape[1]

    # last time step
    key, sub_key = jr.split(key, 2)
    b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
    last_smoother_state = filter_particles[-1, b]
    last_smoother_weight = filter_weights[-1, b]

    def body(carry, args):
        next_b = carry
        particles, ancestors = args
        b = ancestors[next_b]
        return b, particles[b]

    _, smoother_sample = jl.scan(
        body, b, (filter_particles[:-1], filter_ancestors), reverse=True
    )

    smoother_sample = jnp.vstack((smoother_sample, last_smoother_state))
    return smoother_sample, last_smoother_weight


def _backward_sampling(
    key: jax.Array,
    filter_particles: jnp.ndarray,
    filter_weights: jnp.ndarray,
    transition_model: ClosedLoop
):
    nb_particles = filter_particles.shape[1]
    trans_logpdf = jax.vmap(transition_model.logpdf, in_axes=(0, None))

    # last time step
    key, sub_key = jr.split(key, 2)
    b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
    last_smoother_state = filter_particles[-1, b]
    last_smoother_weight = filter_weights[-1, b]

    def body(carry, args):
        key, next_state = carry
        particles, weights = args

        log_mod_weights = jnp.log(weights) + trans_logpdf(particles, next_state)
        log_mod_weights_norm = logsumexp(log_mod_weights)
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        key, sub_key = jr.split(key, 2)
        b = jr.choice(sub_key, a=nb_particles, p=mod_weights)
        state = particles[b]
        return (key, state), next_state

    (_, first_smoother_state), smoother_sample = \
        jl.scan(
            body,
            (key, last_smoother_state),
            (filter_particles[:-1], filter_weights[:-1]),
            reverse=True,
    )

    smoother_sample = jnp.vstack((first_smoother_state, smoother_sample))
    return smoother_sample, last_smoother_weight


def smc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    prior: distrax.Distribution,
    transition_model: ClosedLoop,
    log_observation: Callable
):
    def _propagate(key, particles):
        return transition_model.sample(key, particles)

    def _log_weights(state):
        return log_observation(state)

    def body(carry, args):
        key, prev_particles, prev_weights = carry

        # resample
        key, sub_key = jr.split(key, 2)
        ancestors = jr.choice(key, a=nb_particles,
                              shape=(nb_particles,), p=prev_weights)

        # propagate
        key, sub_key = jr.split(key, 2)
        resampled_particles = prev_particles[ancestors]
        next_particles = _propagate(sub_key, resampled_particles)

        # weights
        log_next_weights = _log_weights(next_particles)
        next_weights_norm = logsumexp(log_next_weights)
        next_weights = jnp.exp(log_next_weights - next_weights_norm)

        return (key, next_particles, next_weights), \
            (prev_particles, prev_weights, ancestors)

    key, sub_key = jr.split(key, 2)
    init_particles = prior.sample(seed=sub_key, sample_shape=(nb_particles,))
    init_weights = jnp.ones((nb_particles,)) / nb_particles

    (key, last_particles, last_weights), \
        (filter_particles, filter_weights, filter_ancestors) = \
        jl.scan(
            body,
            (key, init_particles, init_weights),
            (),
            length=nb_steps-1
        )

    filter_particles = jnp.insert(filter_particles, nb_steps, last_particles, 0)
    filter_weights = jnp.insert(filter_weights, nb_steps, last_weights, 0)

    def body(carry, args):
        key = carry
        key, sub_key = jr.split(key, 2)

        # smoother_path_sample, sample_weight = _backward_sampling(
        #     sub_key,
        #     filter_particles,
        #     filter_weights,
        #     transition_model
        # )

        smoother_path_sample, sample_weight = _backward_tracing(
            sub_key,
            filter_particles,
            filter_ancestors,
            filter_weights
        )
        return key, (smoother_path_sample, sample_weight)

    _, (smoother_samples, smoother_weights) = \
        jl.scan(body, key, (), length=nb_samples)

    return smoother_samples, smoother_weights
