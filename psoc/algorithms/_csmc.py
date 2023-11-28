from typing import Callable

import jax
from jax import random as jr
from jax import numpy as jnp
from jax import lax as jl

from jax.scipy.special import logsumexp

import distrax

from psoc.abstract import FeedbackLoop


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
    last_ref = filter_particles[-1, b]

    def body(carry, args):
        next_b = carry
        particles, ancestors = args
        b = ancestors[next_b]
        return b, particles[b]

    _, reference = jl.scan(
        body, b, (filter_particles[:-1], filter_ancestors), reverse=True
    )

    reference = jnp.vstack((reference, last_ref))
    return reference


def _backward_sampling(
    key: jax.Array,
    filter_particles: jnp.ndarray,
    filter_weights: jnp.ndarray,
    transition_model: FeedbackLoop,
):
    nb_particles = filter_particles.shape[1]
    trans_logpdf = jax.vmap(transition_model.logpdf, in_axes=(0, None))

    # last time step
    key, sub_key = jr.split(key, 2)
    b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
    last_ref = filter_particles[-1, b]

    def body(carry, args):
        key, next_ref = carry
        particles, weights = args

        log_mod_weights = jnp.log(weights) + trans_logpdf(particles, next_ref)
        log_mod_weights_norm = logsumexp(log_mod_weights)
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        key, sub_key = jr.split(key, 2)
        b = jr.choice(sub_key, a=nb_particles, p=mod_weights)
        ref = particles[b]
        return (key, ref), next_ref

    (_, first_ref), reference = \
        jl.scan(
            body,
            (key, last_ref),
            (filter_particles[:-1], filter_weights[:-1]),
            reverse=True
        )

    reference = jnp.vstack((first_ref, reference))
    return reference


def csmc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    reference: jnp.ndarray,
    prior: distrax.Distribution,
    transition_model: FeedbackLoop,
    log_observation: Callable,
):
    def _propagate(key, particles):
        return transition_model.sample(key, particles)

    def _log_weights(state):
        return log_observation(state)

    def body(carry, args):
        prev_particles, prev_weights = carry
        next_ref, res_key, prop_key = args

        rest_ancestors = jr.choice(res_key, a=nb_particles,
                                   shape=(nb_particles - 1,), p=prev_weights)
        ancestors = jnp.hstack((0, rest_ancestors))

        resampled_particles = prev_particles[rest_ancestors]
        next_rest_particles = _propagate(prop_key, resampled_particles)
        next_particles = jnp.insert(next_rest_particles, 0, next_ref, 0)

        log_next_weights = _log_weights(next_particles)
        next_weights_norm = logsumexp(log_next_weights)
        next_weights = jnp.exp(log_next_weights - next_weights_norm)

        return (next_particles, next_weights), \
            (prev_particles, prev_weights, ancestors)

    key, sub_key = jr.split(key, 2)
    init_rest_particles = prior.sample(seed=sub_key, sample_shape=(nb_particles - 1,))
    init_particles = jnp.insert(init_rest_particles, 0, reference[0], 0)
    init_weights = jnp.ones((nb_particles,)) / nb_particles

    keys = jr.split(key, nb_steps)
    key, resampling_keys = keys[0], keys[1:]

    keys = jr.split(key, nb_steps)
    key, propagation_keys = keys[0], keys[1:]

    (last_particles, last_weights), \
        (filter_particles, filter_weights, filter_ancestors) = \
        jl.scan(
            body,
            (init_particles, init_weights),
            (reference[1:], resampling_keys, propagation_keys)
        )

    filter_particles = jnp.insert(filter_particles, nb_steps, last_particles, 0)
    filter_weights = jnp.insert(filter_weights, nb_steps, last_weights, 0)

    key, sub_key = jr.split(key, 2)
    sample = _backward_sampling(
        sub_key,
        filter_particles,
        filter_weights,
        transition_model
    )

    return sample
