from typing import Callable
from functools import partial

import jax
from jax import random as jr
from jax import numpy as jnp
from jax import lax as jl

from jax.scipy.special import logsumexp

import distrax

from psoc.abstract import OpenLoop


def mppi(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    prior_dist: distrax.Distribution,
    transition_model: OpenLoop,
    reward_fn: Callable,
):
    def body(carry, args):
        key, prev_particles = carry

        key, sub_key = jr.split(key, 2)
        next_particles = transition_model.sample(sub_key, prev_particles)
        return (key, next_particles), prev_particles

    key, init_key, scan_key = jr.split(key, 3)
    init_particles = prior_dist.sample(
        seed=init_key,
        sample_shape=(nb_particles,)
    )

    (_, last_particles), particles = \
        jl.scan(
            body,
            (scan_key, init_particles),
            (),
            length=nb_steps-1
        )

    particles = jnp.insert(particles, nb_steps, last_particles, 0)

    log_weights = jnp.sum(reward_fn(particles), axis=0)
    log_weights_norm = logsumexp(log_weights)
    weights = jnp.exp(log_weights - log_weights_norm)

    return particles, weights
