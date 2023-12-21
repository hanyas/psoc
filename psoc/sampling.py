from functools import partial
from typing import Dict, Callable

import jax
from jax import numpy as jnp
from jax import random as jr

from psoc.algorithms import csmc
from psoc.algorithms import smc

from psoc.algorithms import rao_blackwell_csmc


@partial(jax.jit, static_argnums=(1, 2, 3, -1))
def smc_sampling(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    make_env: Callable
):
    prior_dist, loop_obj, reward_fn = \
        make_env(init_state, parameters, tempering)

    key, sub_key = jr.split(key, 2)
    samples = smc(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        prior_dist,
        loop_obj,
        reward_fn,
    )
    return samples


@partial(jax.jit, static_argnums=(1, 2, 3, -1))
def csmc_sampling(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    make_env: Callable
):
    prior_dist, loop_obj, reward_fn = \
        make_env(init_state, parameters, tempering)

    def body(carry, args):
        key, reference = carry
        key, sub_key = jr.split(key, 2)
        sample = csmc(
            sub_key,
            nb_steps,
            nb_particles,
            reference,
            prior_dist,
            loop_obj,
            reward_fn,
        )
        return (key, sample), sample

    key, sub_key = jr.split(key, 2)
    _, samples = \
        jax.lax.scan(body, (sub_key, reference), (), length=nb_samples)
    return samples, samples[-1]


@partial(jax.jit, static_argnums=(1, 2, 3, -1))
def rb_csmc_sampling(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    make_env: Callable
):
    prior_dist, loop_obj, reward_fn = \
        make_env(init_state, parameters, tempering)

    key, sub_key = jr.split(key, 2)
    samples, reference = rao_blackwell_csmc(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        prior_dist,
        loop_obj,
        reward_fn,
    )
    return jnp.swapaxes(samples, 0, 1), reference
