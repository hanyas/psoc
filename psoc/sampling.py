from functools import partial
from typing import Dict

import jax
from jax import numpy as jnp
from jax import random as jr

from psoc.algorithms import vanilla_csmc, smc


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
    environment,
):
    prior, loop, reward_fn = \
        environment.create_env(init_state, parameters, tempering)

    def body(carry, args):
        key, reference = carry
        key, sub_key = jr.split(key, 2)
        sample = vanilla_csmc(
            sub_key,
            nb_steps,
            nb_particles,
            reference,
            prior,
            loop,
            reward_fn,
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
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
    environment,
):
    prior, loop, reward_fn = \
        environment.create_env(init_state, parameters, tempering)

    key, sub_key = jr.split(key, 2)
    samples = smc(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        prior,
        loop,
        reward_fn,
    )
    return samples
