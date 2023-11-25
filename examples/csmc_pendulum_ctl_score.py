from typing import Dict
from functools import partial

import jax
from jax import random as jr
from jax import numpy as jnp

from flax import linen as nn
from flax.training.train_state import TrainState

import optax

from psoc.algorithms import smc, csmc
from psoc.environments import pendulum_env as pendulum

jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


def create_train_state(
    key: jax.Array,
    module: nn.Module,
    learning_rate: float
):
    init_data = jnp.zeros((2,))
    params = module.init(key, init_data)["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx
    )


key = jr.PRNGKey(5121)

nb_steps = 101
nb_particles = 16

eta = 0.1

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(sub_key, pendulum.network, 5e-4)

prior, closedloop, cost = \
    pendulum.create_env(opt_state.params, eta)

key, sub_key = jr.split(key, 2)
reference = smc(
    sub_key,
    nb_steps,
    nb_particles,
    prior,
    closedloop,
    cost,
)


def loss_fn(state, next_state, params):
    _, _closedloop, _log_obsrv = \
        pendulum.create_env(params, eta)
    loss = pendulum.log_complete_likelihood(state, next_state, _closedloop, _log_obsrv)
    return - loss


score_fn = lambda x, xn, p: jax.grad(loss_fn, 2)(x, xn, p)


@partial(jax.jit, static_argnums=(1, 2))
def sample(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    reference: jnp.ndarray,
    params: Dict,
    eta: float
):
    prior, closedloop, cost = \
        pendulum.create_env(params, eta)

    next_reference, score = csmc(
        sub_key,
        nb_steps,
        nb_particles,
        reference,
        prior,
        closedloop,
        cost,
        score_fn,
        opt_state.params,
    )
    return next_reference, score


key, sub_key = jr.split(key, 2)
_, _ = sample(
    sub_key,
    nb_steps,
    nb_particles,
    reference,
    opt_state.params,
    eta,
)
