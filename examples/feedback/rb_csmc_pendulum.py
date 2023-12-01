from typing import Dict
from functools import partial

import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.algorithms import smc, rb_csmc
from psoc.environments.feedback import pendulum_env as pendulum

from psoc.common import create_train_state
from psoc.common import log_complete_likelihood
from psoc.common import rollout
from psoc.common import smc_sampling

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(1337)

nb_steps = 101
nb_particles = 64
nb_samples = 20

init_state = jnp.zeros((3,))
tempering = 0.25

nb_iter = 100
lr = 5e-4

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=pendulum.network,
    init_data=jnp.zeros((2,)),
    learning_rate=lr
)

prior, closedloop, reward_fn = \
    pendulum.create_env(init_state, opt_state.params, tempering)

key, sub_key = jr.split(key, 2)
samples, _ = smc(
    sub_key,
    nb_steps,
    int(10 * nb_particles),
    1,
    prior,
    closedloop,
    reward_fn,
)
reference = samples[0]

# plt.plot(reference)
# plt.show()


def loss_fn(
    state: jnp.ndarray,
    next_state: jnp.ndarray,
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float
):
    _, closedloop, rwrd_fn = pendulum.create_env(init_state, parameters, tempering)
    loss = log_complete_likelihood(state, next_state, closedloop, rwrd_fn)
    return - 1.0 * loss


@partial(jax.jit, static_argnums=(1, 2, 3, -1))
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

    prior, closedloop, rwrd_fn = \
        environment.create_env(init_state, parameters, tempering)

    loss_fn_ = lambda x, xn, p: loss_fn(x, xn, init_state, p, tempering)
    score_fn_ = lambda x, xn, p: jax.grad(loss_fn, 3)(x, xn, init_state, p, tempering)

    key, sub_key = jr.split(key, 2)
    reference, loss, score = rb_csmc(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        prior,
        closedloop,
        rwrd_fn,
        loss_fn_,
        score_fn_,
        parameters
    )
    return reference, loss, score


for i in range(nb_iter):
    key, sub_key = jr.split(key, 2)
    reference, loss, score = compute_score(
        sub_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        init_state,
        opt_state.params,
        tempering,
        pendulum
    )

    opt_state = opt_state.apply_gradients(grads=score)

    print(
        f" iter: {i},"
        f" loss: {loss},"
        f" log_std: {opt_state.params['log_std']}"
    )

# for n in range(nb_samples):
#     plt.plot(samples[n, :, :])
# plt.show()

key, sub_key = jr.split(key, 2)
sample = rollout(
    sub_key,
    nb_steps,
    init_state,
    opt_state.params,
    tempering,
    pendulum,
)

plt.plot(sample)
plt.show()
