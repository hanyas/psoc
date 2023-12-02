import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.feedback import double_pendulum_env as double_pendulum
from psoc.common import initialize, rollout
from psoc.experiments import feedback_experiment

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(23123)

nb_steps = 101
nb_particles = 64
nb_samples = 32

init_state = jnp.zeros((6,))
tempering = 5e-3

nb_iter = 100
learning_rate = 1e-4
batch_size = 32

key, sub_key = jr.split(key, 2)
opt_state, reference = initialize(
    sub_key,
    nb_steps,
    nb_particles,
    init_state,
    tempering,
    4,
    learning_rate,
    double_pendulum
)

key, sub_key = jr.split(key, 2)
opt_state = feedback_experiment(
    sub_key,
    nb_iter,
    nb_steps,
    nb_particles,
    nb_samples,
    reference,
    init_state,
    opt_state,
    tempering,
    batch_size,
    double_pendulum
)

key, sub_key = jr.split(key, 2)
sample = rollout(
    sub_key,
    nb_steps,
    init_state,
    opt_state.params,
    tempering,
    double_pendulum,
)

plt.plot(sample[:, :-2])
plt.show()
