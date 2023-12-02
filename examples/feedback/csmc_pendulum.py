import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.feedback import pendulum_env as pendulum
from psoc.common import initialize, rollout
from psoc.optimization import score_optimization

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
learning_rate = 5e-4
batch_size = 32

key, sub_key = jr.split(key, 2)
opt_state, reference = initialize(
    sub_key,
    nb_steps,
    nb_particles,
    init_state,
    tempering,
    2,
    learning_rate,
    pendulum
)

key, sub_key = jr.split(key, 2)
opt_state = score_optimization(
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
    pendulum
)

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
