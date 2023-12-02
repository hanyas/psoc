import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.feedback import const_linear_env as linear
from psoc.common import initialize, rollout
from psoc.experiments import rao_blackwell_feedback_experiment

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(83453)

nb_steps = 51
nb_particles = 16
nb_samples = 16

init_state = jnp.array([1.0, 2.0, 0.0])
tempering = 0.1

nb_iter = 1000
learning_rate = 1e-2

key, sub_key = jr.split(key, 2)
opt_state, reference = initialize(
    sub_key,
    nb_steps,
    nb_particles,
    init_state,
    tempering,
    2,
    learning_rate,
    linear
)

key, sub_key = jr.split(key, 2)
opt_state, _ = rao_blackwell_feedback_experiment(
    sub_key,
    nb_iter,
    nb_steps,
    nb_particles,
    nb_samples,
    reference,
    init_state,
    opt_state,
    tempering,
    linear
)

key, sub_key = jr.split(key, 2)
sample = rollout(
    sub_key,
    nb_steps,
    init_state,
    opt_state.params,
    tempering,
    linear,
)

plt.plot(sample)
plt.show()
