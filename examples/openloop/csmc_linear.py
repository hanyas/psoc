import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.openloop import linear_env as linear

from psoc.common import rollout
from psoc.common import csmc_sampling

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(83453)

nb_steps = 51
horizon = 15

nb_particles = 16
nb_samples = 10

init_state = jnp.array([1.0, 2.0, 0.0])
tempering = 0.5

key, sub_key = jr.split(key, 2)
parameters = linear.module.init(
    sub_key, jnp.zeros((2,))
)["params"]

state = init_state

trajectory = jnp.zeros((nb_steps, 3))
trajectory = trajectory.at[0, :].set(state)

for t in range(nb_steps):
    key, sub_key = jr.split(key, 2)
    reference = rollout(
        sub_key,
        horizon,
        state,
        parameters,
        tempering,
        linear,
    )

    key, sub_key = jr.split(key, 2)
    sample = csmc_sampling(
        sub_key,
        horizon,
        nb_particles,
        nb_samples,
        reference,
        state,
        parameters,
        tempering,
        linear
    )[-1]

    x = sample[0, :2]
    u = sample[1, -1:]

    key, sub_key = jr.split(key, 2)
    xn = linear.dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

plt.plot(trajectory)
plt.show()
