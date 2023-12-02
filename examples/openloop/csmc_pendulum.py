import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.openloop import pendulum_env as pendulum

from psoc.common import rollout
from psoc.utils import create_train_state
from psoc.optimization import rao_blackwell_score_optimization

import optax
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(8)

nb_steps = 101
horizon = 20

nb_particles = 32
nb_samples = 16

init_state = jnp.zeros((3,))
tempering = 0.75

nb_iter = 100
learning_rate = 1e-1

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=pendulum.module,
    init_data=jnp.zeros((2,)),
    learning_rate=learning_rate,
    optimizer=optax.sgd
)

state = init_state

trajectory = jnp.zeros((nb_steps, 3))
trajectory = trajectory.at[0, :].set(state)

for t in range(nb_steps):
    key, sub_key = jr.split(key, 2)
    reference = rollout(
        sub_key,
        horizon,
        state,
        opt_state.params,
        tempering,
        pendulum,
    )

    key, sub_key = jr.split(key, 2)
    opt_state, sample, _ = \
        rao_blackwell_score_optimization(
            sub_key,
            nb_iter,
            horizon,
            nb_particles,
            nb_samples,
            reference,
            state,
            opt_state,
            tempering,
            pendulum
        )
    x = sample[0, :2]
    u = sample[1, -1:]

    key, sub_key = jr.split(key, 2)
    xn = pendulum.dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

plt.plot(trajectory)
plt.show()
