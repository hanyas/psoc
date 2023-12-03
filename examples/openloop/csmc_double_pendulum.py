import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.openloop import double_pendulum_env as double_pendulum

from psoc.common import rollout
from psoc.utils import create_train_state
from psoc.optimization import rao_blackwell_score_optimization

import optax
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(1513)

nb_steps = 101
horizon = 35

nb_particles = 64
nb_samples = 64

init_state = jnp.zeros((6,))
tempering = 5e-3

nb_iter = 250
learning_rate = 1e-1

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=double_pendulum.module,
    init_data=jnp.zeros((2,)),
    learning_rate=learning_rate,
    optimizer=optax.sgd
)

state = init_state

trajectory = jnp.zeros((nb_steps, 6))
trajectory = trajectory.at[0, :].set(state)

for t in range(nb_steps):
    key, sub_key = jr.split(key, 2)
    reference = rollout(
        sub_key,
        horizon,
        state,
        opt_state.params,
        tempering,
        double_pendulum,
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
            double_pendulum
        )
    x = sample[0, :4]
    u = sample[1, -2:]

    key, sub_key = jr.split(key, 2)
    xn = double_pendulum.dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

plt.plot(trajectory[:, :-2])
plt.show()
