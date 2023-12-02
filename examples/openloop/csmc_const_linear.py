import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.openloop import const_linear_env as linear

from psoc.common import rollout
from psoc.utils import create_train_state
from psoc.optimization import rao_blackwell_score_optimization

import optax
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

nb_iter = 100
learning_rate = 1e-1

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=linear.module,
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
        linear,
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
            linear
        )
    x = sample[0, :2]
    u = sample[1, -1:]

    key, sub_key = jr.split(key, 2)
    xn = linear.dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

plt.plot(trajectory)
plt.show()
