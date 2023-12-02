import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.openloop import cartpole_env as cartpole

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

init_state = jnp.zeros((5,))
tempering = 0.5

nb_iter = 100
learning_rate = 1e-1

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=cartpole.module,
    init_data=jnp.zeros((4,)),
    learning_rate=learning_rate,
    optimizer=optax.sgd
)

state = init_state

trajectory = jnp.zeros((nb_steps, 5))
trajectory = trajectory.at[0, :].set(state)

for t in range(nb_steps):
    key, sub_key = jr.split(key, 2)
    reference = rollout(
        sub_key,
        horizon,
        state,
        opt_state.params,
        tempering,
        cartpole,
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
            cartpole
        )
    x = sample[0, :4]
    u = sample[1, -1:]

    key, sub_key = jr.split(key, 2)
    xn = cartpole.dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

plt.plot(trajectory[:, :-1])
plt.show()
