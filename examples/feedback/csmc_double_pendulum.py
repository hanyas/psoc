import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.feedback import double_pendulum_env as double_pendulum

from psoc.sampling import smc_sampling, csmc_sampling
from psoc.utils import batcher, create_train_state
from psoc.common import maximization, rollout

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
lr = 1e-4
batch_size = 32

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=double_pendulum.network,
    init_data=jnp.zeros((4,)),
    learning_rate=lr
)

key, sub_key = jr.split(key, 2)
reference = smc_sampling(
    sub_key,
    nb_steps,
    int(10 * nb_particles),
    1,
    init_state,
    opt_state.params,
    tempering,
    double_pendulum
)[0]

# plt.plot(reference[:, :])
# plt.show()

for i in range(nb_iter):
    key, sample_key, max_key = jr.split(key, 3)

    # sampling step
    samples = csmc_sampling(
        sample_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        init_state,
        opt_state.params,
        tempering,
        double_pendulum
    )

    # maximization step
    loss = 0.0
    batches = batcher(max_key, samples, batch_size)
    for batch in batches:
        states, next_states = batch
        opt_state, batch_loss = maximization(
            states,
            next_states,
            init_state,
            opt_state,
            tempering,
            double_pendulum
        )
        loss += batch_loss

    print(
        f" iter: {i},"
        f" loss: {loss},"
        f" log_std: {opt_state.params['log_std']}"
    )

    # choose new reference
    reference = samples[-1]


# for n in range(nb_particles):
#     plt.plot(samples[n, :, :])
# plt.show()

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