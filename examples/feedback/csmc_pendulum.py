import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.algorithms import smc
from psoc.environments.feedback import pendulum_env as pendulum

from psoc.common import batcher
from psoc.common import create_train_state
from psoc.common import csmc_sampling
from psoc.common import maximization
from psoc.common import simulate

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(5121)

nb_steps = 101
nb_particles = 16
nb_samples = 50

nb_iter = 25
eta = 0.1

lr = 1e-3
batch_size = 32

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=pendulum.network,
    init_data=jnp.zeros((2,)),
    learning_rate=lr
)

prior, closedloop, reward = \
    pendulum.create_env(opt_state.params, eta)

key, sub_key = jr.split(key, 2)
samples, weights = smc(
    sub_key,
    nb_steps,
    int(nb_particles * 10),
    int(nb_particles * 10),
    prior,
    closedloop,
    reward,
)
key, sub_key = jr.split(key, 2)
idx = jr.choice(sub_key, a=int(nb_particles * 10), p=weights)
reference = samples[idx, ...]

# plt.plot(reference)
# plt.show()

for i in range(nb_iter):
    key, estep_key, mstep_key = jr.split(key, 3)

    # sampling step
    samples = csmc_sampling(
        estep_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        opt_state.params,
        eta,
        pendulum
    )

    # maximization step
    loss = 0.0
    batches = batcher(mstep_key, samples, batch_size)
    for batch in batches:
        states, next_states = batch
        opt_state, batch_loss = \
            maximization(opt_state, states, next_states, eta, pendulum)
        loss += batch_loss

    print(
        f" iter: {i},"
        f" loss: {loss},"
        f" log_std: {opt_state.params['log_std']}"
    )

    # choose new reference
    reference = samples[-1]

# for n in range(nb_samples):
#     plt.plot(samples[n, :, :])
# plt.show()

prior, closedloop, _ = \
    pendulum.create_env(opt_state.params, eta)
states = simulate(prior, closedloop, nb_steps)

plt.figure()
plt.plot(states)
plt.legend([r"$\theta$", r"$d\theta/dt$", "u"])
plt.title("Sample closed loop trajectory")
plt.show()
