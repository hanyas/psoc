import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.algorithms import smc
from psoc.environments.feedback import linear_env as linear

from psoc.common import batcher
from psoc.common import create_train_state
from psoc.common import csmc_sampling
from psoc.common import maximization
from psoc.common import simulate

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(83453)

nb_steps = 51
nb_particles = 32
nb_samples = 10

nb_iter = 100
eta = 0.1

lr = 5e-3
batch_size = 32

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=linear.network,
    init_data=jnp.zeros((2,)),
    learning_rate=lr,
)

prior, closedloop, reward = \
    linear.create_env(opt_state.params, eta)

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

    # expectation step
    samples = csmc_sampling(
        estep_key,
        nb_steps,
        nb_particles,
        nb_samples,
        reference,
        opt_state.params,
        eta,
        linear
    )

    # maximization step
    loss = 0.0
    batches = batcher(mstep_key, samples, batch_size)
    for batch in batches:
        states, next_states = batch
        opt_state, batch_loss = \
            maximization(opt_state, states, next_states, eta, linear)
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
    linear.create_env(opt_state.params, eta)
states = simulate(prior, closedloop, nb_steps)

plt.figure()
plt.plot(states)
plt.show()
