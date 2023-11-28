import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.algorithms import smc
from psoc.environments.openloop import linear_env as linear

from psoc.common import batcher
from psoc.common import create_train_state
from psoc.common import csmc_sampling
from psoc.common import maximization

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(83453)

nb_steps = 51
nb_particles = 32
nb_samples = 1

nb_iter = 5000
eta = 0.1

lr = 1e-3
batch_size = 50

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=linear.module,
    init_data=jnp.zeros((1,)),
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
    key, sample_key, max_key = jr.split(key, 3)

    # sampling step
    samples = csmc_sampling(
        sample_key,
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
    batches = batcher(max_key, samples, batch_size)
    for batch in batches:
        states, next_states = batch
        opt_state, batch_loss = \
            maximization(opt_state, states, next_states, eta, linear)
        loss += batch_loss

    print(
        f" iter: {i},"
        f" loss: {loss},"
    )

    # choose new reference
    reference = samples[-1]


nb_rollouts = 10000

prior, closedloop, reward = \
    linear.create_env(opt_state.params, eta)

key, sub_key = jr.split(key, 2)
samples = csmc_sampling(
    sub_key,
    nb_steps,
    nb_particles,
    nb_rollouts,
    reference,
    opt_state.params,
    eta,
    linear
)

plt.plot(jnp.mean(samples, axis=0))
plt.show()
