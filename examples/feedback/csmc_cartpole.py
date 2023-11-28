import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.algorithms import smc
from psoc.environments.feedback import cartpole_env as cartpole

from psoc.common import batcher
from psoc.common import create_train_state
from psoc.common import csmc_sampling
from psoc.common import maximization
from psoc.common import simulate

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(98123)

nb_steps = 101
nb_particles = 32
nb_samples = 10

nb_iter = 50
eta = 0.1

lr = 5e-4
batch_size = 64

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=cartpole.network,
    init_data=jnp.zeros((4,)),
    learning_rate=lr
)

prior, closedloop, reward = \
    cartpole.create_env(opt_state.params, eta)

key, sub_key = jr.split(key, 2)
samples, weights = smc(
    sub_key,
    nb_steps,
    int(nb_particles * 10),
    nb_samples,
    prior,
    closedloop,
    reward,
)
key, sub_key = jr.split(key, 2)
idx = jr.choice(sub_key, a=nb_samples, p=weights)
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
        cartpole
    )

    # maximization step
    loss = 0.0
    batches = batcher(mstep_key, samples, batch_size)
    for batch in batches:
        states, next_states = batch
        opt_state, batch_loss = \
            maximization(opt_state, states, next_states, eta, cartpole)
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

prior, closedloop, _ = \
    cartpole.create_env(opt_state.params, eta)
states = simulate(prior, closedloop, nb_steps)

plt.figure()
plt.plot(states[:, :-1])
plt.legend(["x", r"$\theta$", "dx", r"$d\theta/dt$", "u"])
plt.title("Sample closed loop trajectory")
plt.show()
