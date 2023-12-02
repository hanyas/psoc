import jax
from jax import random as jr
from jax import numpy as jnp

from psoc.environments.feedback import const_linear_env as linear

from psoc.sampling import smc_sampling, csmc_sampling
from psoc.utils import batcher, create_train_state
from psoc.common import maximization, rollout

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


key = jr.PRNGKey(83453)

nb_steps = 51
nb_particles = 16
nb_samples = 25

init_state = jnp.array([1.0, 2.0, 0.0])
tempering = 0.1

nb_iter = 100
lr = 5e-3
batch_size = 64

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=linear.network,
    init_data=jnp.zeros((2,)),
    learning_rate=lr,
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
    linear
)[0]

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
        init_state,
        opt_state.params,
        tempering,
        linear
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
            linear
        )
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

key, sub_key = jr.split(key, 2)
sample = rollout(
    sub_key,
    nb_steps,
    init_state,
    opt_state.params,
    tempering,
    linear,
)

plt.plot(sample)
plt.show()