from typing import Dict
from functools import partial

import jax

from jax import random as jr
from jax import numpy as jnp

import numpy as onp

from flax import struct
from flax import linen as nn
from flax.training import train_state

import optax
from clu import metrics

from psoc.algorithms import smc, csmc
from psoc.environments import pendulum_env as pendulum

import matplotlib.pyplot as plt
import time as clock

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


@partial(jax.jit, static_argnums=1)
def expectation(
    key: jax.Array,
    nb_samples: int,
    reference: jnp.ndarray,
    params: Dict,
    eta: float,
):
    prior, closedloop, log_obsrv = \
        pendulum.create_env(params, eta)

    def body(carry, args):
        key, reference = carry
        key, sub_key = jr.split(key, 2)
        sample = csmc(
            sub_key, nb_steps, nb_particles, 1,
            reference, prior, closedloop, log_obsrv,
        )[-1]
        return (key, sample), sample

    _, samples = \
        jax.lax.scan(body, (key, reference), (), length=nb_samples)
    return samples


def lower_bound(
    states: jnp.ndarray,
    next_states: jnp.ndarray,
    params: Dict,
    eta: float,
):
    _, closedloop, log_obsrv = \
        pendulum.create_env(params, eta)

    lls = pendulum.log_complete_likelihood(
        states, next_states, closedloop, log_obsrv
    )
    return -jnp.sum(lls)


@jax.jit
def maximization(
    opt_state: train_state.TrainState,
    states: jnp.ndarray,
    next_states: jnp.ndarray,
    eta: float,
):
    def loss_fn(params):
        loss = lower_bound(
            states, next_states,
            params, eta
        )
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(opt_state.params)
    return opt_state.apply_gradients(grads=grads)


def create_pairs(samples: jnp.ndarray):
    nb_samples, nb_steps, dim = samples.shape
    states, next_states = samples[:, :-1, :], samples[:, 1:, :]
    states = jnp.reshape(states, (nb_samples * (nb_steps - 1), dim))
    next_states = jnp.reshape(next_states, (nb_samples * (nb_steps - 1), dim))
    return states, next_states


def compute_cost(
    samples: jnp.ndarray,
    params: Dict,
    eta: float
):
    _, _, cost = \
        pendulum.create_env(params, eta)
    _, next_states = create_pairs(samples)
    return jnp.mean(jax.vmap(cost)(next_states))


def batcher(
    key: jax.Array,
    samples: jnp.ndarray,
    batch_size: int,
):
    states, next_states = create_pairs(samples)

    # Shuffle
    batch_idx = jr.permutation(key, len(states))
    batch_idx = onp.asarray(batch_idx)

    # Skip incomplete batch
    steps_per_epoch = len(states) // batch_size
    batch_idx = batch_idx[: steps_per_epoch * batch_size]
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        yield states[idx], next_states[idx]


@jax.jit
def compute_metrics(
    opt_state: train_state.TrainState,
    samples: jnp.ndarray,
    eta: float
):
    loss = compute_cost(samples, opt_state.params, eta)
    metric_updates = opt_state.metrics.single_from_model_output(loss=loss)
    metrics = opt_state.metrics.merge(metric_updates)
    opt_state = opt_state.replace(metrics=metrics)
    return opt_state


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(
    key: jax.Array,
    module: nn.Module,
    learning_rate: float
):
    params = module.init(key, jnp.zeros((2,)))["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params,
        tx=tx, metrics=Metrics.empty()
    )


key = jr.PRNGKey(123)

nb_steps = 101
nb_particles = 512
nb_samples = 20

nb_iter = 15
init_eta = 1.0
final_eta = 1.0
log_eta = jnp.linspace(jnp.log(init_eta), jnp.log(final_eta), nb_iter)

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(sub_key, pendulum.network, 1e-3)

start = clock.time()

prior, closedloop, cost = \
    pendulum.create_env(opt_state.params, init_eta)

key, sub_key = jr.split(key, 2)
samples, weights = smc(
    sub_key, nb_steps, nb_particles, nb_samples,
    prior, closedloop, cost,
)

key, sub_key = jr.split(key, 2)
idx = jr.choice(sub_key, a=nb_samples)
reference = samples[idx, :, :]

# for n in range(nb_samples):
#     plt.plot(samples[n, :, 0])
# plt.show()

metrics_history = {'loss': []}

for i in range(nb_iter):
    # update temperature
    eta = jnp.exp(log_eta[i])

    key, estep_key, mstep_key, ref_key = jr.split(key, 4)

    # expectation step
    samples = expectation(
        estep_key, nb_samples, reference, opt_state.params, eta
    )

    # maximization step
    batches = batcher(mstep_key, samples, 64)
    for batch in batches:
        states, next_states = batch
        opt_state = maximization(
            opt_state, states, next_states, eta
        )

    # compute lower bound
    opt_state = compute_metrics(opt_state, samples, eta)

    for metric, value in opt_state.metrics.compute().items():
        metrics_history[f'{metric}'].append(value)
    opt_state = opt_state.replace(metrics=opt_state.metrics.empty())

    print(
        f"iter: {i},"
        f" loss: {metrics_history['loss'][-1]},"
        f" log_std: {opt_state.params['log_std']}"
    )

    # choose new reference
    reference = samples[-1]


jax.block_until_ready(opt_state)
end = clock.time()
print("Compilation + Execution Time:", end - start)

for n in range(nb_particles):
    plt.plot(samples[n, :, :])
plt.show()

prior, closedloop, _ = \
    pendulum.create_env(opt_state.params, final_eta)
states = pendulum.simulate(prior, closedloop, nb_steps)

plt.plot(states)
plt.show()
