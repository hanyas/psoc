from typing import Dict
from functools import partial

import jax
from jax import random as jr
from jax import numpy as jnp

import distrax
from flax import linen as nn

from psoc.abstract import StochasticDynamics
from psoc.abstract import Network
from psoc.abstract import FeedbackPolicyWithSquashing
from psoc.abstract import FeedbackLoop
from psoc.bijector import Tanh

from psoc.environments.feedback import double_pendulum_env as double_pendulum
from psoc.experiments.feedback.common import smc_experiment

from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


dynamics = StochasticDynamics(
    dim=4,
    ode=double_pendulum.ode,
    step=0.05,
    stddev=1e-2 * jnp.ones((4,))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def polar(x):
    sin_q, cos_q = jnp.sin(x[0]), jnp.cos(x[0])
    sin_p, cos_p = jnp.sin(x[1]), jnp.cos(x[1])
    return jnp.hstack([sin_q, cos_q, sin_p, cos_p, x[2], x[3]])


network = Network(
    dim=2,
    layer_size=[256, 256],
    transform=polar,
    activation=nn.relu,
    init_log_std=nn.initializers.constant(1.5)
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 25.0),
    Tanh()
])


def make_env(
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
):
    prior_dist = distrax.MultivariateNormalDiag(
        loc=init_state,
        scale_diag=jnp.ones((6,)) * 1e-4
    )

    policy = FeedbackPolicyWithSquashing(
        network, bijector, parameters
    )

    loop_obj = FeedbackLoop(
        dynamics, policy
    )

    reward_fn = lambda z: double_pendulum.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


init_state = jnp.zeros((6,))

batch_smc_exp = lambda exp_key: smc_experiment(
    key=exp_key,
    state_dim=4,
    action_dim=2,
    dynamics=dynamics,
    network=network,
    bijector=bijector,
    make_env=make_env,
    nb_steps=101,
    nb_particles=512,
    nb_samples=30,
    init_state=init_state,
    tempering=5e-3,
    nb_iter=300,
    learning_rate=5e-4,
)

key = jr.PRNGKey(1)

nb_experiments = 10
batch_keys = jr.split(key, nb_experiments)
_, reward = jax.vmap(batch_smc_exp)(batch_keys)

filt_reward = savgol_filter(reward, 10, 1)

reward_mean = jnp.mean(filt_reward, axis=0)
reward_std = jnp.std(filt_reward, axis=0)

iters = jnp.linspace(1, 301, 301)
plt.plot(iters, reward_mean)
plt.fill_between(
    iters,
    reward_mean - reward_std,
    reward_mean + reward_std,
    alpha=0.25
)
plt.show()

# jnp.save("smc_double_pendulum.npy", _reward)
