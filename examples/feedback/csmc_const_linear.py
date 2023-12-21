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

from psoc.common import rollout
from psoc.utils import create_train_state
from psoc.sampling import smc_sampling
from psoc.optimization import batched_markovian_score_optimization

from psoc.environments.feedback import const_linear_env as linear

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


dynamics = StochasticDynamics(
    dim=2,
    ode=linear.ode,
    step=0.1,
    stddev=1e-2 * jnp.ones((2,))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def identity(x):
    return x


network = Network(
    dim=1,
    layer_size=[64, 64],
    transform=identity,
    activation=nn.relu,
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 2.5),
    Tanh()
])


def make_env(
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
):
    prior_dist = distrax.MultivariateNormalDiag(
        loc=init_state,
        scale_diag=jnp.ones((3,)) * 1e-4
    )

    policy = FeedbackPolicyWithSquashing(
        network, bijector, parameters
    )

    loop_obj = FeedbackLoop(
        dynamics, policy
    )

    reward_fn = lambda z: linear.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


key = jr.PRNGKey(1)

nb_steps = 51
nb_particles = 16
nb_samples = 5

init_state = jnp.array([1.0, 2.0, 0.0])
tempering = 0.1

nb_iter = 100
learning_rate = 1e-2
batch_size = 64

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=network,
    init_data=jnp.zeros((2,)),
    learning_rate=learning_rate
)

key, sub_key = jr.split(key, 2)
reference = smc_sampling(
    sub_key,
    nb_steps,
    int(10 * nb_particles),
    int(10 * nb_particles),
    init_state,
    opt_state.params,
    tempering,
    make_env
)[0]

key, sub_key = jr.split(key, 2)
opt_state, _ = batched_markovian_score_optimization(
    sub_key,
    nb_iter,
    nb_steps,
    nb_particles,
    nb_samples,
    reference,
    init_state,
    opt_state,
    tempering,
    batch_size,
    make_env,
    True
)

key, sub_key = jr.split(key, 2)
sample, _ = rollout(
    sub_key,
    nb_steps,
    1,
    init_state,
    opt_state.params,
    tempering,
    make_env
)

plt.plot(sample[0, ...])
plt.show()
