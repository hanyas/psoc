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
from psoc.sampling import smc_sampling
from psoc.utils import create_train_state
from psoc.optimization import rao_blackwell_score_optimization

from psoc.environments.feedback import pendulum_env as pendulum

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


dynamics = StochasticDynamics(
    dim=2,
    ode=pendulum.ode,
    step=0.05,
    stddev=1e-2 * jnp.ones((2,))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def polar(x):
    sin_q, cos_q = jnp.sin(x[0]), jnp.cos(x[0])
    return jnp.hstack([sin_q, cos_q, x[1]])


proposal = Network(
    dim=1,
    layer_size=[256, 256],
    transform=polar,
    activation=nn.relu,
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 5.0),
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
        proposal, bijector, parameters
    )

    loop_obj = FeedbackLoop(
        dynamics, policy
    )

    reward_fn = lambda z: pendulum.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


key = jr.PRNGKey(3)

nb_steps = 101
nb_particles = 64
nb_samples = 32

init_state = jnp.zeros((3,))
tempering = 0.5

nb_iter = 250
learning_rate = 1e-3

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=proposal,
    init_data=jnp.zeros((2,)),
    learning_rate=learning_rate
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
    make_env
)[0]

key, sub_key = jr.split(key, 2)
opt_state = rao_blackwell_score_optimization(
    sub_key,
    nb_iter,
    nb_steps,
    nb_particles,
    nb_samples,
    reference,
    init_state,
    opt_state,
    tempering,
    make_env
)[0]

key, sub_key = jr.split(key, 2)
sample = rollout(
    sub_key,
    nb_steps,
    init_state,
    opt_state.params,
    tempering,
    make_env,
)

plt.plot(sample)
plt.show()
