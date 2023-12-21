from typing import Dict

import jax
from jax import random as jr
from jax import numpy as jnp

import distrax
import optax
from flax import linen as nn

from psoc.abstract import StochasticDynamics
from psoc.abstract import GaussMarkov
from psoc.abstract import OpenloopPolicyWithSquashing
from psoc.abstract import OpenLoop
from psoc.bijector import Tanh

from psoc.common import rollout
from psoc.utils import create_train_state
from psoc.utils import positivity_constraint

from psoc.optimization import rao_blackwell_markovian_score_optimization
from psoc.environments.openloop import pendulum_env as pendulum

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

# stochastic dynamical system
dynamics = StochasticDynamics(
    dim=2,
    ode=pendulum.ode,
    step=0.05,
    stddev=1e-2 * jnp.ones((2,))
)

# stoachstic policy proposal
gauss_markov = GaussMarkov(
    dim=1,
    step=0.05,
    inv_length_init=nn.initializers.constant(25.0),
    diffusion_init=nn.initializers.constant(50.0)
)

# policy bijector
bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 5.0),
    Tanh(),
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

    policy = OpenloopPolicyWithSquashing(
        gauss_markov, bijector, parameters, positivity_constraint
    )

    loop_obj = OpenLoop(dynamics, policy)

    reward_fn = lambda z: pendulum.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


key = jr.PRNGKey(135)

nb_steps = 101
horizon = 20

nb_particles = 32
nb_samples = 16

init_state = jnp.zeros((3,))
tempering = 0.75

nb_iter = 100
learning_rate = 1e-1

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=gauss_markov,
    init_data=jnp.zeros((1,)),
    learning_rate=learning_rate,
    optimizer=optax.sgd
)

state = init_state

trajectory = jnp.zeros((nb_steps, 3))
trajectory = trajectory.at[0, :].set(state)

for t in range(nb_steps):
    key, sub_key = jr.split(key, 2)
    reference = rollout(
        sub_key,
        horizon,
        state,
        opt_state.params,
        tempering,
        make_env,
    )

    key, sub_key = jr.split(key, 2)
    opt_state, sample, _ = \
        rao_blackwell_markovian_score_optimization(
            sub_key,
            nb_iter,
            horizon,
            nb_particles,
            nb_samples,
            reference,
            state,
            opt_state,
            tempering,
            make_env,
        )
    x = sample[0, :2]
    u = sample[1, -1:]

    key, sub_key = jr.split(key, 2)
    xn = dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

plt.plot(trajectory)
plt.show()
