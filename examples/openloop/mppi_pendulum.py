from typing import Dict

import jax
from jax import random as jr
from jax import numpy as jnp

import distrax

from psoc.environments.openloop import pendulum_env as pendulum

from psoc.abstract import StochasticDynamics
from psoc.abstract import Gaussian
from psoc.abstract import OpenloopPolicyWithClipping
from psoc.abstract import OpenLoop

from psoc.algorithms import mppi

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
proposal = Gaussian(
    dim=1,
)

# policy bijector
bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 5.0),
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

    policy = OpenloopPolicyWithClipping(
        proposal, bijector, parameters
    )

    loop_obj = OpenLoop(dynamics, policy)

    reward_fn = lambda z: pendulum.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


key = jr.PRNGKey(11)

nb_steps = 101
horizon = 20

nb_particles = 32

init_state = jnp.zeros((3,))
tempering = 0.5

key, sub_key = jr.split(key, 2)
parameters = proposal.init(
    sub_key, jnp.zeros((1,))
)["params"]

state = init_state

trajectory = jnp.zeros((nb_steps, 3))
trajectory = trajectory.at[0, :].set(state)

for t in range(nb_steps):
    prior_dist, loop_obj, reward_fn = \
        make_env(state, parameters, tempering)

    key, sub_key = jr.split(key, 2)
    samples, weights = \
        mppi(
            sub_key,
            horizon,
            nb_particles,
            prior_dist,
            loop_obj,
            reward_fn
        )

    key, sub_key = jr.split(key, 2)
    idx = jr.choice(sub_key, a=len(weights), p=weights)
    sample = samples[:, idx, :]

    x = sample[0, :2]
    u = sample[1, -1:]

    key, sub_key = jr.split(key, 2)
    xn = dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

plt.plot(trajectory)
plt.show()
