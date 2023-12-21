from typing import Dict

import jax
from jax import random as jr
from jax import numpy as jnp

import distrax
import optax
from flax import linen as nn

from psoc.abstract import StochasticDynamics
from psoc.abstract import Gaussian, GaussMarkov
from psoc.abstract import OpenloopPolicyWithSquashing
from psoc.abstract import OpenloopPolicyWithClipping
from psoc.abstract import OpenLoop
from psoc.bijector import Tanh, Sigmoid

from psoc.common import rollout
from psoc.utils import create_train_state
from psoc.utils import positivity_constraint
from psoc.optimization import rao_blackwell_markovian_score_optimization

from psoc.environments.openloop import double_pendulum_env as double_pendulum

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


# stochastic dynamical system
dynamics = StochasticDynamics(
    dim=4,
    ode=double_pendulum.ode,
    step=0.05,
    stddev=1e-4 * jnp.ones((4,))
)

# stoachstic policy proposal
gauss_markov = GaussMarkov(
    dim=2,
    step=0.05,
    inv_length_init=nn.initializers.constant(100.0),
    diffusion_init=nn.initializers.constant(200.0)
)

# policy bijector
bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 25.0),
    # Tanh()
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

    policy = OpenloopPolicyWithClipping(
        gauss_markov, bijector, parameters, positivity_constraint
    )

    loop_obj = OpenLoop(
        dynamics, policy
    )

    reward_fn = lambda z: double_pendulum.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


key = jr.PRNGKey(6)

nb_steps = 75
horizon = 35

nb_particles = 64
nb_samples = 64

init_state = jnp.zeros((6,))
tempering = 5e-3

nb_iter = 100
learning_rate = 5e-1

key, sub_key = jr.split(key, 2)
opt_state = create_train_state(
    key=sub_key,
    module=gauss_markov,
    init_data=jnp.zeros((2,)),
    learning_rate=learning_rate,
    optimizer=optax.sgd
)

state = init_state

trajectory = jnp.zeros((nb_steps, 6))
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
            make_env
        )
    x = sample[0, :4]
    u = sample[1, -2:]

    key, sub_key = jr.split(key, 2)
    xn = dynamics.sample(sub_key, x, u)

    state = jnp.hstack((xn, u))
    trajectory = trajectory.at[t + 1, :].set(state)

    print(state[:2])

plt.plot(trajectory[:, :-2])
plt.show()
