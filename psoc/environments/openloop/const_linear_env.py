from typing import Dict
from functools import partial

import jax
from jax import numpy as jnp

import distrax

from psoc.abstract import StochasticDynamics
from psoc.abstract import Gaussian, GaussMarkov
from psoc.abstract import OpenloopPolicy
from psoc.abstract import OpenLoop

from psoc.bijector import Tanh, Sigmoid

jax.config.update("jax_enable_x64", True)


@partial(jnp.vectorize, signature='(k),(h)->(k)')
def ode(
    x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:

    A = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 0.0]
        ]
    )
    B = jnp.array(
        [
            [0.0],
            [1.0]
        ]
    )
    return A @ x + B @ u


@partial(jnp.vectorize, signature='(k),()->()')
def reward(state, eta):
    goal = jnp.array([0.0, 0.0, 0.0])
    weights = jnp.array([1e2, 1e0, 1e0])
    cost = jnp.dot(state - goal, weights * (state - goal))
    return - 0.5 * eta * cost


dynamics = StochasticDynamics(
    dim=2,
    ode=ode,
    step=0.1,
    log_std=jnp.log(1e-2 * jnp.ones((2,)))
)

module = Gaussian(
    dim=1,
    init_params={
        'loc': jnp.zeros((1,)),
        'scale': jnp.ones((1,))
    }
)

# module = GaussMarkov(
#     dim=1,
#     step=0.1,
#     init_params={
#         'l': 25.0 * jnp.ones((1,)),
#         'q': 100.0 * jnp.ones((1,))
#     }
# )

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 2.5),
    Tanh(),
])

# bijector = distrax.Chain([
#     distrax.ScalarAffine(-2.5, 5.0),
#     Sigmoid(), distrax.ScalarAffine(0.0, 1.5),
# ])


def create_env(
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
):
    prior = distrax.MultivariateNormalDiag(
        loc=init_state,
        scale_diag=jnp.ones((3,)) * 1e-4
    )

    policy = OpenloopPolicy(
        module, bijector, parameters
    )

    loop = OpenLoop(
        dynamics, policy
    )

    reward_fn = lambda z: reward(z, tempering)
    return prior, loop, reward_fn
