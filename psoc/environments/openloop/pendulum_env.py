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
def ode(x, u):
    l, m = 1.0, 1.0
    g, d = 9.81, 1e-3

    q, dq = x
    ddq = - g / l * jnp.sin(q) \
          + (u - d * dq) / (m * l ** 2)
    return jnp.hstack((dq, ddq))


@partial(jnp.vectorize, signature='(k),()->()')
def reward(state, eta):
    x, x_dot = state[:2]
    u = jnp.atleast_1d(state[2:])

    g0 = jnp.array([jnp.pi, 0.0])

    def wrap_angle(_q: float) -> float:
        return _q % (2.0 * jnp.pi)

    Q = jnp.diag(jnp.array([1e1, 1e-1]))
    R = jnp.diag(jnp.array([1e-1]))

    xw = jnp.hstack((wrap_angle(x), x_dot))
    cost = (xw - g0).T @ Q @ (xw - g0)
    cost += u.T @ R @ u
    return - 0.5 * eta * cost


dynamics = StochasticDynamics(
    dim=2,
    ode=ode,
    step=0.05,
    log_std=jnp.log(1e-2 * jnp.ones((2,)))
)

module = Gaussian(
    dim=1,
    init_params=jnp.array([0.0, 1.0]),
)

# module = GaussMarkov(
#     dim=1,
#     step=0.05,
#     init_params=jnp.array([20.0, 50.0]),
# )

# bijector = distrax.Chain([
#     distrax.ScalarAffine(0.0, 5.0),
#     Tanh(),
# ])

bijector = distrax.Chain([
    distrax.ScalarAffine(-5.0, 10.0),
    Sigmoid(), distrax.ScalarAffine(0.0, 1.5),
])


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
