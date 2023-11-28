from functools import partial

import jax
from jax import numpy as jnp

import distrax
from flax import linen as nn

from psoc.abstract import StochasticDynamics
from psoc.abstract import Network
from psoc.abstract import FeedbackPolicy
from psoc.abstract import FeedbackLoop

from psoc.utils import Tanh

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
    R = jnp.diag(jnp.array([1e-3]))

    xw = jnp.hstack((wrap_angle(x), x_dot))
    cost = (xw - g0).T @ Q @ (xw - g0)
    cost += u.T @ R @ u
    return - 0.5 * eta * cost


prior = distrax.MultivariateNormalDiag(
    loc=jnp.zeros((3,)),
    scale_diag=jnp.ones((3,)) * 1e-4
)

dynamics = StochasticDynamics(
    dim=2,
    ode=ode,
    step=0.05,
    log_std=jnp.log(jnp.array([1e-4, 1e-2]))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def polar(x):
    sin_q, cos_q = jnp.sin(x[0]), jnp.cos(x[0])
    return jnp.hstack([sin_q, cos_q, x[1]])


network = Network(
    dim=1,
    layer_size=[256, 256],
    transform=polar,
    activation=nn.relu,
    init_log_std=jnp.log(2.5 * jnp.ones((1,))),
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 5.0),
    Tanh()
])


def create_env(params, eta):
    policy = FeedbackPolicy(
        network, bijector, params
    )

    closedloop = FeedbackLoop(
        dynamics, policy
    )

    anon_rwrd = lambda z: reward(z, eta)
    return prior, closedloop, anon_rwrd