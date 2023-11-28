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
    # https://underactuated.mit.edu/multibody.html#section1

    g = 9.81
    l1, l2 = 1.0, 1.0
    m1, m2 = 2.0, 2.0
    k1, k2 = 1e-3, 1e-3

    th1, th2, dth1, dth2 = x
    u1, u2 = u

    s1, c1 = jnp.sin(th1), jnp.cos(th1)
    s2, c2 = jnp.sin(th2), jnp.cos(th2)
    s12 = jnp.sin(th1 + th2)

    # inertia
    M = jnp.array(
        [
            [
                (m1 + m2) * l1**2 + m2 * l2**2 + 2.0 * m2 * l1 * l2 * c2,
                m2 * l2**2 + m2 * l1 * l2 * c2,
            ],
            [
                m2 * l2**2 + m2 * l1 * l2 * c2,
                m2 * l2**2
            ],
        ]
    )

    # Corliolis
    C = jnp.array(
        [
            [
                0.0,
                -m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2
            ],
            [
                0.5 * m2 * l1 * l2 * (2.0 * dth1 + dth2) * s2,
                -0.5 * m2 * l1 * l2 * dth1 * s2,
            ],
        ]
    )

    # gravity
    tau = -g * jnp.array(
        [
            (m1 + m2) * l1 * s1 + m2 * l2 * s12,
            m2 * l2 * s12
        ]
    )

    B = jnp.eye(2)

    u1 = u1 - k1 * dth1
    u2 = u2 - k2 * dth2

    u = jnp.hstack([u1, u2])
    v = jnp.hstack([dth1, dth2])

    inv_M = jnp.linalg.inv(M)
    a = inv_M @ (tau + B @ u - C @ v)
    # a = jnp.linalg.solve(M, tau + B @ u - C @ v)

    return jnp.hstack((v, a))


@partial(jnp.vectorize, signature='(k),()->()')
def reward(state, eta):
    q, p, qd, pd = state[:4]
    u = jnp.atleast_1d(state[4:])

    goal = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

    def wrap_angle(_q: float) -> float:
        return _q % (2.0 * jnp.pi)

    Q = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))
    R = jnp.diag(jnp.array([1e-3, 1e-3]))

    _state = jnp.hstack(
        (wrap_angle(q), wrap_angle(p), qd, pd)
    )
    cost = (_state - goal).T @ Q @ (_state - goal)
    cost += u.T @ R @ u
    return - 0.5 * eta * cost


prior = distrax.MultivariateNormalDiag(
    loc=jnp.zeros((6,)),
    scale_diag=jnp.ones((6,)) * 1e-4
)

dynamics = StochasticDynamics(
    dim=4,
    ode=ode,
    step=0.05,
    log_std=jnp.log(jnp.array([1e-4, 1e-4, 1e-2, 1e-2]))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def polar(x):
    sin_q, cos_q = jnp.sin(x[0]), jnp.cos(x[0])
    sin_p, cos_p = jnp.sin(x[1]), jnp.cos(x[1])
    return jnp.hstack([sin_q, cos_q, sin_p, cos_p, x[2], x[3]])


network = Network(
    dim=2,
    layer_size=[512, 512],
    transform=polar,
    activation=nn.relu,
    init_log_std=jnp.log(1.0 * jnp.ones((2,))),
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 25.0),
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