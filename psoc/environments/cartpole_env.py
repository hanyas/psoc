from typing import Callable
from functools import partial

import jax

from jax import numpy as jnp
from jax import lax as jl

import distrax
from flax import linen as nn

from psoc.abstract import StochasticDynamics
from psoc.abstract import PolicyNetwork
from psoc.abstract import StochasticPolicy
from psoc.abstract import ClosedLoop

from psoc.utils import Tanh

jax.config.update("jax_enable_x64", True)


@partial(jnp.vectorize, signature='(k),(h)->(k)')
def ode(x, u):
    # https://underactuated.mit.edu/acrobot.html#cart_pole

    g = 9.81  # gravity
    l = 0.5  # pole length
    mc = 10.0  # cart mass
    mp = 1.0  # pole mass

    x, q, xd, qd = x

    sth = jnp.sin(q)
    cth = jnp.cos(q)

    xdd = (
        u + mp * sth * (l * qd**2 + g * cth)
    ) / (mc + mp * sth**2)

    qdd = (
        - u * cth
        - mp * l * qd**2 * cth * sth
        - (mc + mp) * g * sth
    ) / (l * mc + l * mp * sth**2)

    return jnp.hstack((xd, qd, xdd, qdd))


@partial(jnp.vectorize, signature='(k),()->()')
def cost(state, eta):
    x, q, xd, qd = state[:4]
    u = jnp.atleast_1d(state[4:])

    goal = jnp.array([0.0, jnp.pi, 0.0, 0.0])

    def wrap_angle(_q: float) -> float:
        return _q % (2.0 * jnp.pi)

    Q = jnp.diag(jnp.array([1e0, 1e1, 1e-1, 1e-1]))
    R = jnp.diag(jnp.array([1e-3]))

    _state = jnp.hstack((x, wrap_angle(q), xd, qd))
    cost = (_state - goal).T @ Q @ (_state - goal)
    cost += u.T @ R @ u
    return - 0.5 * eta * cost


prior = distrax.MultivariateNormalDiag(
    loc=jnp.zeros((5,)),
    scale_diag=jnp.ones((5,)) * 1e-4
)

dynamics = StochasticDynamics(
    dim=4,
    ode=ode,
    step=0.05,
    log_std=jnp.log(1e-2 * jnp.ones((4,)))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def polar(x):
    cos_q, sin_q = jnp.sin(x[1]), jnp.cos(x[1])
    return jnp.hstack([x[0], cos_q, sin_q, x[2], x[3]])


network = PolicyNetwork(
    dim=1,
    layer_size=[256, 256],
    transform=polar,
    activation=nn.relu,
    init_log_std=jnp.log(2.5 * jnp.ones((1,))),
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 50.0),
    Tanh()
])


def create_env(params, eta):
    policy = StochasticPolicy(
        network, bijector, params
    )

    closedloop = ClosedLoop(
        dynamics, policy
    )

    anon_cost = lambda z: cost(z, eta)
    return prior, closedloop, anon_cost


def simulate(
    prior: distrax.Distribution,
    transition_model: ClosedLoop,
    length: int,
):
    def body(carry, args):
        prev_state = carry
        next_state = transition_model.mean(prev_state)
        return next_state, next_state

    init_state = prior.mean()
    _, states = \
        jl.scan(body, init_state, (), length=length - 1)

    states = jnp.insert(states, 0, init_state, 0)
    return states


@partial(jax.vmap, in_axes=(0, 0, None, None))
def log_complete_likelihood(
    state: jnp.ndarray,
    next_state: jnp.ndarray,
    transition_model: ClosedLoop,
    log_observation: Callable,
):
    ll = transition_model.logpdf(state, next_state) \
         + log_observation(next_state)
    return ll
