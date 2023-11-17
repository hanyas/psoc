from typing import Callable
from functools import partial

import jax

from jax import numpy as jnp
from jax import lax as jl

import distrax

from psoc.abstract import StochasticDynamics
from psoc.abstract import PolicyNetwork
from psoc.abstract import StochasticPolicy
from psoc.abstract import ClosedLoop

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
def cost(state, eta):
    x, x_dot = state[:2]
    u = jnp.atleast_1d(state[2:])

    g0 = jnp.array([jnp.pi, 0.0])

    def wrap_angle(x: float) -> float:
        return x % (2.0 * jnp.pi)

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
    log_std=jnp.log(1e-2 * jnp.ones((2,)))
)

network = PolicyNetwork(
    dim=1,
    layer_size=[256, 256, 1],
    init_log_std=jnp.log(1.65 * jnp.ones((1,))),
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0., 5.),
    Tanh()
])

def create_env(params, eta):
    policy = StochasticPolicy(
        network, params, bijector
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


def log_complete_likelihood(
    state: jnp.ndarray,
    next_state: jnp.ndarray,
    transition_model: ClosedLoop,
    log_observation: Callable,
):
    ll = transition_model.logpdf(state, next_state) \
         + log_observation(next_state)
    return ll
