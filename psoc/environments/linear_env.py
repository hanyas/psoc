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
def cost(state, eta):
    goal = jnp.array([0.0, 0.0, 0.0])
    weights = jnp.array([1e2, 1e0, 1e-1])
    cost = - 0.5 * jnp.dot(state - goal, weights * (state - goal))
    return eta * cost


prior = distrax.MultivariateNormalDiag(
    loc=jnp.array([1.0, 2.0, 0.0]),
    scale_diag=jnp.ones((3,)) * 1e-4
)

dynamics = StochasticDynamics(
    dim=2,
    ode=ode,
    step=0.1,
    log_std=jnp.log(1e-2 * jnp.ones((2,)))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def identity(x):
    return x


network = PolicyNetwork(
    dim=1,
    layer_size=[],
    transform=identity,
    activation=nn.relu,
    init_log_std=jnp.log(1.0 * jnp.ones((1,))),
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 2.5),
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
