from functools import partial

from jax import numpy as jnp


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
