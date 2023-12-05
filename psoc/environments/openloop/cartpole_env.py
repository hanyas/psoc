from functools import partial

from jax import numpy as jnp


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
def reward(state, eta):
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
