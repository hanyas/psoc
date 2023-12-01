from typing import Tuple

import jax
from jax import numpy as jnp

import distrax


class Tanh(distrax.Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: jnp.array) -> jnp.array:
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def forward_and_log_det(self, x: jnp.array) -> Tuple[jnp.array, jnp.array]:
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse(self, y):
        x = jnp.where(
            jnp.less_equal(jnp.abs(y), 1.0),
            jnp.clip(y, -0.99999997, 0.99999997),
            y,
        )
        return jnp.arctanh(x)

    def inverse_and_log_det(self, y: jnp.array) -> Tuple[jnp.array, jnp.array]:
        x = self.inverse(y)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        return type(other) is Tanh  # pylint: disable=unidiomatic-typecheck
