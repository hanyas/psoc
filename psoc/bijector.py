import math
from typing import Optional, Union

import jax
from jax import Array, numpy as jnp

from psoc.core import PRNGKey

from distrax._src.utils import math
from distrax import (
    Bijector,
    Chain,
    Block,
    Inverse,
    ScalarAffine,
    MaskedCoupling,
    Transformed
)


def _stable_sigmoid(x: Array) -> Array:
    return jnp.where(x < -9.0, jnp.exp(x), jax.nn.sigmoid(x))


def _stable_softplus(x: Array) -> Array:
    return jnp.where(x < -9.0, jnp.log1p(jnp.exp(x)), jax.nn.softplus(x))


class Tanh(Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse(self, y):
        x = jnp.clip(y, -0.99995, 0.99995)
        return jnp.arctanh(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = self.inverse(y)
        return x, -self.forward_log_det_jacobian(x)


class Sigmoid(Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return -_stable_softplus(-x) - _stable_softplus(x)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return _stable_sigmoid(x), self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = jnp.clip(y, -0.99995, 0.99995)
        z = jnp.log(x) - jnp.log1p(-x)
        return z, -self.forward_log_det_jacobian(z)


class BlockConditional(Block):
    def __init__(self, bijector: Bijector, ndims: int):
        super().__init__(bijector=bijector, ndims=ndims)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.forward(x, context)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.inverse(y, context)

    def forward_log_det_jacobian(self, x: Array, context: Optional[Array] = None) -> Array:
        log_det = self._bijector.forward_log_det_jacobian(x, context)
        return math.sum_last(log_det, self._ndims)

    def inverse_log_det_jacobian(self, y: Array, context: Optional[Array] = None) -> Array:
        log_det = self._bijector.inverse_log_det_jacobian(y, context)
        return math.sum_last(log_det, self._ndims)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        y, log_det = self._bijector.forward_and_log_det(x, context)
        return y, math.sum_last(log_det, self._ndims)

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        x, log_det = self._bijector.inverse_and_log_det(y, context)
        return x, math.sum_last(log_det, self._ndims)


class TanhConditional(Tanh):
    def __init__(self):
        super().__init__()

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return super().forward(x)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return super().inverse(y)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        return super().forward_and_log_det(x)

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        return super().inverse_and_log_det(y)


class ScalarAffineConditional(ScalarAffine):
    def __init__(self, shift: Union[float, Array], scale: Union[float, Array]):
        super().__init__(shift=shift, scale=scale)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return super().forward(x)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return super().inverse(y)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        return super().forward_and_log_det(x)

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        return super().inverse_and_log_det(y)


class InverseConditional(Inverse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.inverse(x, context)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.forward(y, context)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        return self._bijector.inverse_and_log_det(x, context)

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        return self._bijector.forward_and_log_det(y, context)


class ChainConditional(Chain):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x, context)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        for bijector in self._bijectors:
            y = bijector.inverse(y, context)
        return y

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        x, log_det = self._bijectors[-1].forward_and_log_det(x, context)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, context)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        y, log_det = self._bijectors[0].inverse_and_log_det(y, context)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, context)
            log_det += ld
        return y, log_det


class MaskedCouplingConditional(MaskedCoupling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        self._check_forward_input_shape(x)
        masked_x = jnp.where(self._event_mask, x, 0.0)
        params = self._conditioner(masked_x, context)
        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        y = jnp.where(self._event_mask, x, y0)
        logdet = math.sum_last(jnp.where(self._mask, 0.0, log_d), self._event_ndims - self._inner_event_ndims)
        return y, logdet

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> tuple[Array, Array]:
        self._check_inverse_input_shape(y)
        masked_y = jnp.where(self._event_mask, y, 0.0)
        params = self._conditioner(masked_y, context)
        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = jnp.where(self._event_mask, y, x0)
        logdet = math.sum_last(jnp.where(self._mask, 0.0, log_d), self._event_ndims - self._inner_event_ndims)
        return x, logdet


class TransformedConditional(Transformed):
    def __init__(self, distribution, flow):
        super().__init__(distribution, flow)

    def sample(self, seed: PRNGKey, context: Optional[Array] = None, sample_shape: list[int] = ()) -> Array:
        x = self.distribution.sample(seed=seed, sample_shape=sample_shape)
        y, _ = self.bijector.forward_and_log_det(x, context)
        return y

    def log_prob(self, x: Array, context: Optional[Array] = None) -> Array:
        x, ildj_y = self.bijector.inverse_and_log_det(x, context)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample_and_log_prob(self, seed: PRNGKey, context: Optional[Array] = None, sample_shape: list[int] = ()) -> tuple[Array, Array]:
        x, lp_x = self.distribution.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        y, fldj = self.bijector.forward_and_log_det(x, context)
        lp_y = lp_x - fldj
        return y, lp_y
