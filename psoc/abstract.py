from typing import Dict, Callable, NamedTuple, Sequence, Union

from jax import random as jr
from jax import numpy as jnp

import distrax
from flax import linen as nn

from psoc.utils import constrain


class Network(nn.Module):
    dim: int
    layer_size: Sequence[int]
    transform: Callable
    activation: Callable
    init_log_std: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        y = self.transform(x)
        for _layer_size in self.layer_size:
            y = self.activation(nn.Dense(_layer_size)(y))
        u = nn.Dense(self.dim)(y)

        log_std = \
            self.param('log_std', lambda rng, shape: self.init_log_std, 1)

        return u


class FeedbackPolicy(NamedTuple):
    module: Network
    bijector: distrax.Chain
    params: Dict

    @property
    def dim(self):
        return self.module.dim

    def mean(self, x):
        u = self.module.apply({'params': self.params}, x)
        return self.bijector.forward(u)

    def distribution(self, x):
        u = self.module.apply({'params': self.params}, x)
        log_std = self.params['log_std']

        raw_dist = distrax.MultivariateNormalDiag(
            loc=u, scale_diag=jnp.exp(log_std)
        )
        squashed_dist = distrax.Transformed(
            distribution=raw_dist,
            bijector=distrax.Block(self.bijector, ndims=1)
        )
        return squashed_dist

    def sample(self, key, x):
        dist = self.distribution(x)
        return dist.sample(seed=key)

    def logpdf(self, x, u):
        dist = self.distribution(x)
        return dist.log_prob(u)


class Gaussian(nn.Module):
    dim: int
    init_params: jnp.ndarray

    @nn.compact
    def __call__(self, u):
        loc = self.param('loc', lambda rng, shape: self.init_params[0], 1)
        scale = self.param('scale', lambda rng, shape: self.init_params[1], 1)
        return loc * jnp.ones_like(u), scale * jnp.ones_like(u)


class GaussMarkov(nn.Module):
    dim: int
    step: int
    init_params: jnp.ndarray

    @nn.compact
    def __call__(self, u):
        l = self.param('l', lambda rng, shape: self.init_params[0], 1)
        q = self.param('q', lambda rng, shape: self.init_params[1], 1)

        sigma_sqr = q / (2.0 * l) * (1.0 - jnp.exp(-2.0 * l * self.step))

        loc = jnp.exp(- l * self.step) * u
        scale = jnp.sqrt(sigma_sqr)
        return loc, scale


class OpenloopPolicy(NamedTuple):
    module: Union[Gaussian, GaussMarkov]
    bijector: distrax.Chain
    params: Dict

    @property
    def dim(self):
        return self.module.dim

    def mean(self, u):
        _params = constrain(self.params)
        un, _ = self.module.apply({'params': _params}, u)
        return self.bijector.forward(un)

    def distribution(self, u):
        _params = constrain(self.params)
        loc, scale = self.module.apply({'params': _params}, u)

        raw_dist = distrax.MultivariateNormalDiag(
            loc=loc, scale_diag=jnp.atleast_1d(scale)
        )

        squashed_dist = distrax.Transformed(
            distribution=raw_dist,
            bijector=distrax.Block(self.bijector, ndims=1)
        )
        return squashed_dist

    def sample(self, key, u):
        dist = self.distribution(u)
        return dist.sample(seed=key)

    def logpdf(self, u, un):
        dist = self.distribution(u)
        return dist.log_prob(un)


class StochasticDynamics(NamedTuple):
    dim: int
    ode: Callable
    step: float
    log_std: jnp.ndarray

    @staticmethod
    def runge_kutta(
        x: jnp.ndarray,
        u: jnp.ndarray,
        ode: Callable,
        step: float,
    ):
        k1 = ode(x, u)
        k2 = ode(x + 0.5 * step * k1, u)
        k3 = ode(x + 0.5 * step * k2, u)
        k4 = ode(x + step * k3, u)
        return x + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def euler(x, u, ode, step):
        dx = ode(x, u)
        return x + step * dx

    def mean(self, x, u):
        return self.euler(x, u, self.ode, self.step)
        # return self.runge_kutta(x, u, self.ode, self.step)

    def distribution(self, x, u):
        return distrax.MultivariateNormalDiag(
            loc=self.mean(x, u),
            scale_diag=jnp.exp(self.log_std)
        )

    def sample(self, key, x, u):
        dist = self.distribution(x, u)
        return dist.sample(seed=key)

    def logpdf(self, x, u, xn):
        dist = self.distribution(x, u)
        return dist.log_prob(xn)


class FeedbackLoop(NamedTuple):
    dynamics: StochasticDynamics
    policy: FeedbackPolicy

    @property
    def xdim(self):
        return self.dynamics.dim

    @property
    def udim(self):
        return self.policy.dim

    def mean(self, z):
        x = jnp.atleast_1d(z[:self.xdim])
        u = jnp.atleast_1d(z[-self.udim:])
        xn = self.dynamics.mean(x, u)
        un = self.policy.mean(xn)
        return jnp.hstack((xn, un))

    def sample(self, key, z):
        u_key, x_key = jr.split(key, 2)

        x = jnp.atleast_1d(z[..., :self.xdim])
        u = jnp.atleast_1d(z[..., -self.udim:])

        xn = self.dynamics.sample(x_key, x, u)
        un = self.policy.sample(u_key, xn)
        return jnp.column_stack((xn, un))

    def forward(self, key, z):
        x = jnp.atleast_1d(z[:self.xdim])
        u = jnp.atleast_1d(z[-self.udim:])

        xn = self.dynamics.sample(key, x, u)
        un = self.policy.mean(xn)
        return jnp.hstack((xn, un))

    def logpdf(self, z, zn):
        x = jnp.atleast_1d(z[:self.xdim])
        u = jnp.atleast_1d(z[-self.udim:])

        xn = jnp.atleast_1d(zn[:self.xdim])
        un = jnp.atleast_1d(zn[-self.udim:])

        ll = self.dynamics.logpdf(x, u, xn)
        ll += self.policy.logpdf(xn, un)
        return ll


class OpenLoop(NamedTuple):
    dynamics: StochasticDynamics
    policy: OpenloopPolicy

    @property
    def xdim(self):
        return self.dynamics.dim

    @property
    def udim(self):
        return self.policy.dim

    def mean(self, z):
        x = jnp.atleast_1d(z[:self.xdim])
        u = jnp.atleast_1d(z[-self.udim:])
        xn = self.dynamics.mean(x, u)
        un = self.policy.mean(u)
        return jnp.hstack((xn, un))

    def sample(self, key, z):
        u_key, x_key = jr.split(key, 2)

        x = jnp.atleast_1d(z[..., :self.xdim])
        u = jnp.atleast_1d(z[..., -self.udim:])

        xn = self.dynamics.sample(x_key, x, u)
        un = self.policy.sample(u_key, u)
        return jnp.column_stack((xn, un))

    def forward(self, key, z):
        x = jnp.atleast_1d(z[:self.xdim])
        u = jnp.atleast_1d(z[-self.udim:])

        xn = self.dynamics.sample(key, x, u)
        un = self.policy.mean(u)
        return jnp.hstack((xn, un))

    def logpdf(self, z, zn):
        x = jnp.atleast_1d(z[:self.xdim])
        u = jnp.atleast_1d(z[-self.udim:])

        xn = jnp.atleast_1d(zn[:self.xdim])
        un = jnp.atleast_1d(zn[-self.udim:])

        ll = self.dynamics.logpdf(x, u, xn)
        ll += self.policy.logpdf(u, un)
        return ll
