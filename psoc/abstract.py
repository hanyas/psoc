from typing import Dict, Callable, NamedTuple, Sequence
from functools import partial

import jax
from jax import random as jr
from jax import numpy as jnp

import distrax
from flax import linen as nn


class PolicyNetwork(nn.Module):
    dim: int
    layer_size: Sequence[int]
    init_log_std: jnp.ndarray

    @staticmethod
    @partial(jnp.vectorize, signature='(k)->(h)')
    def polar(x):
        cos_q, sin_q = jnp.sin(x[0]), jnp.cos(x[0])
        return jnp.hstack([cos_q, sin_q, x[1]])

        # cos_q, sin_q = jnp.sin(x[1]), jnp.cos(x[1])
        # return jnp.hstack([x[0], cos_q, sin_q, x[2], x[3]])

        # cos_q, sin_q = jnp.sin(x[0]), jnp.cos(x[0])
        # cos_p, sin_p = jnp.sin(x[1]), jnp.cos(x[1])
        # return jnp.hstack([cos_q, sin_q, cos_p, sin_p, x[2], x[3]])

    @nn.compact
    def __call__(self, x):
        y = self.polar(x)
        y = nn.relu(nn.Dense(self.layer_size[0])(y))
        y = nn.relu(nn.Dense(self.layer_size[1])(y))
        u = nn.Dense(self.layer_size[2])(y)

        # u = nn.Dense(self.dim)(x)

        log_std = \
            self.param('log_std', lambda rng, shape: self.init_log_std, 1)

        return u


class StochasticPolicy(NamedTuple):
    network: PolicyNetwork
    bijector: distrax.Chain
    params: Dict

    @property
    def dim(self):
        return self.network.dim

    def mean(self, x):
        u = self.network.apply({'params': self.params}, x)
        return self.bijector.forward(u)

    def distribution(self, x):
        u = self.network.apply({'params': self.params}, x)
        log_std = self.params['log_std']

        raw_dist = distrax.MultivariateNormalDiag(
            loc=u, scale_diag=jnp.exp(log_std)
        )
        squashed_dist = distrax.Transformed(
            distribution=raw_dist,
            bijector=distrax.Block(self.bijector, ndims=self.dim)
        )
        return squashed_dist

    def sample(self, key, x):
        dist = self.distribution(x)
        return dist.sample(seed=key)

    def logpdf(self, x, u):
        dist = self.distribution(x)
        return dist.log_prob(u)


class StochasticDynamics(NamedTuple):
    dim: int
    ode: Callable
    step: float
    log_std: jnp.ndarray

    def mean(self, x, u):
        dx = self.ode(x, u)
        return x + self.step * dx

    def sample(self, key, x, u):
        dist = distrax.MultivariateNormalDiag(
            loc=self.mean(x, u),
            scale_diag=jnp.exp(self.log_std)
        )
        return dist.sample(seed=key)

    def logpdf(self, x, u, xn):
        dist = distrax.MultivariateNormalDiag(
            loc=self.mean(x, u),
            scale_diag=jnp.exp(self.log_std)
        )
        return dist.log_prob(xn)


class ClosedLoop(NamedTuple):
    dynamics: StochasticDynamics
    policy: StochasticPolicy

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

    def logpdf(self, z, zn):
        x = jnp.atleast_1d(z[:self.xdim])
        u = jnp.atleast_1d(z[-self.udim:])

        xn = jnp.atleast_1d(zn[:self.xdim])
        un = jnp.atleast_1d(zn[-self.udim:])

        ll = self.dynamics.logpdf(x, u, xn)
        ll += self.policy.logpdf(xn, un)

        NINF = jnp.finfo(jnp.float64).min
        return jnp.nan_to_num(ll, nan=NINF)
