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

    @nn.compact
    def __call__(self, x):
        y = self.polar(x)
        y = nn.relu(nn.Dense(self.layer_size[0])(y))
        y = nn.relu(nn.Dense(self.layer_size[1])(y))
        u = nn.Dense(self.layer_size[2])(y)

        log_std = \
            self.param('log_std', lambda rng, shape: self.init_log_std, 1)

        return u


class StochasticPolicy(NamedTuple):
    network: PolicyNetwork
    params: Dict
    bijector: distrax.Chain

    @property
    def dim(self):
        return self.network.dim

    def mean(self, x):
        u = self.network.apply({'params': self.params}, x)
        return self.bijector.forward(u)

    def squash_dist(self, x):
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

    def explicit_sample(self, x, r):
        u = self.network.apply({'params': self.params}, x)
        return self.bijector.forward(u + jnp.exp(self.params['log_std']) * r)

    def sample(self, key, x):
        dist = self.squash_dist(x)
        return dist.sample(seed=key)

    def logpdf(self, x, u):
        dist = self.squash_dist(x)
        return dist.log_prob(u)


class StochasticDynamics(NamedTuple):
    dim: int
    ode: Callable
    step: float
    log_std: jnp.ndarray

    def mean(self, x, u):
        dx = self.ode(x, u)
        return x + self.step * dx

    def explicit_sample(self, x, u, q):
        return self.mean(x, u) + jnp.exp(self.log_std) * q

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
        x = jnp.atleast_1d(z[..., :self.xdim])
        u = self.policy.mean(x)
        xn = self.dynamics.mean(x, u)
        return jnp.hstack((xn, u))

    def sample(self, key, z):
        x = jnp.atleast_1d(z[..., :self.xdim])
        u_key, x_key = jr.split(key, 2)
        u = self.policy.sample(u_key, x)
        xn = self.dynamics.sample(x_key, x, u)
        return jnp.column_stack((xn, u))

    def explicit_sample(self, z, v):
        q = jnp.atleast_1d(v[..., :self.xdim])
        r = jnp.atleast_1d(v[..., -self.udim:])
        x = jnp.atleast_1d(z[..., :self.xdim])
        u = self.policy.explicit_sample(x, r)
        xn = self.dynamics.explicit_sample(x, u, q)
        return jnp.column_stack((xn, u))

    def logpdf(self, z, zn):
        x = jnp.atleast_1d(z[..., :self.xdim])
        u = jnp.atleast_1d(zn[..., -self.udim:])
        xn = jnp.atleast_1d(zn[..., :self.xdim])

        ll = self.dynamics.logpdf(x, u, xn)
        ll += self.policy.logpdf(x, u)
        return ll
