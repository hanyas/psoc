from typing import Dict, Callable, NamedTuple, Sequence

import jax
from jax import random as jr
from jax import numpy as jnp

import distrax
from flax import linen as nn


class PolicyNetwork(nn.Module):
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
            bijector=distrax.Block(self.bijector, ndims=1)
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
