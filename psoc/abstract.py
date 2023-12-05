from typing import Dict, Callable, NamedTuple, Sequence, Union

from jax import random as jr
from jax import numpy as jnp

import distrax
from flax import linen as nn

from psoc.utils import identity_constraint


class Network(nn.Module):
    dim: int
    layer_size: Sequence[int]
    transform: Callable
    activation: Callable
    log_std_init: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x):
        y = self.transform(x)
        for _layer_size in self.layer_size:
            y = self.activation(nn.Dense(_layer_size)(y))
        u = nn.Dense(self.dim)(y)

        log_std = \
            self.param('log_std', self.log_std_init, 1)

        return u


class FeedbackPolicyWithSquashing(NamedTuple):
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
    loc_init: Callable = nn.initializers.zeros
    scale_init: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, u):
        loc = self.param('loc', self.loc_init, self.dim)
        scale = self.param('scale', self.scale_init, self.dim)
        return jnp.broadcast_to(loc, u.shape), jnp.broadcast_to(scale, u.shape)


class TimeVariantGaussian(nn.Module):
    dim: int
    nb_steps: int
    loc_init: Callable = nn.initializers.zeros
    scale_init: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, u, k):
        loc = self.param('loc', self.loc_init, (self.nb_steps, self.dim))
        scale = self.param('scale', self.scale_init, (self.nb_steps, self.dim))
        return jnp.broadcast_to(loc[k, :], u.shape), jnp.broadcast_to(scale[k, :], u.shape)


class GaussMarkov(nn.Module):
    dim: int
    step: int
    inv_length_init: Callable = nn.initializers.ones
    diffusion_init: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, u):
        inv_length = self.param('inv_length', self.inv_length_init, self.dim)
        diffusion = self.param('diffusion', self.diffusion_init, self.dim)

        sigma_sqr = diffusion / (2.0 * inv_length) \
                    * (1.0 - jnp.exp(-2.0 * inv_length * self.step))

        loc = jnp.exp(- inv_length * self.step) * u
        scale = jnp.sqrt(sigma_sqr)
        return loc, scale


class OpenloopPolicyWithClipping(NamedTuple):
    proposal: Gaussian
    bijector: distrax.Chain
    params: Dict
    constraint: Callable = identity_constraint

    @property
    def dim(self):
        return self.proposal.dim

    def mean(self, u):
        _params = self.constraint(self.params)
        un, _ = self.proposal.apply({'params': _params}, u)
        return self.bijector.forward(un)

    def distribution(self, u):
        _params = self.constraint(self.params)
        loc, scale = self.proposal.apply({'params': _params}, u)

        clipped_dist = distrax.ClippedNormal(
            loc=loc,
            scale=scale,
            minimum=-1.0,
            maximum=1.0
        )

        scaled_dist = distrax.Transformed(
            distribution=distrax.Independent(
                clipped_dist,
                reinterpreted_batch_ndims=1
            ),
            bijector=distrax.Block(self.bijector, ndims=1)
        )
        return scaled_dist

    def sample(self, key, u):
        dist = self.distribution(u)
        return dist.sample(seed=key)

    def logpdf(self, u, un):
        dist = self.distribution(u)
        return dist.log_prob(un)


class OpenloopPolicyWithSquashing(NamedTuple):
    proposal: Union[Gaussian, GaussMarkov]
    bijector: distrax.Chain
    params: Dict
    constraint: Callable = identity_constraint

    @property
    def dim(self):
        return self.proposal.dim

    def mean(self, u):
        _params = self.constraint(self.params)
        un, _ = self.proposal.apply({'params': _params}, u)
        return self.bijector.forward(un)

    def distribution(self, u):
        _params = self.constraint(self.params)
        loc, scale = self.proposal.apply({'params': _params}, u)

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
    stddev: jnp.ndarray

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
            scale_diag=self.stddev
        )

    def sample(self, key, x, u):
        dist = self.distribution(x, u)
        return dist.sample(seed=key)

    def logpdf(self, x, u, xn):
        dist = self.distribution(x, u)
        return dist.log_prob(xn)


class FeedbackLoop(NamedTuple):
    dynamics: StochasticDynamics
    policy: FeedbackPolicyWithSquashing

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
    policy: Union[
        OpenloopPolicyWithSquashing,
        OpenloopPolicyWithClipping
    ]

    @property
    def xdim(self):
        return self.dynamics.dim

    @property
    def udim(self):
        return self.policy.dim

    def mean(self, z):
        x = jnp.atleast_1d(z[:self.xdim])
        up = jnp.atleast_1d(z[-self.udim:])

        u = self.policy.mean(up)
        xn = self.dynamics.mean(x, u)
        return jnp.hstack((xn, u))

    def sample(self, key, z):
        u_key, x_key = jr.split(key, 2)

        x = jnp.atleast_1d(z[..., :self.xdim])
        up = jnp.atleast_1d(z[..., -self.udim:])

        u = self.policy.sample(u_key, up)
        xn = self.dynamics.sample(x_key, x, u)
        return jnp.column_stack((xn, u))

    def forward(self, key, z):
        x = jnp.atleast_1d(z[:self.xdim])
        up = jnp.atleast_1d(z[-self.udim:])

        u = self.policy.mean(up)
        xn = self.dynamics.sample(key, x, u)
        return jnp.hstack((xn, u))

    def logpdf(self, z, zn):
        x = jnp.atleast_1d(z[:self.xdim])
        up = jnp.atleast_1d(z[-self.udim:])

        xn = jnp.atleast_1d(zn[:self.xdim])
        u = jnp.atleast_1d(zn[-self.udim:])

        ll = self.policy.logpdf(up, u)
        ll += self.dynamics.logpdf(x, u, xn)
        return ll
