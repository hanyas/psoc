from typing import Dict

import jax
from jax import numpy as jnp
from jax import random as jr

import distrax

from psoc.abstract import StochasticDynamics
from psoc.abstract import Gaussian
from psoc.abstract import OpenloopPolicyWithClipping
from psoc.abstract import OpenLoop

from psoc.experiments.openloop.common import mppi_experiment
from psoc.environments.openloop import pendulum_env as pendulum

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


dynamics = StochasticDynamics(
    dim=2,
    ode=pendulum.ode,
    step=0.05,
    stddev=1e-2 * jnp.ones((2,))
)

proposal = Gaussian(
    dim=1,
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 5.0),
])


def make_env(
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
):
    prior_dist = distrax.MultivariateNormalDiag(
        loc=init_state,
        scale_diag=jnp.ones((3,)) * 1e-4
    )

    policy = OpenloopPolicyWithClipping(
        proposal, bijector, parameters
    )

    loop_obj = OpenLoop(
        dynamics, policy
    )

    reward_fn = lambda z: pendulum.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


init_state = jnp.zeros((3,))

batch_mppi_exp = lambda exp_key: mppi_experiment(
    key=exp_key,
    state_dim=2,
    action_dim=1,
    dynamics=dynamics,
    proposal=proposal,
    bijector=bijector,
    make_env=make_env,
    nb_steps=101,
    horizon=20,
    nb_particles=32,
    init_state=init_state,
    tempering=0.75,
)

key = jr.PRNGKey(1337)

nb_experiments = 25
batch_keys = jr.split(key, nb_experiments)
samples, costs = jax.vmap(batch_mppi_exp)(batch_keys)

print(f"mean: {jnp.mean(costs):.2f}", f"std: {jnp.std(costs):.2f}")
