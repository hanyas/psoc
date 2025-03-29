from typing import Dict

import jax
from jax import numpy as jnp
from jax import random as jr

import distrax
from flax import linen as nn

from psoc.abstract import StochasticDynamics
from psoc.abstract import Gaussian, GaussMarkov
from psoc.abstract import OpenloopPolicyWithClipping
from psoc.abstract import OpenloopPolicyWithSquashing
from psoc.abstract import OpenLoop
from psoc.bijector import Tanh

from psoc.utils import identity_constraint, positivity_constraint
from psoc.experiments.openloop.common import csmc_experiment
from psoc.environments.openloop import cartpole_env as cartpole

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


dynamics = StochasticDynamics(
    dim=4,
    ode=cartpole.ode,
    step=0.05,
    stddev=1e-2 * jnp.ones((4,))
)

proposal = GaussMarkov(
    dim=1,
    step=0.05,
    inv_length_init=nn.initializers.constant(100.0),
    diffusion_init=nn.initializers.constant(200.0)
)

proposal = Gaussian(
    dim=1,
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 50.0),
    Tanh(),
])


def make_env(
    init_state: jnp.ndarray,
    parameters: Dict,
    tempering: float,
):
    prior_dist = distrax.MultivariateNormalDiag(
        loc=init_state,
        scale_diag=jnp.ones((5,)) * 1e-4
    )

    policy = OpenloopPolicyWithClipping(
        proposal, bijector, parameters,
    )

    # policy = OpenloopPolicyWithSquashing(
    #     proposal, bijector, parameters, positivity_constraint
    # )

    loop_obj = OpenLoop(dynamics, policy)

    reward_fn = lambda z: cartpole.reward(z, tempering)
    return prior_dist, loop_obj, reward_fn


init_state = jnp.zeros((5,))

batch_csmc_exp = lambda exp_key: csmc_experiment(
    key=exp_key,
    state_dim=4,
    action_dim=1,
    dynamics=dynamics,
    proposal=proposal,
    bijector=bijector,
    make_env=make_env,
    nb_steps=101,
    horizon=20,
    nb_particles=64,
    nb_samples=64,
    init_state=init_state,
    tempering=0.5,
    nb_iter=1,
    learning_rate=1e-1,
)

key = jr.PRNGKey(1337)

nb_experiments = 25
batch_keys = jr.split(key, nb_experiments)
samples, costs = jax.vmap(batch_csmc_exp)(batch_keys)

print(f"mean: {jnp.mean(costs):.2f}", f"std: {jnp.std(costs):.2f}")
