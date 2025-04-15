from functools import partial
from typing import Callable, Dict

import jax
from jax import Array, random, numpy as jnp

from distrax import Distribution

from psoc.core import (
    PRNGKey,
    Parameters,
    SMCState,
    SMCParticles,
    TransitionPrior,
    Policy,
    TransitionPosterior,
    RewardFn,
)
from psoc.new_utils import (
    resample,
    propagate,
    log_potential,
    systematic_resampling,
    custom_split
)


def smc_init(
    rng_key: PRNGKey,
    num_particles: int,
    init_prior: Distribution,
    policy_prior: Policy,
    policy_prior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
) -> SMCState:

    key, sub_key = random.split(rng_key)
    states = init_prior.sample(
        seed=sub_key, sample_shape=(num_particles,)
    )
    actions = jnp.zeros((num_particles, policy_prior.dim))

    particles = SMCParticles(states, actions)
    return SMCState(
        particles=particles,
        log_weights=jnp.zeros((num_particles,)),
        weights=jnp.ones((num_particles,)) / num_particles,
        resampling_indices=jnp.zeros((num_particles,), dtype=jnp.int32),
        rewards=jnp.zeros(num_particles),
    )


def smc_step(
    rng_key: PRNGKey,
    smc_state: SMCState,
    trans_prior: TransitionPrior,
    policy_prior: Policy,
    policy_prior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    resample_fn: Callable,
) -> tuple[SMCState, Array]:

    num_particles = smc_state.weights.shape[0]

    key, resample_key = random.split(rng_key, 2)
    smc_state = resample(resample_key, smc_state, resample_fn)
    particles = smc_state.particles
    resampling_indices = smc_state.resampling_indices

    key, action_key = random.split(key, 2)
    actions, _ = policy_prior.sample(action_key, particles.states, policy_prior_params)

    key, propagate_keys = custom_split(key, num_particles + 1)
    states = jax.vmap(propagate, in_axes=(0, None, 0, 0))(
        propagate_keys, trans_prior, particles.states, actions
    )

    prev_actions = smc_state.particles.actions
    log_potentials, rewards = jax.vmap(
        log_potential, in_axes=(0, 0, 0, None, None, None)
    )(states, actions, particles.actions, reward_fn, slew_rate_penalty, tempering)

    log_weights = log_potentials + smc_state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jax.nn.softmax(log_weights)

    log_marginal = logsum_weights - jax.nn.logsumexp(smc_state.log_weights)

    particles = SMCParticles(states, actions)
    smc_state = SMCState(
        particles=particles,
        weights=weights,
        log_weights=log_weights,
        resampling_indices=resampling_indices,
        rewards=rewards,
    )
    return smc_state, log_marginal


@partial(
    jax.jit,
    static_argnames=(
        "num_time_steps",
        "num_particles",
        "init_prior",
        "trans_prior",
        "policy_prior",
        "reward_fn",
        "resample_fn",
    )
)
def smc(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_particles: int,
    init_prior: Distribution,
    trans_prior: TransitionPrior,
    policy_prior: Policy,
    policy_prior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    resample_fn: Callable = systematic_resampling
) -> tuple[SMCState, Array]:

    def smc_loop(carry, key):
        smc_state, log_marginal = carry
        smc_state, log_marginal_incr = \
            smc_step(
                rng_key=key,
                smc_state=smc_state,
                trans_prior=trans_prior,
                policy_prior=policy_prior,
                policy_prior_params=policy_prior_params,
                reward_fn=reward_fn,
                slew_rate_penalty=slew_rate_penalty,
                tempering=tempering,
                resample_fn=resample_fn,
            )

        log_marginal += log_marginal_incr
        return (smc_state, log_marginal), smc_state

    init_key, loop_key = random.split(rng_key, 2)
    init_smc_state = \
        smc_init(
            rng_key=init_key,
            num_particles=num_particles,
            init_prior=init_prior,
            policy_prior=policy_prior,
            policy_prior_params=policy_prior_params,
            reward_fn=reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
        )

    (_, log_marginal), smc_states = jax.lax.scan(
        f=smc_loop,
        init=(init_smc_state, jnp.array(0.0)),
        xs=random.split(loop_key, num_time_steps),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    smc_states = concat_trees(init_smc_state, smc_states)
    return smc_states, log_marginal


@partial(jax.jit, static_argnames=("resample", "resample_fn"))
def backward_tracing(
    rng_key: PRNGKey,
    smc_states: SMCState,
    resample: bool = True,
    resample_fn: Callable = systematic_resampling,
) -> SMCParticles:
    """Genealogy tracking to get the smoothed trajectories.

    Args:
        rng_key: The random number generator key.
        smc_states: The states of the SMC.
        resample: If True, sample the genealogy, otherwise trace back all final
          particles.
        resample_fn: The resampling function.

    Returns:
        The traced smc particles.
    """
    _, num_particles = smc_states.weights.shape

    resampling_idx = jax.lax.select(
        resample,
        resample_fn(rng_key, smc_states.weights[-1], num_particles),
        jnp.arange(num_particles, dtype=jnp.int32),
    )

    last_particles = jax.tree.map(
        lambda x: x[-1, resampling_idx],
        smc_states.particles
    )

    def tracing_fn(carry, args):
        idx = carry
        particles, resampling_indices = args
        a = resampling_indices[idx]
        ancestors = jax.tree.map(lambda x: x[a], particles)
        return a, (a, ancestors)

    _, (_, traced_particles) = jax.lax.scan(
        f=tracing_fn,
        init=resampling_idx,
        xs=(
            jax.tree.map(lambda x: x[:-1], smc_states.particles),
            smc_states.resampling_indices[1:],
        ),
        reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    traced_particles = concat_trees(traced_particles, last_particles)
    return traced_particles


def reg_smc_init(
    rng_key: PRNGKey,
    num_particles: int,
    init_prior: Distribution,
    policy_posterior: Policy,
    policy_posterior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
) -> SMCState:

    key, sub_key = random.split(rng_key)
    states = init_prior.sample(
        seed=sub_key, sample_shape=(num_particles,)
    )
    actions = jnp.zeros((num_particles, policy_posterior.dim))

    particles = SMCParticles(
        states=states, actions=actions,
    )
    return SMCState(
        particles=particles,
        log_weights=jnp.zeros((num_particles,)),
        weights=jnp.ones((num_particles,)) / num_particles,
        resampling_indices=jnp.zeros((num_particles,), dtype=jnp.int32),
        rewards=jnp.zeros(num_particles),
    )


def reg_smc_step(
    rng_key: PRNGKey,
    smc_state: SMCState,
    trans_prior: TransitionPrior,
    trans_posterior: TransitionPosterior,
    trans_posterior_params: Parameters,
    policy_prior: Policy,
    policy_prior_params: Parameters,
    policy_posterior: Policy,
    policy_posterior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    damping: float,
    resample_fn: Callable,
) -> tuple[SMCState, Array]:

    num_particles = smc_state.weights.shape[0]

    key, resample_key = random.split(rng_key, 2)
    smc_state = resample(resample_key, smc_state, resample_fn)
    particles = smc_state.particles
    resampling_indices = smc_state.resampling_indices

    # sample actions from policy prior
    key, action_key = random.split(key, 2)
    actions, action_prior_log_prob, _ = policy_prior.sample_and_log_prob(
        action_key, particles.states, policy_prior_params
    )
    action_prop_log_prob = policy_posterior.log_prob(
        actions, particles.states, policy_posterior_params,
    )

    # sample states from transition prior
    key, prop_keys = custom_split(key, num_particles + 1)
    states = \
        jax.vmap(trans_prior.sample)(prop_keys, particles.states, actions)
    states_prior_log_prob = \
        jax.vmap(trans_prior.log_prob)(states, particles.states, actions)
    states_prop_log_prob = \
        trans_posterior.log_prob(states, particles.states, actions, trans_posterior_params)

    log_potentials, rewards = jax.vmap(
        log_potential, in_axes=(0, 0, 0, None, None, None)
    )(states, actions, particles.actions, reward_fn, slew_rate_penalty, tempering)

    log_weights = smc_state.log_weights \
                  + (1. - damping) * log_potentials \
                  - damping * states_prior_log_prob \
                  + damping * states_prop_log_prob \
                  - damping * action_prior_log_prob \
                  + damping * action_prop_log_prob

    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jax.nn.softmax(log_weights)

    log_marginal = logsum_weights - jax.nn.logsumexp(smc_state.log_weights)

    particles = SMCParticles(states, actions)
    smc_state = SMCState(
        particles=particles,
        weights=weights,
        log_weights=log_weights,
        resampling_indices=resampling_indices,
        rewards=rewards,
    )
    return smc_state, log_marginal


@partial(
    jax.jit,
    static_argnames=(
        "num_time_steps",
        "num_particles",
        "init_prior",
        "trans_prior",
        "trans_posterior",
        "policy_prior",
        "policy_posterior",
        "reward_fn",
        "resample_fn",
    )
)
def regularized_smc(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_particles: int,
    init_prior: Distribution,
    trans_prior: TransitionPrior,
    trans_posterior: TransitionPosterior,
    trans_posterior_params: Parameters,
    policy_prior: Policy,
    policy_prior_params: Parameters,
    policy_posterior: Policy,
    policy_posterior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    damping: float = 0.0,
    resample_fn: Callable = systematic_resampling
) -> tuple[SMCState, Array]:

    def reg_smc_loop(carry, key):
        smc_state, log_marginal = carry
        smc_state, log_marginal_incr = \
            reg_smc_step(
                rng_key=key,
                smc_state=smc_state,
                trans_prior=trans_prior,
                trans_posterior=trans_posterior,
                trans_posterior_params=trans_posterior_params,
                policy_prior=policy_prior,
                policy_prior_params=policy_prior_params,
                policy_posterior=policy_posterior,
                policy_posterior_params=policy_posterior_params,
                reward_fn=reward_fn,
                slew_rate_penalty=slew_rate_penalty,
                tempering=tempering,
                damping=damping,
                resample_fn=resample_fn,
            )

        log_marginal += log_marginal_incr
        return (smc_state, log_marginal), smc_state

    init_key, loop_key = random.split(rng_key, 2)
    init_smc_state = \
        reg_smc_init(
            rng_key=init_key,
            num_particles=num_particles,
            init_prior=init_prior,
            policy_posterior=policy_posterior,
            policy_posterior_params=policy_posterior_params,
            reward_fn=reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
        )

    (_, log_marginal), smc_states = jax.lax.scan(
        f=reg_smc_loop,
        init=(init_smc_state, jnp.array(0.0)),
        xs=random.split(loop_key, num_time_steps),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    smc_states = concat_trees(init_smc_state, smc_states)
    return smc_states, log_marginal
