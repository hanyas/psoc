import math
from functools import partial
from typing import Callable

import jax
from jax import Array, random
from jax import numpy as jnp

from psoc.core import (
    PRNGKey,
    Parameters,
    SMCParticles,
    SMCState,
    TransitionPrior,
    Policy,
    RewardFn,
)
from psoc.envs.core import MDPEnv


def propagate(
    rng_key: PRNGKey,
    model: TransitionPrior,
    state: Array,
    action: Array,
) -> Array:
    return model.sample(rng_key, state, action)


def log_potential(
    state: Array,
    action: Array,
    prev_action: Array,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
) -> tuple[Array, Array]:

    rewards = reward_fn(state, action)
    mod_rewards = rewards - \
        slew_rate_penalty * jnp.dot(action - prev_action, action - prev_action)
    return tempering * mod_rewards, rewards


def custom_split(rng_key: PRNGKey, num: int):
    """
    Splits a random number generator key into multiple sub-keys.

    Args:
        rng_key (PRNGKey): The random number generator key to split.
        num (int): The number of sub-keys to generate.

    Returns:
        tuple: A tuple containing the next key and an array of sub-keys.
    """
    key, *sub_keys = random.split(rng_key, num)
    return key, jnp.array(sub_keys)


def log_ess(log_weights: Array) -> Array:
    """Computes the log of the effective sample size.

    Args:
        log_weights: Log-weights of particles.

    Returns:
        The logarithm of the effective sample size.
    """
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)


@partial(jnp.vectorize, signature="(m)->()")
def effective_sample_size(log_weights: Array) -> Array:
    """Computes the effective sample size.

    Args:
        log_weights: Log-weights of particles.

    Returns:
        The effective sample size.
    """
    return jnp.exp(log_ess(log_weights))


def systematic_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    """
    Perform systematic resampling of particles based on their weights.

    Args:
        rng_key (PRNGKey): The random key for sampling.
        weights (Array): The weights of the particles.
        num_samples (int): The number of samples to draw.

    Returns:
        Array: The indices of the resampled particles.
    """
    n = weights.shape[0]
    u = random.uniform(rng_key, ())
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1).astype(jnp.int32)


def multinomial_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    """
    Perform multinomial resampling of particles based on their weights.

    Args:
        rng_key (PRNGKey): The random key for sampling.
        weights (Array): The weights of the particles.
        num_samples (int): The number of samples to draw.

    Returns:
        Array: The indices of the resampled particles.
    """
    idx = random.choice(rng_key, num_samples, shape=(num_samples,), p=weights)
    return idx.astype(jnp.int32)


def resample(
    rng_key: PRNGKey,
    smc_state: SMCState,
    resample_fn: Callable,
    conditional: bool = False,
) -> SMCState:
    num_particles = smc_state.weights.shape[0]

    def true_fn(state: SMCState) -> SMCState:
        resampling_idx = resample_fn(rng_key, state.weights, num_particles)
        # Set zeroth resampling index to zero if conditional resampling is enabled
        resampling_idx = jax.lax.select(
            conditional, resampling_idx.at[0].set(0), resampling_idx
        )
        resampled_particles = jax.tree.map(lambda x: x[resampling_idx], state.particles)
        resampled_rewards = state.rewards[resampling_idx]
        return SMCState(
            particles=resampled_particles,
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            resampling_indices=resampling_idx,
            rewards=resampled_rewards,
        )

    def false_fn(state: SMCState) -> SMCState:
        resampling_idx = jnp.arange(num_particles, dtype=jnp.int32)
        return state._replace(resampling_indices=resampling_idx)

    predicate = effective_sample_size(smc_state.log_weights) < 0.75 * num_particles
    resampled_state = jax.lax.cond(predicate, true_fn, false_fn, smc_state)
    return resampled_state


@partial(jax.jit, static_argnames=("data_size", "batch_size", "skip_last"))
def batch_data(
    rng_key: Array,
    data_size: int,
    batch_size: int,
    skip_last: bool = True,
) -> list[Array]:
    """Generates batched indices.

    Args:
        rng_key (Array): Random key for shuffling the data.
        data_size (int): The size of the dataset to be batched.
        batch_size (int): The size of each batch.
        skip_last (bool, optional): If True, skips the last incomplete batch. Defaults to False.

    Returns:
        list[Array]: A list of batched indices.
    """
    batch_idx = random.permutation(rng_key, data_size)

    if skip_last:
        # Skip incomplete batch
        num_batches = data_size // batch_size
        batch_idx = batch_idx[: num_batches * batch_size]
    else:
        # Include incomplete batch
        num_batches = math.ceil(data_size / batch_size)

    batch_idx = jnp.array_split(batch_idx, num_batches)
    return batch_idx


@jax.jit
def flatten_particle_trajectories(particles: SMCParticles) -> tuple[Array, Array, Array]:
    """
    Aligns particle trajectories in time and concatenates them to flatten the time dimension.

    Args:
        particles (SMCParticles): The particle trajectories to be flattened.
            Must include a time component with shape (num_time_steps, batch_size, ...).

    Returns:
        SMCParticles: The flattened particle trajectories with the time dimension concatenated.
    """

    if particles.states.ndim != 3:
        raise ValueError("`particles` must include a time component.")

    actions = particles.actions[1:].reshape((-1, particles.actions.shape[-1]))
    states = particles.states[:-1].reshape((-1, particles.states.shape[-1]))
    next_states = particles.states[1:].reshape((-1, particles.states.shape[-1]))
    return next_states, states, actions


@partial(jax.jit, static_argnames=("env_obj", "policy", "num_samples"))
def policy_evaluation(
    rng_key: PRNGKey,
    env_obj: MDPEnv,
    policy: Policy,
    params: Parameters,
    num_samples: int = 128,
):
    """
    Deploy the (deterministic) policy to sample trajectories and evaluate the average reward.

    Args:
        rng_key (PRNGKey): The random number generator key.
        env_obj (POMDPEnv): The partially observable Markov decision process environment.
        policy (RecurrentPolicy): The policy to be evaluated.
        params (Parameter): Stochastic policy parameters.
        num_samples (int, optional): The number of samples to draw. Defaults to 100.

    Returns:
        tuple: A tuple containing the expected reward, states, and actions.
    """

    def body(states, key):
        # Sample actions.
        key, action_key = random.split(key)
        _, actions = policy.sample(action_key, states, params)

        # Sample next states.
        key, state_keys = custom_split(key, num_samples + 1)
        states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

        # Compute rewards.
        rewards = jax.vmap(env_obj.reward_fn, in_axes=(0, 0))(states, actions)
        return states, (states, actions, rewards)

    key, state_key = random.split(rng_key)
    init_states = env_obj.prior_dist.sample(seed=state_key, sample_shape=num_samples)

    _, (states, actions, rewards) = \
        jax.lax.scan(
            f=body,
            init=init_states,
            xs=random.split(key, env_obj.num_time_steps),
    )
    states = jnp.concatenate([init_states[None, ...], states], axis=0)
    expected_reward = jnp.mean(jnp.sum(rewards, axis=0))
    return expected_reward, states, actions


def damping_schedule(step, total_steps, steepness=1.0, init_value=0.1, max_value=0.95):
    x = (step / total_steps) * 12.0 - 6.0
    beta = 1 / (1 + jnp.exp(-steepness * x))
    # Rescale beta to start from initial_value and go up to max_value
    beta_scaled = init_value + (max_value - init_value) * beta
    return jnp.minimum(beta_scaled, max_value)
