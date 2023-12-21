from typing import Callable, Dict, Union

import jax
from jax import random as jr
from jax import numpy as jnp
from jax import lax as jl

from jax.scipy.special import logsumexp

import distrax

from psoc.abstract import OpenLoop, FeedbackLoop


def _backward_tracing(
    key: jax.Array,
    filter_particles: jnp.ndarray,
    filter_ancestors: jnp.ndarray,
    filter_weights: jnp.ndarray,
):
    nb_particles = filter_particles.shape[1]

    # last time step
    key, sub_key = jr.split(key, 2)
    b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
    last_state = filter_particles[-1, b]

    def body(carry, args):
        next_b = carry
        particles, ancestors = args
        b = ancestors[next_b]
        return b, particles[b]

    _, sample = jl.scan(
        body, b, (filter_particles[:-1], filter_ancestors), reverse=True
    )

    sample = jnp.vstack((sample, last_state))
    return sample


def _backward_sampling(
    key: jax.Array,
    filter_particles: jnp.ndarray,
    filter_weights: jnp.ndarray,
    transition_model: FeedbackLoop,
):
    nb_particles = filter_particles.shape[1]
    trans_logpdf = jax.vmap(transition_model.logpdf, in_axes=(0, None))

    # last time step
    key, sub_key = jr.split(key, 2)
    b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
    last_state = filter_particles[-1, b]

    def body(carry, args):
        key, next_state = carry
        particles, weights = args

        log_mod_weights = jnp.log(weights) + trans_logpdf(particles, next_state)
        log_mod_weights_norm = logsumexp(log_mod_weights)
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        key, sub_key = jr.split(key, 2)
        b = jr.choice(sub_key, a=nb_particles, p=mod_weights)
        state = particles[b]
        return (key, state), next_state

    (_, first_state), sample = \
        jl.scan(
            body,
            (key, last_state),
            (filter_particles[:-1], filter_weights[:-1]),
            reverse=True,
    )

    sample = jnp.vstack((first_state, sample))
    return sample


def _batch_backward_sampling(
    key: jax.Array,
    nb_samples: int,
    filter_particles: jnp.ndarray,
    filter_weights: jnp.ndarray,
    transition_model: Union[OpenLoop, FeedbackLoop],
):
    nb_steps = filter_particles.shape[0] - 1
    nb_particles = filter_particles.shape[1]

    # if x_t has shape (N,) and x_t_p_1 has shape (M,), then
    # trans_logpdf(x_t, x_t_p_1) has shape (M, N)
    trans_logpdf = jax.vmap(
        jax.vmap(transition_model.logpdf, in_axes=(0, None)),
        in_axes=(None, 0)
    )  # (nb_samples, nb_partilces)

    # last time step
    key, sub_key = jr.split(key, 2)
    last_bs = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1], shape=(nb_samples,))
    last_states = filter_particles[-1, last_bs]

    def body(carry, args):
        next_states = carry
        keys, particles, weights = args

        log_pdf_weights = trans_logpdf(particles, next_states)  # (nb_samples, nb_particles)
        log_mod_weights = log_pdf_weights + jnp.log(weights)[None, :]  # weights correspond to particles
        log_mod_weights_norm = logsumexp(log_mod_weights, axis=1, keepdims=True)  # Normalizer
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        choice_fn = lambda k, p: jr.choice(k, a=nb_particles, p=p)
        bs = jax.vmap(choice_fn, in_axes=(0, 0))(keys, mod_weights)
        states = particles[bs]

        return states, next_states

    key, sub_key = jr.split(key, 2)
    res_keys = jr.split(sub_key, (nb_steps, nb_samples))

    first_states, samples = \
        jl.scan(
            body,
            last_states,
            (res_keys, filter_particles[:-1], filter_weights[:-1]),
            reverse=True
        )

    samples = jnp.insert(samples, 0, first_states, 0)
    samples = jnp.swapaxes(samples, 0, 1)
    return samples


def _batch_backward_sampling_with_score(
    key: jax.Array,
    nb_samples: int,
    filter_particles: jnp.ndarray,
    filter_weights: jnp.ndarray,
    transition_model: Union[OpenLoop, FeedbackLoop],
    loss_fn: Callable,
    score_fn: Callable,
    loss_fn_params: Dict,
):
    nb_steps = filter_particles.shape[0] - 1
    nb_particles = filter_particles.shape[1]

    # if x_t has shape (N,) and x_t_p_1 has shape (M,), then
    # trans_logpdf(x_t, x_t_p_1) has shape (M, N)
    trans_logpdf = jax.vmap(
        jax.vmap(transition_model.logpdf, in_axes=(0, None)),
        in_axes=(None, 0)
    )  # (nb_samples, nb_partilces)

    # Same here
    loss_fn_ = jax.vmap(loss_fn, in_axes=(0, 0, None))  # (nb_samples, nb_partilces)
    score_fn_ = jax.vmap(score_fn, in_axes=(0, 0, None))  # (nb_samples, nb_partilces)

    # last time step
    key, sub_key = jr.split(key, 2)
    last_bs = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1], shape=(nb_samples,))
    last_states = filter_particles[-1, last_bs]

    def body(carry, args):
        next_states, loss_val, score_val = carry
        keys, particles, weights = args

        log_pdf_weights = trans_logpdf(particles, next_states)  # (nb_samples, nb_particles)
        log_mod_weights = log_pdf_weights + jnp.log(weights)[None, :]  # weights correspond to particles
        log_mod_weights_norm = logsumexp(log_mod_weights, axis=1, keepdims=True)  # Normalizer
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        choice_fn = lambda k, p: jr.choice(k, a=nb_particles, p=p)
        bs = jax.vmap(choice_fn, in_axes=(0, 0))(keys, mod_weights)
        states = particles[bs]

        loss_val += jnp.mean(loss_fn_(states, next_states, loss_fn_params))
        scores = score_fn_(states, next_states, loss_fn_params)  # (nb_samples, ...)
        score_val = jax.tree_map(lambda a, b: a + jnp.mean(b, 0), score_val, scores)

        return (states, loss_val, score_val), next_states

    key, sub_key = jr.split(key, 2)
    res_keys = jr.split(sub_key, (nb_steps, nb_samples))

    init_loss = 0.0
    init_score = jax.tree_map(lambda a: jnp.zeros_like(a), loss_fn_params)

    (first_states, final_loss, final_score), samples = \
        jl.scan(
            body,
            (last_states, init_loss, init_score),
            (res_keys, filter_particles[:-1], filter_weights[:-1]),
            reverse=True
        )

    samples = jnp.insert(samples, 0, first_states, 0)
    samples = jnp.swapaxes(samples, 0, 1)
    return samples, final_loss, final_score


def _abstract_smc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    prior: distrax.Distribution,
    transition_model: Union[OpenLoop, FeedbackLoop],
    log_observation: Callable,
    backward_sampling_fn: Callable,
):
    def _propagate(key, particles):
        return transition_model.sample(key, particles)

    def _log_weights(state):
        return log_observation(state)

    def body(carry, args):
        key, prev_particles, prev_weights = carry

        # resample
        key, sub_key = jr.split(key, 2)
        ancestors = jr.choice(sub_key, a=nb_particles,
                              shape=(nb_particles,), p=prev_weights)

        # propagate
        key, sub_key = jr.split(key, 2)
        resampled_particles = prev_particles[ancestors]
        next_particles = _propagate(sub_key, resampled_particles)

        # weights
        log_next_weights = _log_weights(next_particles)
        next_weights_norm = logsumexp(log_next_weights)
        next_weights = jnp.exp(log_next_weights - next_weights_norm)

        return (key, next_particles, next_weights), \
            (prev_particles, prev_weights, ancestors)

    key, init_key, scan_key = jr.split(key, 3)
    init_particles = prior.sample(seed=init_key, sample_shape=(nb_particles,))
    init_weights = jnp.ones((nb_particles,)) / nb_particles

    (_, last_particles, last_weights), \
        (filter_particles, filter_weights, filter_ancestors) = \
        jl.scan(
            body,
            (scan_key, init_particles, init_weights),
            (),
            length=nb_steps-1
        )

    filter_particles = jnp.insert(filter_particles, nb_steps, last_particles, 0)
    filter_weights = jnp.insert(filter_weights, nb_steps, last_weights, 0)

    key, sub_key = jr.split(key, 2)
    return backward_sampling_fn(
        sub_key,
        filter_particles,
        filter_weights,
        transition_model
    )


def smc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    prior: distrax.Distribution,
    transition_model: Union[OpenLoop, FeedbackLoop],
    log_observation: Callable,
):
    def _backward_sampling_fn(
        key,
        filter_particles,
        filter_weights,
        transition_model,
    ):
        return _batch_backward_sampling(
            key,
            nb_samples,
            filter_particles,
            filter_weights,
            transition_model,
        )

    key, sub_key = jr.split(key, 2)
    return _abstract_smc(
        sub_key,
        nb_steps,
        nb_particles,
        prior,
        transition_model,
        log_observation,
        _backward_sampling_fn
    )


def smc_with_score(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    prior: distrax.Distribution,
    transition_model: Union[OpenLoop, FeedbackLoop],
    log_observation: Callable,
    loss_fn: Callable,
    score_fn: Callable,
    loss_fn_params: Dict,
):
    def _backward_sampling_fn(
        key,
        filter_particles,
        filter_weights,
        transition_model,
    ):
        return _batch_backward_sampling_with_score(
            key,
            nb_samples,
            filter_particles,
            filter_weights,
            transition_model,
            loss_fn,
            score_fn,
            loss_fn_params
        )

    key, sub_key = jr.split(key, 2)
    return _abstract_smc(
        sub_key,
        nb_steps,
        nb_particles,
        prior,
        transition_model,
        log_observation,
        _backward_sampling_fn,
    )
