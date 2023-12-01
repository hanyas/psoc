from typing import Dict, Union
from typing import Callable

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
    last_ref = filter_particles[-1, b]

    def body(carry, args):
        next_b = carry
        particles, ancestors = args
        b = ancestors[next_b]
        return b, particles[b]

    _, reference = jl.scan(
        body, b, (filter_particles[:-1], filter_ancestors), reverse=True
    )

    reference = jnp.vstack((reference, last_ref))
    return reference


def _rao_blackwell_backward_sampling(
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
    ) # (nb_samples, nb_partilces)

    # Same here
    loss_fn_ = jax.vmap(loss_fn, in_axes=(0, 0, None))  # (nb_samples, nb_partilces)
    score_fn_ = jax.vmap(score_fn, in_axes=(0, 0, None))  # (nb_samples, nb_partilces)

    # last time step
    key, sub_key = jr.split(key, 2)
    last_bs = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1], shape=(nb_samples,))
    # The first value of last_bs is our actual reference trajectory, the rest is used for R-B.
    # We select [0] when talking about the reference.

    def body(carry, args):
        next_refs, loss_val, score_val = carry
        keys, particles, weights = args

        log_pdf_weights = trans_logpdf(particles, next_refs)  # noqa, shape is (nb_samples, nb_particles)
        log_mod_weights = log_pdf_weights + jnp.log(weights)[None, :]  # weights correspond to particles
        log_mod_weights_norm = logsumexp(log_mod_weights, axis=1, keepdims=True)  # Normalizer per reference
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        choice_fn = lambda k, p: jr.choice(k, a=nb_particles, p=p)
        bs = jax.vmap(choice_fn, in_axes=(0, 0))(keys, mod_weights)
        refs = particles[bs]

        loss_val += jnp.mean(loss_fn_(refs, next_refs, loss_fn_params))
        scores = score_fn_(refs, next_refs, loss_fn_params)  # shape is (nb_samples, ...)
        score_val = jax.tree_map(lambda a, b: a + jnp.mean(b, 0), score_val, scores)

        return (refs, loss_val, score_val), next_refs[0]

    key, sub_key = jr.split(key, 2)
    res_keys = jr.split(sub_key, (nb_steps, nb_samples))

    init_loss = 0.0
    init_score = jax.tree_map(lambda a: jnp.zeros_like(a), loss_fn_params)

    (first_refs, final_loss, final_score), reference = \
        jl.scan(
            body,
            (filter_particles[-1, last_bs], init_loss, init_score),
            (res_keys, filter_particles[:-1], filter_weights[:-1]),
            reverse=True
        )

    reference = jnp.vstack((first_refs[0], reference))
    return reference, final_loss, final_score


def _backward_sampling(
    key: jax.Array,
    filter_particles: jnp.ndarray,
    filter_weights: jnp.ndarray,
    transition_model: Union[OpenLoop, FeedbackLoop],
):
    nb_particles = filter_particles.shape[1]
    trans_logpdf = jax.vmap(transition_model.logpdf, in_axes=(0, None))

    # last time step
    key, sub_key = jr.split(key, 2)
    b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
    last_ref = filter_particles[-1, b]

    def body(carry, args):
        key, next_ref = carry
        particles, weights = args

        log_mod_weights = jnp.log(weights) + trans_logpdf(particles, next_ref)
        log_mod_weights_norm = logsumexp(log_mod_weights)
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        key, sub_key = jr.split(key, 2)
        b = jr.choice(sub_key, a=nb_particles, p=mod_weights)
        ref = particles[b]
        return (key, ref), next_ref

    (_, first_ref), reference = \
        jl.scan(
            body,
            (key, last_ref),
            (filter_particles[:-1], filter_weights[:-1]),
            reverse=True
        )

    reference = jnp.vstack((first_ref, reference))
    return reference


def _abstract_csmc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    reference: jnp.ndarray,
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
        prev_particles, prev_weights = carry
        next_ref, res_key, prop_key = args

        rest_ancestors = jr.choice(res_key, a=nb_particles,
                                   shape=(nb_particles - 1,), p=prev_weights)
        ancestors = jnp.hstack((0, rest_ancestors))

        resampled_particles = prev_particles[rest_ancestors]
        next_rest_particles = _propagate(prop_key, resampled_particles)
        next_particles = jnp.insert(next_rest_particles, 0, next_ref, 0)

        log_next_weights = _log_weights(next_particles)
        next_weights_norm = logsumexp(log_next_weights)
        next_weights = jnp.exp(log_next_weights - next_weights_norm)

        return (next_particles, next_weights), \
            (prev_particles, prev_weights, ancestors)

    key, sub_key = jr.split(key, 2)
    init_rest_particles = prior.sample(seed=sub_key, sample_shape=(nb_particles - 1,))
    init_particles = jnp.insert(init_rest_particles, 0, reference[0], 0)
    init_weights = jnp.ones((nb_particles,)) / nb_particles

    keys = jr.split(key, nb_steps)
    key, resampling_keys = keys[0], keys[1:]

    keys = jr.split(key, nb_steps)
    key, propagation_keys = keys[0], keys[1:]

    (last_particles, last_weights), \
        (filter_particles, filter_weights, filter_ancestors) = \
        jl.scan(
            body,
            (init_particles, init_weights),
            (reference[1:], resampling_keys, propagation_keys)
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


def vanilla_csmc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    reference: jnp.ndarray,
    prior: distrax.Distribution,
    transition_model: Union[OpenLoop, FeedbackLoop],
    log_observation: Callable,
):
    key, sub_key = jr.split(key, 2)
    return _abstract_csmc(
        sub_key,
        nb_steps,
        nb_particles,
        reference,
        prior,
        transition_model,
        log_observation,
        _backward_sampling,
    )


def rao_blackwell_csmc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    nb_samples: int,
    reference: jnp.ndarray,
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
        return _rao_blackwell_backward_sampling(
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
    return _abstract_csmc(
        sub_key,
        nb_steps,
        nb_particles,
        reference,
        prior,
        transition_model,
        log_observation,
        _backward_sampling_fn,
    )
