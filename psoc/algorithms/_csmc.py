from typing import Callable, Dict

import jax
from jax import random as jr
from jax import numpy as jnp
from jax import lax as jl

from jax.scipy.special import logsumexp

import distrax

from psoc.abstract import ClosedLoop


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


def _rb_backward_sampling(
    key: jax.Array,
    filter_particles: jnp.ndarray,
    filter_weights: jnp.ndarray,
    transition_model: ClosedLoop,
    score_fn: Callable,
    score_fn_params: Dict,
):
    nb_particles = filter_particles.shape[1]

    # if x_t has shape (N,) and x_t_p_1 has shape (M,), then
    # trans_logpdf(x_t, x_t_p_1) has shape (N, M)
    trans_logpdf = jax.vmap(
        jax.vmap(transition_model.logpdf, in_axes=(0, None)),
        in_axes=(None, 0)
    )

    # Same here
    _score_fn = jax.vmap(
        jax.vmap(score_fn, in_axes=(0, None, None)),
        in_axes=(None, 0, None)
    )

    # last time step
    key, sub_key = jr.split(key, 2)
    last_b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
    # I moved the keys inside the loop because there was a shape mismatch
    # I didn't want to deal with now. We can move them back in later

    def body(carry, args):
        key, next_refs, true_ref_index, scores_val = carry
        particles, weights = args

        log_pdf_weights = trans_logpdf(particles, next_refs)  # noqa
        log_mod_weights = log_pdf_weights + jnp.log(weights)[:, None]  # weights correspond to particles_t
        log_mod_weights_norm = logsumexp(log_mod_weights, axis=1, keepdims=True)  # Get the normalizing constant per reference
        mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)

        # added this key splitting
        _keys = jr.split(key, nb_particles + 1)
        key, sub_keys = _keys[0], _keys[1:]
        bs = jax.vmap(lambda k, p: jr.choice(k, a=nb_particles, p=p), in_axes=(0, 1))(sub_keys, mod_weights)
        _b = bs[true_ref_index]

        # This computes the vectorized scores
        scores = _score_fn(particles, next_refs, score_fn_params)

        # This multiplies the scores with the weights and sums over first dim
        weighted_scores = jax.tree_map(lambda _x: jnp.einsum('ij...,ij->j...', _x, mod_weights), scores)

        # This accumulates the score values
        scores_val = jax.tree_map(lambda _x, _y: _x + _y, scores_val, weighted_scores)

        refs = particles[bs]
        return (key, refs, _b, scores_val), next_refs[true_ref_index]

    # The score is the gradient of the neural net and that has a dict struct in flax
    # I initialize it here by using the parameters and repeat over nb_particles
    # There is probably a better way to do this
    _skeleton = jax.tree_map(lambda _x: jnp.zeros_like(_x), score_fn_params)
    scores_val = jax.tree_map(lambda _x: jnp.repeat(_x[None, ...], nb_particles, axis=0), _skeleton)

    # The initial next_refs values were a single vector, and didn't not have
    # nb_particles size. Can you check that again?
    (_, first_refs, first_b, scores_val), reference = \
        jl.scan(
            body,
            (key, filter_particles[-1], last_b, scores_val),
            (filter_particles[:-1], filter_weights[:-1]),
            reverse=True
        )

    reference = jnp.vstack((first_refs[first_b], reference))
    score = jax.tree_map(lambda _x: jnp.einsum('i...,i->...', _x, filter_weights[-1]), scores_val)
    return reference, score


# def _backward_sampling(
#     key: jax.Array,
#     filter_particles: jnp.ndarray,
#     filter_weights: jnp.ndarray,
#     transition_model: ClosedLoop,
# ):
#     nb_particles = filter_particles.shape[1]
#     trans_logpdf = jax.vmap(transition_model.logpdf, in_axes=(0, None))
#
#     # last time step
#     key, sub_key = jr.split(key, 2)
#     b = jr.choice(sub_key, a=nb_particles, p=filter_weights[-1])
#     last_ref = filter_particles[-1, b]
#
#     def body(carry, args):
#         key, next_ref = carry
#         particles, weights = args
#
#         log_mod_weights = jnp.log(weights) + trans_logpdf(particles, next_ref)
#         log_mod_weights_norm = logsumexp(log_mod_weights)
#         mod_weights = jnp.exp(log_mod_weights - log_mod_weights_norm)
#
#         key, sub_key = jr.split(key, 2)
#         b = jr.choice(sub_key, a=nb_particles, p=mod_weights)
#         ref = particles[b]
#         return (key, ref), next_ref
#
#     (_, first_ref), reference = \
#         jl.scan(
#             body,
#             (key, last_ref),
#             (filter_particles[:-1], filter_weights[:-1]),
#             reverse=True
#         )
#
#     reference = jnp.vstack((first_ref, reference))
#     return reference


def csmc(
    key: jax.Array,
    nb_steps: int,
    nb_particles: int,
    reference: jnp.ndarray,
    prior: distrax.Distribution,
    transition_model: ClosedLoop,
    log_observation: Callable,
    score_fn: Callable,
    score_fn_params: Dict,
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
    sample = _rb_backward_sampling(
        sub_key,
        filter_particles,
        filter_weights,
        transition_model,
        score_fn,
        score_fn_params,
    )

    return sample
