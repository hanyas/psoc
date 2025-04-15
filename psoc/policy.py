from functools import partial
from typing import Callable, Optional

import jax
import optax

from jax import Array, random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from distrax import (
    Bijector,
    Transformed,
    MultivariateNormalDiag,
)

from psoc.core import (
    PRNGKey,
    Parameters,
    SMCParticles,
    Policy,
    TransitionPosterior,
)


class NeuralGaussPolicy(nn.Module):
    feature_fn: Callable
    hidden_size: tuple[int, ...]
    action_dim: int
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, state: Array) -> tuple[Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.action_dim)

        x = self.feature_fn(state)
        for size in self.hidden_size:
            x = nn.relu(nn.Dense(size)(x))
        y = nn.Dense(self.action_dim)(x)
        return y, log_std

    @property
    def dim(self):
        return self.action_dim


def create_neural_gauss_policy(
    network: NeuralGaussPolicy,
    bijector: Bijector
) -> Policy:
    """
    Creates a squashed neural policy that conforms to the Policy interface.

    Args:
        network (NeuralGaussPolicy): The neural network used for the policy
        bijector (Chain): policy bijector to enforce action limits

    Returns:
        Policy: A policy that implements the Policy interface
    """

    def sample(
        rng_key: PRNGKey,
        states: Array,
        params: Parameters,
    ) -> tuple[Array, Array]:
        mean, log_std = network.apply({"params": params}, states)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        sample = dist.sample(seed=rng_key)
        return sample, bijector.forward(mean)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        states: Array,
        params: Parameters,
    ) -> tuple[Array, Array, Array]:
        mean, log_std = network.apply({"params": params}, states)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        sample, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return sample, log_prob, bijector.forward(mean)

    def log_prob(
        actions: Array,
        states: Array,
        params: Parameters,
    ) -> Array:
        mean, log_std = network.apply({"params": params}, states)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        return dist.log_prob(actions)

    def pathwise_log_prob(
        particles: SMCParticles,
        params: Parameters
    ) -> Array:
        def body(t, log_probs):
            actions = particles.actions[t]
            states = particles.states[t - 1]
            log_prob_incs = log_prob(actions, states, params)
            return log_probs + log_prob_incs

        num_time_steps, batch_size, _ = particles.actions.shape
        init_log_probs = jnp.zeros(batch_size)
        log_probs = jax.lax.fori_loop(1, num_time_steps, body, init_log_probs)
        return log_probs

    def entropy(params: Parameters) -> Array:
        sigma = jnp.diag(jnp.exp(2. * params["log_std"]))
        return 0.5 * (
            network.dim * jnp.log(2.0 * jnp.pi * jnp.exp(1))
            + jnp.linalg.slogdet(sigma)[1]
        )

    def init(
        rng_key: PRNGKey,
        state_dim: int,
        action_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        input_key, param_key = random.split(rng_key, 2)
        dummy_input = random.normal(input_key, (batch_dim, state_dim))
        init_params = network.init(param_key, dummy_input)["params"]
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=init_params,
            tx=optax.adamw(learning_rate)
        )
        return train_state

    return Policy(
        dim=network.dim,
        init=init,
        sample=sample,
        log_prob=log_prob,
        pathwise_log_prob=pathwise_log_prob,
        sample_and_log_prob=sample_and_log_prob,
        entropy=entropy,
    )


@partial(jax.jit, static_argnames="policy")
def train_neural_gauss_policy_pathwise(
    policy: Policy,
    train_state: TrainState,
    particles: SMCParticles,
) -> tuple[TrainState, Array]:
    def loss_fn(params):
        log_probs = policy.pathwise_log_prob(particles, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


@partial(jax.jit, static_argnames="policy")
def train_neural_gauss_policy_stepwise(
    policy: Policy,
    train_state: TrainState,
    actions: Array,
    states: Array,
    damping: float = 1.0
):
    def loss_fn(params):
        log_probs = damping * policy.log_prob(actions, states, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


def create_gauss_policy(
    dim: int,
    log_std: Array,
    bijector: Bijector
) -> Policy:
    """
    Creates a squashed gaussian policy that conforms to the Policy interface.

    Args:
        dim (int): The dimension of the action space
        log_std (Array): The neural network used for the policy
        bijector (Chain): policy bijector to enforce action limits

    Returns:
        Policy: A policy that implements the Policy interface
    """

    def sample(
        rng_key: PRNGKey,
        states: Optional[Array] = None,
        params: Optional[Parameters] = None,
    ) -> tuple[Array, Array]:
        mean = jnp.zeros((dim,))
        scale_diag = jnp.exp(log_std) * jnp.ones((dim,))
        base = MultivariateNormalDiag(loc=mean, scale_diag=scale_diag)
        dist = Transformed(distribution=base, bijector=bijector)
        sample = dist.sample(seed=rng_key)
        return sample, bijector.forward(mean)

    def log_prob(
        actions: Array,
        states: Optional[Array] = None,
        params: Optional[Parameters] = None
    ) -> Array:
        mean = jnp.zeros((dim,))
        scale_diag = jnp.exp(log_std) * jnp.ones((dim,))
        base = MultivariateNormalDiag(loc=mean, scale_diag=scale_diag)
        dist = Transformed(distribution=base, bijector=bijector)
        return dist.log_prob(actions)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        states: Optional[Array] = None,
        params: Optional[Parameters] = None,
    ) -> tuple[Array, Array, Array]:
        mean = jnp.zeros((dim,))
        scale_diag = jnp.exp(log_std) * jnp.ones((dim,))
        base = MultivariateNormalDiag(loc=mean, scale_diag=scale_diag)
        dist = Transformed(distribution=base, bijector=bijector)
        sample, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return sample, log_prob, bijector.forward(mean)

    def pathwise_log_prob(
        particles: SMCParticles,
        params: Parameters
    ) -> Array:
        raise NotImplementedError

    def entropy(params: Parameters) -> Array:
        raise NotImplementedError

    def init(
        rng_key: PRNGKey,
        state_dim: int,
        action_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        raise NotImplementedError

    return Policy(
        dim=dim,
        init=init,
        sample=sample,
        log_prob=log_prob,
        pathwise_log_prob=pathwise_log_prob,
        sample_and_log_prob=sample_and_log_prob,
        entropy=entropy,
    )


class NeuralGaussTransition(nn.Module):
    hidden_size: tuple[int, ...]
    state_dim: int
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, state: Array, action: Array) -> tuple[Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.state_dim)

        x = jnp.concatenate([state, action], axis=-1)
        for size in self.hidden_size:
            x = nn.relu(nn.Dense(size)(x))
        y = nn.Dense(self.state_dim)(x)
        return y, log_std

    @property
    def dim(self):
        return self.state_dim


def create_neural_gauss_transition(
    network: NeuralGaussTransition,
) -> TransitionPosterior:
    """
    Creates a squashed neural policy that conforms to the Policy interface.

    Args:
        network (NeuralGaussPolicy): The neural network used for the policy

    Returns:
        Policy: A policy that implements the Policy interface
    """

    def sample(
        rng_key: PRNGKey,
        states: Array,
        actions: Array,
        params: Parameters,
    ) -> Array:
        mean, log_std = network.apply({"params": params}, states, actions)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sample = dist.sample(seed=rng_key)
        return sample

    def sample_and_log_prob(
        rng_key: PRNGKey,
        states: Array,
        actions: Array,
        params: Parameters,
    ) -> tuple[Array, Array]:
        mean, log_std = network.apply({"params": params}, states, actions)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sample, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return sample, log_prob

    def log_prob(
        next_states: Array,
        states: Array,
        actions: Array,
        params: Parameters,
    ) -> Array:
        mean, log_std = network.apply({"params": params}, states, actions)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return dist.log_prob(next_states)

    def pathwise_log_prob(
        particles: SMCParticles,
        params: Parameters
    ) -> Array:
        def body(t, log_probs):
            states = particles.states[t - 1]
            actions = particles.actions[t]
            next_states = particles.states[t]
            log_prob_incs = log_prob(next_states, states, actions, params)
            return log_probs + log_prob_incs

        num_time_steps, batch_size, _ = particles.actions.shape
        init_log_probs = jnp.zeros(batch_size)
        log_probs = jax.lax.fori_loop(1, num_time_steps, body, init_log_probs)
        return log_probs

    def init(
        rng_key: PRNGKey,
        state_dim: int,
        action_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        state_key, action_key, param_key = random.split(rng_key, 3)
        dummy_state = random.normal(state_key, (batch_dim, state_dim))
        dummy_action = random.normal(action_key, (batch_dim, action_dim))
        init_params = network.init(param_key, dummy_state, dummy_action)["params"]
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=init_params,
            tx=optax.adamw(learning_rate)
        )
        return train_state

    return TransitionPosterior(
        dim=network.dim,
        sample=sample,
        log_prob=log_prob,
        pathwise_log_prob=pathwise_log_prob,
        sample_and_log_prob=sample_and_log_prob,
        init=init,
    )


@partial(jax.jit, static_argnames="transition")
def train_neural_gauss_transition_stepwise(
    transition: TransitionPosterior,
    train_state: TrainState,
    next_states: Array,
    states: Array,
    actions: Array,
    damping: float = 1.0
):
    def loss_fn(params):
        log_probs = damping * transition.log_prob(next_states, states, actions, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss
