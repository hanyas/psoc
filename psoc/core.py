from typing import Dict, NamedTuple, Protocol, Union, Any, Optional

import chex
from jax import Array

from flax.core import FrozenDict
from flax.training.train_state import TrainState

PRNGKey = chex.PRNGKey
Parameters = Union[Dict[str, Any], FrozenDict[str, Any]]


class SMCParticles(NamedTuple):
    states: Array
    actions: Array


class SMCState(NamedTuple):
    r"""State of the particle filter.

    Attributes:
        particles: NamedTuple of the states and actions $(z_t^{1:N}, a_t^{1:N}, c_t^{1:N})$.
        log_weights: Log weights of states and actions $(z_t^{1:N}, a_t^{1:N})$.
        weights: Weights of states and actions $(z_t^{1:N}, a_t^{1:N})$.
        resampling_indices: Resampling indicies of states and actions $(z_t^{1:N}, a_t^{1:N})$.
        rewards: Expected rewards of states and actions $(s_{t}^{1:N}, a_{t-1}^{1:N})$.
    """

    particles: SMCParticles
    log_weights: Array
    weights: Array
    resampling_indices: Array
    rewards: Array


class Reference(NamedTuple):
    particles: SMCParticles


class RewardFn(Protocol):
    def __call__(self, s: Array, a: Array) -> Array:
        r"""The  reward function $r(s_t, a_t)$."""


class SampleTransitionPrior(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array, a: Array) -> Array:
        r"""Sample from $f(s_t \mid s_{t-1}, a_{t-1})$."""


class LogProbTransitionPrior(Protocol):
    def __call__(self, sn: Array, s: Array, a: Array) -> Array:
        r"""Compute the log density of $f(s_t \mid s_{t-1}, a_{t-1})$."""


class TransitionPrior(NamedTuple):
    r"""The transition kernel $f(s_t \mid s_{t-1}, a_{t-1})$."""

    sample: SampleTransitionPrior
    log_prob: LogProbTransitionPrior


class SamplePolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        states: Optional[Array] = None,
        params: Optional[Parameters] = None,
    ) -> tuple[Array, Array]:
        r"""Sample from $\pi_\phi(a_t, \mid s_t)$."""


class LogProbPolicy(Protocol):
    def __call__(
        self,
        actions: Array,
        states: Optional[Array] = None,
        params: Optional[Parameters] = None
    ) -> Array:
        r"""Compute the log density of $\pi_\phi(a_t, \mid s_t,)$."""


class PathwiseLogProbPolicy(Protocol):
    def __call__(
        self,
        particles: SMCParticles,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $\pi_\phi(a_t \mid s_t)$."""


class SampleAndLogProbPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        states: Optional[Array] = None,
        params: Optional[Parameters] = None,
    ) -> tuple[Array, Array, Array]:
        r"""Sample from $\pi_\phi(a_t, \mid s_t)$ and compute its log density."""


class EntropyPolicy(Protocol):
    def __call__(
        self,
        params: Parameters,
    ) -> Array:
        r"""Compute the entropy of $\pi_\phi$."""


class InitializePolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        state_dim: int,
        action_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        r"""Initialize the recurrent state of the policy."""


class Policy(NamedTuple):
    r"""The stochastic recurrent policy $\pi_\phi$."""

    dim: int
    sample: SamplePolicy
    log_prob: LogProbPolicy
    pathwise_log_prob: PathwiseLogProbPolicy
    sample_and_log_prob: SampleAndLogProbPolicy
    entropy: EntropyPolicy
    init: InitializePolicy


class SampleTransitionPosterior(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        states: Array,
        actions: Array,
        params: Parameters,
    ) -> Array:
        r"""Sample from $q(s_{t+1} \mid s_t, a_t)$."""


class LogProbTransitionPosterior(Protocol):
    def __call__(
        self,
        next_states: Array,
        states: Array,
        actions: Array,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $q(s_{t+1} \mid s_t, a_t)$."""


class PathwiseLogProbTransitionPosterior(Protocol):
    def __call__(
        self,
        particles: SMCParticles,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $q(s_{t+1} \mid s_t, a_t)$."""


class SampleAndLogProbTransitionPosterior(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        states: Array,
        actions: Array,
        params: Parameters
    ) -> tuple[Array, Array]:
        r"""Sample from $q(s_{t+1} \mid s_t, a_t)$ and compute its log density."""


class InitializeTransitionPosterior(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        state_dim: int,
        action_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        r"""Initialize the recurrent state of the policy."""


class TransitionPosterior(NamedTuple):
    r"""The posterior distribution $q(s_{t+1} \mid s_t, a_t)$."""

    dim: int
    sample: SampleTransitionPosterior
    log_prob: LogProbTransitionPosterior
    pathwise_log_prob: PathwiseLogProbTransitionPosterior
    sample_and_log_prob: SampleAndLogProbTransitionPosterior
    init: InitializeTransitionPosterior
