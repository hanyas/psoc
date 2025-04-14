from typing import Callable, NamedTuple

from jax import Array
from distrax import Distribution

from psoc.core import TransitionModel, RewardFn


class MDPEnv(NamedTuple):
    num_envs: int
    state_dim: int
    action_dim: int
    num_time_steps: int
    prior_dist: Distribution
    trans_model: TransitionModel
    reward_fn: RewardFn
    feature_fn: Callable
