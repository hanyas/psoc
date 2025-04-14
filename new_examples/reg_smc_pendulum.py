# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax
jax.config.update("jax_enable_x64", True)

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

from psoc.smc import regularized_smc, backward_tracing

from psoc.bijector import Tanh
from psoc.new_utils import (
    batch_data,
    damping_schedule,
    policy_evaluation,
    flatten_particle_trajectories,
)
from psoc.policy import (
    NeuralGaussPolicy,
    NeuralGaussTransition,
    create_neural_gauss_policy,
    create_neural_gauss_transition,
    train_neural_gauss_policy_stepwise,
    train_neural_gauss_transition_stepwise
)

import time
import matplotlib.pyplot as plt

from psoc.envs import PendulumEnv as env


rng_key = random.PRNGKey(17)

num_particles = 64

slew_rate_penalty = 0.005
tempering = 1.0
init_damping = 0.1

learning_rate = 3e-4
batch_size = 16
num_epochs = 150

policy_bijector = Block(Tanh(), ndims=1)

policy_prior_network = NeuralGaussPolicy(
    feature_fn=env.feature_fn,
    hidden_size=(256, 256),
    action_dim=env.action_dim,
    init_log_std=constant(jnp.log(2.0))
)
policy_prior = create_neural_gauss_policy(
    network=policy_prior_network,
    bijector=policy_bijector
)
key, sub_key = random.split(rng_key, 2)
policy_prior_state = policy_prior.init(
    rng_key=sub_key,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    batch_dim=num_particles,
    learning_rate=1e-4
)

###
policy_proposal_network = NeuralGaussPolicy(
    feature_fn=env.feature_fn,
    hidden_size=(256, 256),
    action_dim=env.action_dim,
    init_log_std=constant(jnp.log(2.0)),
)
policy_proposal = create_neural_gauss_policy(
    network=policy_proposal_network,
    bijector=policy_bijector
)

key, sub_key = random.split(key, 2)
policy_proposal_state = policy_proposal.init(
    rng_key=sub_key,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    batch_dim=num_particles,
    learning_rate=learning_rate
)

###
trans_network = NeuralGaussTransition(
    hidden_size=(256, 256),
    state_dim=env.state_dim,
    init_log_std=constant(jnp.log(1.0)),
)
trans_proposal = create_neural_gauss_transition(
    network=trans_network
)

key, sub_key = random.split(key, 2)
trans_state = trans_proposal.init(
    rng_key=sub_key,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    batch_dim=num_particles,
    learning_rate=3e-4
)


num_steps = 0

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    # update damping param
    damping = damping_schedule(
        step=i,
        total_steps=num_epochs,
        steepness=1.,
        init_value=init_damping,
        max_value=1.
    )

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    expected_reward, *_ = policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy_proposal,
        params=policy_proposal_state.params,
        num_samples=1024
    )

    # run nested smc
    key, sub_key = random.split(key)
    smc_states, log_marginal = \
        regularized_smc(
            rng_key=sub_key,
            num_time_steps=env.num_time_steps,
            num_particles=num_particles,
            init_prior=env.prior_dist,
            trans_prior=env.trans_model,
            trans_proposal=trans_proposal,
            trans_proposal_params=trans_state.params,
            policy_prior=policy_prior,
            policy_prior_params=policy_prior_state.params,
            policy_proposal=policy_proposal,
            policy_proposal_params=policy_proposal_state.params,
            reward_fn=env.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
            damping=damping,
        )

    num_steps += (env.num_time_steps + 1) * num_particles

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_particles = backward_tracing(sub_key, smc_states)

    # update policy parameters
    next_states, states, actions = flatten_particle_trajectories(traced_particles)
    data_size, _ = states.shape

    policy_loss = 0.0
    trans_loss = 0.0

    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, data_size, batch_size)
    for batch_idx in batch_indices:
        next_state_batch = next_states[batch_idx, ...]
        state_batch = states[batch_idx, ...]
        action_batch = actions[batch_idx, ...]

        policy_prior_state, _ = train_neural_gauss_policy_stepwise(
            policy=policy_prior,
            train_state=policy_prior_state,
            actions=action_batch,
            states=state_batch,
            damping=(1. - damping)
        )
        policy_proposal_state, _policy_loss = train_neural_gauss_policy_stepwise(
            policy=policy_proposal,
            train_state=policy_proposal_state,
            actions=action_batch,
            states=state_batch,
            damping=damping
        )
        trans_state, _trans_loss = train_neural_gauss_transition_stepwise(
            transition=trans_proposal,
            train_state=trans_state,
            next_states=next_state_batch,
            states=state_batch,
            actions=action_batch,
            damping=damping
        )

        policy_loss += _policy_loss
        trans_loss += _trans_loss

    entropy = policy_proposal.entropy(policy_proposal_state.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Num steps: {num_steps:6d}, "
        f"Log marginal: {log_marginal:.3f}, "
        f"Reward: {expected_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Damping: {damping:.3f}, "
        f"Policy loss: {policy_loss:.3f}, "
        f"Trans loss: {trans_loss:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

key, sub_key = random.split(key)
expected_reward, states, actions = \
    policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy_proposal,
        params=policy_proposal_state.params,
        num_samples=16
    )

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("Simulated trajectories")

axs[0].plot(states[..., 0])
axs[0].set_ylabel("Angle")
axs[0].grid(True)

axs[1].plot(states[..., 1])
axs[1].set_ylabel("Angular Velocity")
axs[1].grid(True)

axs[2].plot(actions[..., 0])
axs[2].set_ylabel("Actions")
axs[2].grid(True)

plt.tight_layout()
plt.show()
