# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

from psoc.smc import smc, backward_tracing

from psoc.bijector import Tanh
from psoc.new_utils import (
    batch_data,
    policy_evaluation,
    flatten_particle_trajectories,
)
from psoc.policy import (
    NeuralGaussPolicy,
    create_neural_gauss_policy,
    train_neural_gauss_policy_stepwise,
)

import time
import matplotlib.pyplot as plt

from psoc.envs import PendulumEnv as env


rng_key = random.PRNGKey(7)

num_particles = 64

slew_rate_penalty = 0.005
tempering = 0.1

learning_rate = 3e-4
batch_size = 32
num_epochs = 150

bijector = Block(Tanh(), ndims=1)
network = NeuralGaussPolicy(
    feature_fn=env.feature_fn,
    hidden_size=(256, 256),
    action_dim=env.action_dim,
    init_log_std=constant(jnp.log(2.0)),
)
policy = create_neural_gauss_policy(network, bijector)

key, sub_key = random.split(rng_key, 2)
train_state = policy.init(
    rng_key=sub_key,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    batch_dim=num_particles,
    learning_rate=learning_rate
)

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    expected_reward, *_ = policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy,
        params=train_state.params,
        num_samples=1024
    )

    # run nested smc
    key, sub_key = random.split(key)
    smc_states, log_marginal = \
        smc(
            rng_key=sub_key,
            num_time_steps=env.num_time_steps,
            num_particles=num_particles,
            init_prior=env.prior_dist,
            trans_prior=env.trans_model,
            policy_prior=policy,
            policy_prior_params=train_state.params,
            reward_fn=env.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering
        )

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_particles = backward_tracing(sub_key, smc_states)

    # update policy parameters
    _, states, actions = flatten_particle_trajectories(traced_particles)
    data_size, _ = states.shape

    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, data_size, batch_size)
    for batch_idx in batch_indices:
        state_batch = states[batch_idx, ...]
        action_batch = actions[batch_idx, ...]

        train_state, batch_loss = train_neural_gauss_policy_stepwise(
            policy=policy,
            train_state=train_state,
            actions=action_batch,
            states=state_batch
        )
        loss += batch_loss

    entropy = policy.entropy(train_state.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Log marginal: {log_marginal:.3f}, "
        f"Reward: {expected_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )


key, sub_key = random.split(key)
expected_reward, states, actions = \
    policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy,
        params=train_state.params,
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
