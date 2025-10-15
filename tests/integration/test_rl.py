import pytest
import torch
import gymnasium as gym
import numpy as np
from collections import deque

from sbb.paradigms.policy_gradient import ReinforcementLearningConfig
from sbb.const import DEVICE
from src.sbb.paradigms.policy_gradient import (
    DiscretePolicyGradientAgent,
    ContinuousPolicyGradientAgent,
    GAEReturnEstimator,
    TD0ReturnEstimator,
    MonteCarloReturnEstimator,
    RLTrajectoryBuffer,
)


def run_rl_training_loop(env, agent, total_timesteps, rollout_steps):
    """
    Minimal training loop for a reinforcement learning agent.
    Returns a list of episode returns (can be empty if no episodes ended).
    """
    agent.train()
    batch_size = agent.cfg.batch_size
    obs, _ = env.reset(seed=agent.cfg.seed)
    state_tuple = agent.recurrent_policy_network.new_state(batch_size)

    all_episode_rewards = []
    episode_rewards = deque(maxlen=100)
    current_episode_reward = 0.0
    global_step = 0

    while global_step < total_timesteps:

        buffer_shape = (rollout_steps + 1, batch_size, agent.cfg.total_neurons)
        states = torch.zeros(buffer_shape, device=DEVICE, dtype=agent.dtype)
        eligibility_traces = torch.zeros_like(states)
        input_projections = torch.zeros_like(states)
        activity_traces = torch.zeros_like(states)
        activity_biases = torch.zeros_like(states)

        if isinstance(agent, DiscretePolicyGradientAgent):
            action_shape = ()
            action_dtype = torch.long
        else:
            action_shape = (agent.policy_head.action_dim,)
            action_dtype = agent.dtype

        actions = torch.zeros(
            (rollout_steps, batch_size, *action_shape),
            device=DEVICE,
            dtype=action_dtype,
        )
        rewards = torch.zeros(
            (rollout_steps, batch_size), device=DEVICE, dtype=agent.dtype
        )
        values = torch.zeros(
            (rollout_steps + 1, batch_size), device=DEVICE, dtype=agent.dtype
        )
        log_probs = torch.zeros(
            (rollout_steps, batch_size), device=DEVICE, dtype=agent.dtype
        )
        dones = torch.zeros(
            (rollout_steps, batch_size), device=DEVICE, dtype=torch.bool
        )

        for step in range(rollout_steps):
            global_step += 1
            obs_tensor = torch.tensor(obs, device=DEVICE, dtype=agent.dtype)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            states[step] = state_tuple.activations
            eligibility_traces[step] = state_tuple.eligibility_trace
            input_projections[step] = state_tuple.input_projection
            activity_traces[step] = state_tuple.homeostatic_trace
            activity_biases[step] = state_tuple.bias

            with torch.no_grad():
                action, log_prob, value = agent.act(state_tuple)

            if isinstance(agent, DiscretePolicyGradientAgent):
                action_np = action.cpu().numpy()
            else:
                action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np[0])
            done = terminated or truncated

            actions[step] = action
            rewards[step] = torch.tensor([reward], device=DEVICE)
            values[step] = value
            log_probs[step] = log_prob
            dones[step] = torch.tensor([done], device=DEVICE)

            with torch.no_grad():
                next_obs_tensor = torch.tensor(
                    next_obs, device=DEVICE, dtype=agent.dtype
                ).unsqueeze(0)
                next_state_tuple, _ = agent.forward(
                    next_obs_tensor.unsqueeze(0), state_tuple
                )

            obs = next_obs
            state_tuple = next_state_tuple
            current_episode_reward += reward

            if done:
                episode_rewards.append(current_episode_reward)
                all_episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, _ = env.reset()
                state_tuple = agent.recurrent_policy_network.new_state(batch_size)

        with torch.no_grad():
            _, _, final_value = agent.act(state_tuple)
            states[-1] = state_tuple.activations
            values[-1] = final_value
            eligibility_traces[-1] = state_tuple.eligibility_trace
            input_projections[-1] = state_tuple.input_projection
            activity_traces[-1] = state_tuple.homeostatic_trace
            activity_biases[-1] = state_tuple.bias

        buffer = RLTrajectoryBuffer(
            system_states=states,
            actions=actions,
            rewards=rewards,
            state_value_estimates=values,
            action_log_probabilities=log_probs,
            terminations=dones,
            eligibility_traces=eligibility_traces,
            projected_fields=input_projections,
            homeostatic_traces=activity_traces,
            biases=activity_biases,
        )

        agent.backward(buffer)

    return all_episode_rewards


def _make_estimator(kind: str, dtype: torch.dtype):
    kind = kind.lower()
    if kind == "td0":
        return TD0ReturnEstimator(gamma=0.99, dtype=dtype)
    elif kind == "gae":
        return GAEReturnEstimator(gamma=0.99, gae_lambda=0.95, dtype=dtype)
    elif kind == "mc":
        return MonteCarloReturnEstimator(gamma=0.99, dtype=dtype)
    else:
        raise ValueError(f"Unknown estimator kind: {kind}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("agent_kind", ["discrete", "continuous"])
@pytest.mark.parametrize("estimator_kind", ["td0", "gae", "mc"])
def test_lunar_lander_smoke_combinations(agent_kind, estimator_kind):
    """
    Smoke tests: verify each combination of
      (discrete|continuous) x (TD0|GAE|MC)
    runs without numerical blow-ups or exceptions, and (if any episodes ended)
    yields finite rewards. No scoring assertions.
    """
    env_name = (
        "LunarLander-v3" if agent_kind == "discrete" else "LunarLanderContinuous-v3"
    )
    env = gym.make(env_name)

    dtype = torch.float32

    cfg = ReinforcementLearningConfig(
        num_blocks=32,
        neurons_per_block=32,
        input_features=env.observation_space.shape[0],
        output_features=(
            env.action_space.n
            if agent_kind == "discrete"
            else env.action_space.shape[0]
        ),
        batch_size=1,
        dtype=dtype,
        seed=42,
        target_connectivity=2,
    )

    estimator = _make_estimator(estimator_kind, dtype=dtype)

    if agent_kind == "discrete":
        agent = DiscretePolicyGradientAgent(cfg=cfg, estimator=estimator)
    else:
        agent = ContinuousPolicyGradientAgent(cfg=cfg, estimator=estimator)
    total_timesteps = 1024

    rollout_steps = 100

    rewards = run_rl_training_loop(env, agent, total_timesteps, rollout_steps)
    env.close()

    assert isinstance(rewards, list)
    if len(rewards) > 0:
        arr = np.array(rewards, dtype=float)
        assert np.isfinite(
            arr
        ).all(), f"Non-finite episode reward encountered for ({agent_kind}, {estimator_kind})"
