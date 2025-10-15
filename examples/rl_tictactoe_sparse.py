"""
Sparse reward RL test: TicTacToe vs random opponent with Monte Carlo returns.

This tests whether the modulatory term helps with credit assignment when:
- Rewards are extremely sparse (only at episode end)
- No bootstrapping (pure Monte Carlo returns)
- Simple opponent (random play)

Perfect for comparing modulatory_term enabled vs disabled.
"""

import os
import torch
import numpy as np
from collections import deque
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.sbb.paradigms.policy_gradient import (
    ReinforcementLearningConfig,
    DiscretePolicyGradientAgent,
    RLTrajectoryBuffer,
    MonteCarloReturnEstimator,
)
from sbb.const import DEVICE
from examples.tictactoe_env import TicTacToeEnv


def get_agent_config(dtype):
    """Small network for fast training."""
    return ReinforcementLearningConfig(
        input_features=9,
        output_features=9,
        seed=42,
        dtype=dtype,
        batch_size=1,
        num_blocks=32,
        neurons_per_block=32,
        target_connectivity=2,
        evolution_substeps=2,
        tau_fast=0.02,
        tau_eligibility=0.2,
        noise=0.01,
        value_lr=0.01,
        eligibility_rank=16,
        policy_lr=1.0,
    )


def random_policy(legal_actions):
    """Random opponent."""
    return np.random.choice(legal_actions)


def play_episode(agent, env, dtype, train=True):
    """
    Play one episode against random opponent.
    Returns trajectory and outcome (+1 win, -1 loss, 0 draw).
    """
    trajectory = []
    state = env.reset()

    # Agent is always player 1
    done = False
    while not done:
        if env.current_player == 1:
            # Agent's turn
            state_tensor = torch.tensor(state, device=DEVICE, dtype=dtype).unsqueeze(0)
            with torch.no_grad():
                state_tuple, _ = agent.recurrent_policy_network.forward(
                    state_tensor.unsqueeze(0),
                    agent.recurrent_policy_network.new_state(1),
                )

            legal_actions = env.get_legal_actions()
            legal_mask = torch.zeros(9, dtype=torch.bool, device=DEVICE)
            legal_mask[legal_actions] = True

            action_tensor, log_prob, value = agent.act(state_tuple, legal_mask)
            action = action_tensor.item()

            next_state, reward, done = env.step(action)

            # Store transition (reward is 0 until terminal)
            trajectory.append(
                {
                    "state": state_tuple.activations.squeeze(0),
                    "eligibility": state_tuple.eligibility_trace.squeeze(0),
                    "activity": state_tuple.homeostatic_trace.squeeze(0),
                    "projection": state_tuple.input_projection.squeeze(0),
                    "action": action_tensor,
                    "reward": reward,
                    "done": done,
                    "log_prob": log_prob,
                    "value": value,
                }
            )

            state = next_state
        else:
            # Random opponent's turn
            legal_actions = env.get_legal_actions()
            action = random_policy(legal_actions)
            state, reward, done = env.step(action)

            # If opponent wins, agent gets -1 reward on last transition
            if done and reward != 0:
                if trajectory:
                    trajectory[-1]["reward"] = -reward

    # Return final outcome from agent's perspective
    final_reward = trajectory[-1]["reward"] if trajectory else 0
    return trajectory, final_reward


def train_step(agent, trajectory, dtype):
    """Train on a single episode trajectory."""
    if not trajectory:
        return 0.0

    states = torch.stack([t["state"] for t in trajectory])
    actions = torch.stack([t["action"] for t in trajectory])
    rewards = torch.tensor(
        [t["reward"] for t in trajectory], device=DEVICE, dtype=dtype
    )
    dones = torch.tensor(
        [t["done"] for t in trajectory], device=DEVICE, dtype=torch.bool
    )
    log_probs = torch.stack([t["log_prob"] for t in trajectory])
    values = torch.stack([t["value"] for t in trajectory])
    eligibility = torch.stack([t["eligibility"] for t in trajectory])
    projections = torch.stack([t["projection"] for t in trajectory])
    activity = torch.stack([t["activity"] for t in trajectory])

    # Add dummy next state for bootstrap
    dummy_next = torch.zeros_like(states[0:1])
    dummy_value = torch.zeros(1, device=DEVICE, dtype=dtype)

    states_with_next = torch.cat([states, dummy_next], dim=0)
    values_with_bootstrap = torch.cat([values.squeeze(-1), dummy_value], dim=0)

    dummy_elig = torch.zeros_like(eligibility[0:1])
    dummy_proj = torch.zeros_like(projections[0:1])
    dummy_act = torch.zeros_like(activity[0:1])

    buffer = RLTrajectoryBuffer(
        system_states=states_with_next.unsqueeze(1),
        actions=actions.unsqueeze(1),
        rewards=rewards.unsqueeze(1),
        terminations=dones.unsqueeze(1),
        action_log_probabilities=log_probs.squeeze(-1).unsqueeze(1),
        state_value_estimates=values_with_bootstrap.unsqueeze(1),
        eligibility_traces=torch.cat([eligibility, dummy_elig], dim=0).unsqueeze(1),
        projected_fields=torch.cat([projections, dummy_proj], dim=0).unsqueeze(1),
        homeostatic_traces=torch.cat([activity, dummy_act], dim=0).unsqueeze(1),
        biases=torch.zeros_like(states_with_next.unsqueeze(1)),
    )

    loss = agent.learn(buffer)
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Sparse reward TicTacToe vs random")
    parser.add_argument(
        "--episodes", type=int, default=10000, help="Number of episodes"
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Run name for tensorboard"
    )
    args = parser.parse_args()

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    torch.manual_seed(42)
    np.random.seed(42)

    cfg = get_agent_config(dtype)
    estimator = MonteCarloReturnEstimator(gamma=0.99, dtype=dtype)
    agent = DiscretePolicyGradientAgent(cfg, estimator).to(DEVICE, dtype)

    env = TicTacToeEnv()

    # Tensorboard logging
    run_name = (
        args.run_name
        if args.run_name
        else f"tictactoe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir = os.path.expanduser(
        f"~/.tensorboard/self-building-box/AANS/ttt-{run_name}-AANS"
    )

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Track statistics
    recent_rewards: deque[float] = deque(maxlen=100)
    wins, losses, draws = 0, 0, 0

    print(f"Training agent on sparse reward TicTacToe (device={DEVICE}, dtype={dtype})")
    print(f"Tensorboard logs: runs/{run_name}")
    print(
        "Reward structure: +1 for win, -1 for loss, 0 for draw (only at episode end)\n"
    )

    for episode in range(1, args.episodes + 1):
        # Pure online learning - always in train mode, learn from every episode
        agent.train()
        trajectory, outcome = play_episode(agent, env, dtype, train=True)
        loss = train_step(agent, trajectory, dtype)

        recent_rewards.append(outcome)
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1

        # Log to tensorboard every episode
        writer.add_scalar("train/episode_reward", outcome, episode)
        writer.add_scalar("train/value_loss", loss, episode)
        writer.add_scalar("train/episode_length", len(trajectory), episode)

        if episode % 100 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            win_rate_100 = (
                sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                if recent_rewards
                else 0
            )
            writer.add_scalar("train/avg_reward_100", avg_reward, episode)
            writer.add_scalar("train/win_rate_100", win_rate_100, episode)

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
