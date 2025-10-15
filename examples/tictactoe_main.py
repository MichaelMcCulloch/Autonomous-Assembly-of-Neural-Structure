import torch
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import argparse

from src.sbb.paradigms.policy_gradient import (
    ReinforcementLearningConfig,
    DiscretePolicyGradientAgent,
    RLTrajectoryBuffer,
    GAEReturnEstimator,
)
from sbb.const import DEVICE
from examples.tictactoe_env import TicTacToeEnv
from examples.player_pool import PlayerPool

try:
    from examples.vizu import AANSVisualizer
except ImportError:
    AANSVisualizer = None  # type: ignore[assignment,misc]


@dataclass
class TicTacToeConfig:
    input_features: int = 9
    output_features: int = 9

    generations: int = 100
    games_per_generation: int = 256
    eval_games: int = 50

    seed: int = 42
    dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    gamma: float = 0.99
    gae_lambda: float = 0.95


def get_agent_config(cfg: TicTacToeConfig) -> ReinforcementLearningConfig:
    return ReinforcementLearningConfig(
        input_features=cfg.input_features,
        output_features=cfg.output_features,
        seed=cfg.seed,
        dtype=cfg.dtype,
        batch_size=1,
        num_blocks=16,
        neurons_per_block=16,
        target_connectivity=2,
        evolution_substeps=2,
        tau_fast=0.02,
        tau_eligibility=0.2,
        noise=0.01,
    )


def run_self_play_game(agent, opponent, env, dtype):
    """Runs a single game of self-play and returns the trajectory."""
    game_trajectory = []
    state = env.reset()

    players = {1: agent, -1: opponent}
    if np.random.rand() > 0.5:
        players = {-1: agent, 1: opponent}

    done = False
    while not done:
        current_player_id = env.current_player
        current_player_agent = players[current_player_id]

        state_tensor = torch.tensor(state, device=DEVICE, dtype=dtype).unsqueeze(0)
        with torch.no_grad():
            state_tuple, _ = current_player_agent.recurrent_policy_network.forward(
                state_tensor.unsqueeze(0),
                current_player_agent.recurrent_policy_network.new_state(1),
            )

        legal_actions = env.get_legal_actions()
        legal_actions_mask = torch.zeros(
            agent.cfg.output_features, dtype=torch.bool, device=DEVICE
        )
        legal_actions_mask[legal_actions] = True

        action_tensor, log_prob, value = current_player_agent.act(
            state_tuple, legal_actions_mask
        )
        action = action_tensor.item()

        next_state, reward, done = env.step(action)

        game_trajectory.append(
            {
                "internal_state": state_tuple.activations.squeeze(0),
                "eligibility_trace": state_tuple.eligibility_trace.squeeze(0),
                "activity_trace": state_tuple.homeostatic_trace.squeeze(0),
                "input_projection": state_tuple.input_projection.squeeze(0),
                "action": action_tensor,
                "reward": reward,
                "done": done,
                "log_prob": log_prob,
                "value": value,
                "player_id": current_player_id,
            }
        )
        state = next_state

    # Assign rewards to both players
    final_reward = game_trajectory[-1]["reward"]
    if len(game_trajectory) > 1 and final_reward != 0:
        game_trajectory[-2]["reward"] = -final_reward

    return game_trajectory


def main():
    parser = argparse.ArgumentParser(description="Train AANS on Tic-Tac-Toe")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable live visualization of the AANS adjacency matrix.",
    )
    args = parser.parse_args()

    cfg = TicTacToeConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    agent_config = get_agent_config(cfg)
    estimator = GAEReturnEstimator(
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        dtype=cfg.dtype,
    )
    agent = DiscretePolicyGradientAgent(agent_config, estimator).to(DEVICE, cfg.dtype)

    visualizer = None
    if args.visualize and AANSVisualizer is not None:
        visualizer = AANSVisualizer(agent.recurrent_policy_network)

    opponent = DiscretePolicyGradientAgent(agent_config, estimator).to(
        DEVICE, cfg.dtype
    )
    env = TicTacToeEnv()
    player_pool = PlayerPool("tictactoe_pool")
    player_pool.add(agent.state_dict(), 0)

    for generation in range(1, cfg.generations + 1):
        print(f"\n--- Generation {generation}/{cfg.generations} ---")

        agent.train()
        all_trajectories = []
        print("Running self-play games...")
        for _ in tqdm(range(cfg.games_per_generation)):
            opponent_state_dict = player_pool.get_opponent_state_dict()
            opponent.load_state_dict(opponent_state_dict)
            opponent.eval()

            trajectory = run_self_play_game(agent, opponent, env, cfg.dtype)
            agent_player_id = [
                pid for pid, p_agent in [(1, agent), (-1, opponent)] if p_agent is agent
            ][0]
            agent_trajectory = [
                t for t in trajectory if t["player_id"] == agent_player_id
            ]
            all_trajectories.extend(agent_trajectory)

        if not all_trajectories:
            print("No training data collected for the agent this generation. Skipping.")
            continue

        internal_states = torch.stack([t["internal_state"] for t in all_trajectories])
        actions = torch.stack([t["action"] for t in all_trajectories])
        rewards = torch.tensor(
            [t["reward"] for t in all_trajectories], device=DEVICE, dtype=cfg.dtype
        )
        dones = torch.tensor(
            [t["done"] for t in all_trajectories], device=DEVICE, dtype=torch.bool
        )
        log_probs = torch.stack([t["log_prob"] for t in all_trajectories])
        values = torch.stack([t["value"] for t in all_trajectories])
        eligibility_traces = torch.stack(
            [t["eligibility_trace"] for t in all_trajectories]
        )
        input_projections = torch.stack(
            [t["input_projection"] for t in all_trajectories]
        )
        activity_traces = torch.stack([t["activity_trace"] for t in all_trajectories])

        # Append bootstrap state and value (zeros for terminal states)
        dummy_next_state = torch.zeros_like(internal_states[0:1])
        dummy_next_value = torch.zeros(1, device=DEVICE, dtype=cfg.dtype)

        # internal_states: [T] -> [T+1], values: [T] -> [T+1]
        states_with_next = torch.cat([internal_states, dummy_next_state], dim=0)
        values_with_bootstrap = torch.cat([values.squeeze(-1), dummy_next_value], dim=0)

        # Append dummy next state/value for bootstrap
        dummy_next_elig = torch.zeros_like(eligibility_traces[0:1])
        dummy_next_proj = torch.zeros_like(input_projections[0:1])
        dummy_next_act = torch.zeros_like(activity_traces[0:1])

        buffer = RLTrajectoryBuffer(
            system_states=states_with_next.unsqueeze(1),
            actions=actions.unsqueeze(1),
            rewards=rewards.unsqueeze(1),
            terminations=dones.unsqueeze(1),
            action_log_probabilities=log_probs.squeeze(-1).unsqueeze(1),
            state_value_estimates=values_with_bootstrap.unsqueeze(1),
            eligibility_traces=torch.cat(
                [eligibility_traces, dummy_next_elig], dim=0
            ).unsqueeze(1),
            projected_fields=torch.cat(
                [input_projections, dummy_next_proj], dim=0
            ).unsqueeze(1),
            homeostatic_traces=torch.cat(
                [activity_traces, dummy_next_act], dim=0
            ).unsqueeze(1),
            biases=torch.zeros_like(states_with_next.unsqueeze(1)),
        )

        print("Updating agent...")
        loss = agent.backward(buffer)
        print(f"Value Loss: {loss.item():.4f}")

        if visualizer:
            visualizer.update_adjacency_matrix(f"Generation: {generation}")

        player_pool.add(agent.state_dict(), generation)

        # Evaluation
        agent.eval()
        wins, losses, draws = 0, 0, 0
        for _ in range(cfg.eval_games):
            opponent_state_dict = player_pool.get_opponent_state_dict()
            opponent.load_state_dict(opponent_state_dict)
            opponent.eval()

            state = env.reset()
            done = False
            while not done:
                if env.current_player == 1:
                    state_tensor = torch.tensor(
                        state, device=DEVICE, dtype=cfg.dtype
                    ).unsqueeze(0)
                    with torch.no_grad():
                        state_tuple, _ = agent.recurrent_policy_network.forward(
                            state_tensor.unsqueeze(0),
                            agent.recurrent_policy_network.new_state(1),
                        )
                    legal_actions_mask = torch.zeros(9, dtype=torch.bool, device=DEVICE)
                    legal_actions_mask[env.get_legal_actions()] = True
                    action_tensor, _, _ = agent.act(state_tuple, legal_actions_mask)
                    action = action_tensor.item()
                else:
                    state_tensor = torch.tensor(
                        state, device=DEVICE, dtype=cfg.dtype
                    ).unsqueeze(0)
                    with torch.no_grad():
                        state_tuple, _ = opponent.recurrent_policy_network.forward(
                            state_tensor.unsqueeze(0),
                            opponent.recurrent_policy_network.new_state(1),
                        )
                    legal_actions_mask = torch.zeros(9, dtype=torch.bool, device=DEVICE)
                    legal_actions_mask[env.get_legal_actions()] = True
                    action_tensor, _, _ = opponent.act(state_tuple, legal_actions_mask)
                    action = int(action_tensor.item())

                state, reward, done = env.step(action)

            if reward == 1.0:
                wins += 1
            elif reward == -1.0:
                losses += 1
            else:
                draws += 1

        print(f"Evaluation vs Pool: Wins={wins}, Losses={losses}, Draws={draws}")

    if visualizer:
        visualizer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
