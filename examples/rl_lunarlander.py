import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque

from src.sbb.const import DEVICE
from src.sbb.paradigms import (
    ContinuousPolicyGradientAgent,
    DiscretePolicyGradientAgent,
    RLTrajectoryBuffer,
    TD0ReturnEstimator,
    ReinforcementLearningConfig,
)
from src.sbb.util import print_config
from .vizu import AANSVisualizer, TraceVisualizer


def get_agent_config(
    dtype,
    batch_size,
    num_blocks=32,
    value_lr=0.01,
    policy_lr=1.0,
):
    """Creates a PolicyGradientHyperparameters object with defaults for LunarLander.

    Note: The automatic scaling laws tend to create networks that are too dense
    for this RL task. We use sparse connectivity (target_connectivity=2) regardless
    of network size, which empirically works better for online TD learning.
    """
    cfg = ReinforcementLearningConfig(
        num_blocks=num_blocks,
        neurons_per_block=32,
        batch_size=batch_size,
        input_features=8,
        output_features=4,
        dtype=dtype,
        seed=0,
        value_lr=value_lr,
        policy_lr=policy_lr,
    )
    # Manually add the learning rates that were missing, using REPO A's scaling
    BASE_LR_COEFF = 1e-2
    BASE_N_FOR_LR_SCALING = 1024
    LR_SCALING_ALPHA = 0.5
    cfg.modulator_lr = (
        BASE_LR_COEFF
        * (BASE_N_FOR_LR_SCALING / (cfg.total_neurons + 1e-12)) ** LR_SCALING_ALPHA
    )
    cfg.hebbian_lr = cfg.modulator_lr * 1e-3
    cfg.oja_lr = cfg.modulator_lr * 1e-3
    return cfg


def run_experiment(
    agent_type: str,
    n_episodes=500,
    visualize_episode=False,
    visualize_aans=False,
    visualize_traces=False,
    scroll_window_size=1000,
    num_blocks=32,
    value_lr=0.01,
    policy_lr=1.0,
    gamma=0.99,
    use_tensorboard=True,
):
    """
    Runs online RL experiment with per-timestep TD(0) learning and plasticity.

    Applies plasticity once per environment step, demonstrating AANS's
    online learning capability with local learning rules.
    """
    env_name = (
        "LunarLanderContinuous-v3" if agent_type == "continuous" else "LunarLander-v3"
    )
    render_mode = "human" if visualize_episode else None
    env = gym.make(env_name, render_mode=render_mode)

    dtype = torch.float32

    cfg = get_agent_config(
        dtype,
        batch_size=1,
        num_blocks=num_blocks,
        value_lr=value_lr,
        policy_lr=policy_lr,
    )
    estimator = TD0ReturnEstimator(gamma=gamma, dtype=dtype)

    cfg.input_features = env.observation_space.shape[0]
    cfg.output_features = (
        env.action_space.shape[0] if agent_type == "continuous" else env.action_space.n
    )

    print_config(cfg, f"Full Computed Configuration for {env_name} (Online)")

    AgentClass = (
        ContinuousPolicyGradientAgent
        if agent_type == "continuous"
        else DiscretePolicyGradientAgent
    )
    agent = AgentClass(cfg, estimator).to(DEVICE, cfg.dtype)
    visualizer = (
        AANSVisualizer(agent.recurrent_policy_network) if visualize_aans else None
    )
    trace_visualizer = (
        TraceVisualizer(
            num_neurons=cfg.total_neurons,
            scroll_window_size=scroll_window_size,
            title=f"AANS Live Traces ({env_name})",
            trace2_name="Advantage Signal",
        )
        if visualize_traces
        else None
    )

    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = os.path.expanduser(
            f"~/.tensorboard/self-building-box/AANS/{env_name}-AANS"
        )
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")
    else:
        writer = None

    all_episode_rewards: deque[float] = deque(maxlen=100)
    global_step = 0

    print(f"Running AANS-RL on {env_name} using device: {DEVICE}...")

    # Create initial state ONCE
    state_tuple = agent.recurrent_policy_network.new_state(batch_size=1)

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + episode)

        # Reset state but preserve noise for exploration diversity
        if episode > 0:
            with torch.no_grad():
                saved_noise = state_tuple.noise.clone()
                state_tuple = agent.recurrent_policy_network.new_state(batch_size=1)
                # Jump the RNG counter to ensure different noise each episode
                state_tuple.noise = saved_noise + 1000

        done = False
        total_reward = 0

        while not done:
            agent.train()
            obs_tensor = torch.tensor(obs, device=DEVICE, dtype=agent.dtype).unsqueeze(
                0
            )

            with torch.no_grad():
                # Single forward pass per step
                current_state_tuple, _ = agent.forward(
                    obs_tensor.unsqueeze(0), state_tuple
                )
                action, log_prob, value = agent.act(current_state_tuple)

            # Take environment step
            next_obs, reward, terminated, truncated, _ = env.step(
                action.cpu().numpy()[0]
            )
            done = terminated or truncated

            # Get next value for bootstrapping (only forward if not done)
            with torch.no_grad():
                if not done:
                    next_obs_tensor = torch.tensor(
                        next_obs, device=DEVICE, dtype=agent.dtype
                    ).unsqueeze(0)
                    next_state_tuple, _ = agent.forward(
                        next_obs_tensor.unsqueeze(0), current_state_tuple
                    )
                    _, _, next_value = agent.act(next_state_tuple)
                else:
                    # Terminal state: no next value
                    next_state_tuple = current_state_tuple  # Dummy for buffer
                    next_value = torch.zeros_like(value)

            # Create single-step trajectory buffer
            buffer = RLTrajectoryBuffer(
                system_states=torch.stack(
                    [
                        current_state_tuple.activations,
                        next_state_tuple.activations,
                    ]
                ),
                actions=action.unsqueeze(0),
                rewards=torch.tensor([[reward]], device=DEVICE, dtype=agent.dtype),
                terminations=torch.tensor([[done]], device=DEVICE, dtype=torch.bool),
                action_log_probabilities=log_prob.unsqueeze(0),
                state_value_estimates=torch.stack([value, next_value]),
                eligibility_traces=torch.stack(
                    [
                        current_state_tuple.eligibility_trace,
                        next_state_tuple.eligibility_trace,
                    ]
                ),
                projected_fields=torch.stack(
                    [
                        current_state_tuple.input_projection,
                        next_state_tuple.input_projection,
                    ]
                ),
                homeostatic_traces=torch.stack(
                    [
                        current_state_tuple.homeostatic_trace,
                        next_state_tuple.homeostatic_trace,
                    ]
                ),
                biases=torch.stack(
                    [
                        current_state_tuple.bias,
                        next_state_tuple.bias,
                    ]
                ),
            )

            loss = agent.backward(buffer)

            if trace_visualizer:
                with torch.no_grad():
                    advantages, _, _ = estimator.compute(buffer)  # type: ignore[misc]
                    advantage_signal = advantages.unsqueeze(2) * agent.value_head.weight
                    eligibility_trace = current_state_tuple.eligibility_trace.squeeze(0)
                    trace_visualizer.update(
                        eligibility_trace.cpu().numpy(),
                        advantage_signal.squeeze(0).squeeze(0).cpu().numpy(),
                    )

            # Update for next iteration
            obs = next_obs
            state_tuple = current_state_tuple  # KEY FIX: Use current, not next
            total_reward += reward
            global_step += 1

            # ... rest of logging code unchanged ...

            # Comprehensive logging - apples-to-apples with reference implementation
            if writer:
                with torch.no_grad():
                    # Loss metrics
                    writer.add_scalar("Loss/value_loss", loss.item(), global_step)
                    writer.add_scalar("Loss/rpe_squared", loss.item(), global_step)

                    # Reward metrics
                    writer.add_scalar("Reward/step_reward", reward, global_step)

                    # State metrics (use activations from current state_tuple)
                    activations = current_state_tuple.activations
                    writer.add_scalar(
                        "State/mean_x", activations.mean().item(), global_step
                    )

                    # Homeostatic bias metrics
                    if hasattr(current_state_tuple, "bias"):
                        writer.add_scalar(
                            "State/mean_bias",
                            current_state_tuple.bias.mean().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "State/var_bias",
                            current_state_tuple.bias.var().item(),
                            global_step,
                        )

                    # Agent/policy metrics - get logits from policy head
                    _, logits = agent.policy_head.forward(activations)
                    writer.add_scalar(
                        "Agent/actor_logits_mean",
                        logits.mean().item(),
                        global_step,
                    )

                    # Recurrent weight norms - block level
                    active_slots = (
                        agent.recurrent_policy_network.active_blocks.nonzero().squeeze(
                            -1
                        )
                    )
                    if active_slots.numel() > 0:
                        active_weights = agent.recurrent_policy_network.weight_values[
                            active_slots
                        ]
                        block_norms = torch.linalg.norm(
                            active_weights.flatten(start_dim=-2), dim=-1
                        )
                        writer.add_scalar(
                            "Norms/weight_recurrent_min",
                            block_norms.min().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "Norms/weight_recurrent_max",
                            block_norms.max().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "Norms/weight_recurrent_mean",
                            block_norms.mean().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "Norms/weight_recurrent_std",
                            block_norms.std().item(),
                            global_step,
                        )

                        # Overall recurrent weight norm
                        wrec_norm = torch.linalg.norm(
                            agent.recurrent_policy_network.weight_values
                        ).item()
                        writer.add_scalar("Weights/Wrec_norm", wrec_norm, global_step)

                    # Input projection norms
                    input_norm = torch.linalg.norm(
                        agent.recurrent_policy_network.weight_in
                    )
                    writer.add_scalar(
                        "Norms/weight_input_projection", input_norm.item(), global_step
                    )

                    # Output head norms
                    policy_norm = torch.linalg.norm(agent.policy_head.weight)
                    value_norm = torch.linalg.norm(agent.value_head.weight)
                    wout_norm = policy_norm.item() + value_norm.item()

                    writer.add_scalar(
                        "Norms/weight_policy_head", policy_norm.item(), global_step
                    )
                    writer.add_scalar(
                        "Norms/weight_value_head", value_norm.item(), global_step
                    )
                    writer.add_scalar("Weights/Wout_norm", wout_norm, global_step)

                    # Topology metrics
                    active_n = agent.recurrent_policy_network.active_blocks.sum().item()
                    neurons_per_block = (
                        agent.recurrent_policy_network.cfg.neurons_per_block
                    )
                    writer.add_scalar(
                        "Topology/active_connections",
                        active_n * (neurons_per_block**2),
                        global_step,
                    )
                    writer.add_scalar("Topology/active_blocks", active_n, global_step)

                    # Block gain statistics (if available)
                    if hasattr(agent.recurrent_policy_network, "block_gain"):
                        block_gain = agent.recurrent_policy_network.block_gain
                        writer.add_scalar(
                            "Weights/block_gain_max",
                            block_gain.max().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "Weights/block_gain_mean",
                            block_gain.mean().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "Weights/block_gain_min",
                            block_gain.min().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            "Weights/block_gain_std",
                            block_gain.std().item(),
                            global_step,
                        )

        all_episode_rewards.append(total_reward)
        avg_reward = np.mean(list(all_episode_rewards))

        # Log episode metrics using global_step for consistency
        if writer:
            writer.add_scalar("Reward/episode_reward", total_reward, global_step)
            writer.add_scalar("Reward/episode_length", global_step, global_step)

        # Log average reward every episode
        print(
            f"Episodes {episode+1}/{n_episodes}, Episode Reward {total_reward}, Avg Reward (last {len(all_episode_rewards)}): {avg_reward:.2f}, Active connections: {active_n}"
        )
        if writer:
            writer.add_scalar("Reward/avg_reward_100_ep", avg_reward, episode + 1)

        if visualizer:
            visualizer.update_adjacency_matrix(f"Episode: {episode+1}/{n_episodes}")

    env.close()
    if writer:
        writer.close()
    if visualizer:
        visualizer.close()
    if trace_visualizer:
        if use_tensorboard:
            save_path = os.path.join(log_dir, "trace_spectrogram.png")
            trace_visualizer.save_figure(save_path)
        trace_visualizer.close()
    print("Finished.")
    return list(all_episode_rewards)


def main():
    parser = argparse.ArgumentParser(
        description="Run AANS Reinforcement Learning Experiment"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["discrete", "continuous"],
        required=True,
        help="Type of action space.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--visualize-episode",
        action="store_true",
        help="Render the environment during training.",
    )
    parser.add_argument(
        "--visualize-aans",
        action="store_true",
        help="Enable live visualization of the AANS adjacency matrix.",
    )
    parser.add_argument(
        "--visualize-traces",
        action="store_true",
        help="Enable live visualization of internal traces.",
    )
    parser.add_argument(
        "--scroll-window-size",
        type=int,
        default=1000,
        help="Number of steps to show in the scrolling trace visualization.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=32,
        help="Number of AANS blocks (default: 32, which gives 32x32=1024 neurons).",
    )
    parser.add_argument(
        "--value-lr",
        type=float,
        default=0.01,
        help="Learning rate for the value head (default: 0.01).",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=1.0,
        help="Learning rate for the policy head (default: 1.0).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9999999,
        help="Discount factor for TD learning (default: 0.99).",
    )
    parser.add_argument(
        "--no-tb",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    args = parser.parse_args()

    print(f"Running AANS-RL ({args.agent_type}, online per-timestep learning)")

    results = run_experiment(
        agent_type=args.agent_type,
        n_episodes=args.episodes,
        visualize_episode=args.visualize_episode,
        visualize_aans=args.visualize_aans,
        visualize_traces=args.visualize_traces,
        scroll_window_size=args.scroll_window_size,
        num_blocks=args.num_blocks,
        value_lr=args.value_lr,
        policy_lr=args.policy_lr,
        gamma=args.gamma,
        use_tensorboard=not args.no_tb,
    )

    print(
        f"Experiment finished. Final average reward: {np.mean(results[-100:]) if results else 'N/A'}"
    )


if __name__ == "__main__":
    main()
