import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch.distributions import Normal, Independent
import argparse
from torch.utils.tensorboard import SummaryWriter
import os

from sbb import ActiveInferenceHyperparameters
from sbb import ActiveInferenceAgent
from sbb.const import DEVICE
from vizu import TraceVisualizer


class PartiallyObservableGridWorld(gym.Env):
    """
    A simple grid world environment with partial observability.
    - The agent's goal is to navigate to a fixed target location.
    - The agent only observes its immediate 3x3 neighborhood.
    - The state is represented from the agent's perspective (agent is always at the center).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, grid_size=10, render_mode=None):
        self.size = grid_size
        self.window_size = 512

        self.observation_space = spaces.Box(
            low=0, high=3, shape=(3, 3), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):

        padded_grid = np.pad(
            self._grid, pad_width=1, mode="constant", constant_values=1
        )
        x, y = self._agent_location

        px, py = x + 1, y + 1

        return padded_grid[px - 1 : px + 2, py - 1 : py + 2].astype(np.float32)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._grid = np.zeros((self.size, self.size), dtype=int)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._grid[tuple(self._target_location)] = 3
        self._grid[tuple(self._agent_location)] = 2

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        self._grid[tuple(self._agent_location)] = 0
        self._agent_location = new_location
        self._grid[tuple(self._agent_location)] = 2

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if terminated else -0.01
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            grid_repr = np.full((self.size, self.size), " ", dtype=str)
            grid_repr[self._grid == 1] = "#"
            grid_repr[self._grid == 3] = "G"
            grid_repr[tuple(self._agent_location)] = "A"
            return "\n".join("".join(row) for row in grid_repr)

    def _render_frame(self):
        if self.render_mode != "human":
            return

        import pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                self._target_location[1] * pix_square_size,
                self._target_location[0] * pix_square_size,
                pix_square_size,
                pix_square_size,
            ),
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (
                (self._agent_location[1] + 0.5) * pix_square_size,
                (self._agent_location[0] + 0.5) * pix_square_size,
            ),
            pix_square_size / 3,
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


def normalize_observation(obs: np.ndarray) -> np.ndarray:
    """Maps observation values from the environment's [0, 3] to the agent's [-1, 1] space."""
    return (obs - 1.5) / 1.5


def main():
    parser = argparse.ArgumentParser(
        description="Run Active Inference GridWorld Experiment"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to run."
    )
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument(
        "--visualize-traces",
        action="store_true",
        help="Enable live visualization of internal traces.",
    )
    parser.add_argument(
        "--scroll-window-size",
        type=int,
        default=100,
        help="Number of steps to show in the scrolling trace visualization.",
    )
    args = parser.parse_args()

    log_dir = os.path.expanduser("~/.tensorboard/runs")
    writer = SummaryWriter(log_dir=log_dir)

    render_mode = "human" if args.render else None
    env = PartiallyObservableGridWorld(grid_size=10, render_mode=render_mode)

    obs_shape = env.observation_space.shape
    flat_obs_dim = int(np.prod(obs_shape))
    action_dim = env.action_space.n

    cfg = ActiveInferenceHyperparameters(
        prediction_horizon=15,
        num_stochastic_policies=4,
        epistemic_weight=0.0,
        input_features=action_dim,
        output_features=flat_obs_dim,
        num_blocks=16,
        neurons_per_block=16,
        max_norm=1.0,
        delta_max_norm=1.0,
        dtype=torch.float32,
        seed=42,
    )
    agent = ActiveInferenceAgent(cfg)
    agent.train()

    trace_visualizer = (
        TraceVisualizer(
            num_neurons=cfg.total_neurons,
            scroll_window_size=args.scroll_window_size,
            title="AANS Live Traces (AIF GridWorld)",
            trace2_name="Surprise Feedback",
        )
        if args.visualize_traces
        else None
    )

    goal_obs = np.zeros(obs_shape, dtype=np.float32)
    goal_obs[1, 1] = 3
    normalized_goal_obs = normalize_observation(goal_obs)
    goal_obs_flat = torch.tensor(
        normalized_goal_obs.flatten(), device=DEVICE, dtype=cfg.dtype
    )
    base_distribution = Normal(loc=goal_obs_flat, scale=0.5)
    goal_distribution = Independent(base_distribution, 1)

    global_step = 0
    for episode in range(args.episodes):
        obs, info = env.reset()
        state_tuple = agent.base.new_state(batch_size=1)
        done = False
        total_reward = 0
        steps = 0

        episode_surprise = []
        episode_efe = []
        episode_instrumental = []
        episode_epistemic = []

        while not done:

            action_tensor, planning_diags = agent.forward(
                state_tuple, goal_distribution
            )
            action = torch.argmax(action_tensor).item()

            episode_efe.append(planning_diags["efe"])
            episode_instrumental.append(planning_diags["instrumental"])
            episode_epistemic.append(planning_diags["epistemic"])
            writer.add_scalar("Diagnostics/EFE", planning_diags["efe"], global_step)
            writer.add_scalar(
                "Diagnostics/Instrumental_Value",
                planning_diags["instrumental"],
                global_step,
            )
            writer.add_scalar(
                "Diagnostics/Epistemic_Value", planning_diags["epistemic"], global_step
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            normalized_next_obs = normalize_observation(next_obs)
            obs_tensor = torch.tensor(
                normalized_next_obs.flatten(), device=DEVICE, dtype=cfg.dtype
            ).unsqueeze(0)
            action_one_hot = torch.nn.functional.one_hot(
                torch.tensor([action]), num_classes=action_dim
            ).to(DEVICE, cfg.dtype)

            state_tuple, update_diags = agent.backward(
                state_tuple, action_one_hot, obs_tensor
            )
            episode_surprise.append(update_diags["surprise_mse"])
            writer.add_scalar(
                "Diagnostics/Surprise_MSE", update_diags["surprise_mse"], global_step
            )

            if trace_visualizer:
                with torch.no_grad():

                    obs_pred = torch.tanh(state_tuple.activations @ agent.readout.T)
                    surprise = obs_pred - obs_tensor
                    surprise_feedback = surprise @ agent.readout

                    eligibility_trace = state_tuple.eligibility_trace.squeeze(0)

                    trace_visualizer.update(
                        eligibility_trace.cpu().numpy(),
                        surprise_feedback.squeeze(0).cpu().numpy(),
                    )

            global_step += 1
            if steps > 100:
                break

        avg_surprise = np.mean(episode_surprise) if episode_surprise else 0
        avg_efe = np.mean(episode_efe) if episode_efe else 0
        avg_epistemic = np.mean(episode_epistemic) if episode_epistemic else 0

        writer.add_scalar("Episode/Total_Reward", total_reward, episode)
        writer.add_scalar("Episode/Steps", steps, episode)
        writer.add_scalar("Episode/Average_Surprise", avg_surprise, episode)
        writer.add_scalar("Episode/Average_EFE", avg_efe, episode)
        writer.add_scalar("Episode/Average_Epistemic", avg_epistemic, episode)

        print(
            f"Episode {episode + 1}/{args.episodes}: Steps={steps}, Reward={total_reward:.2f}, "
            f"AvgSurprise={avg_surprise:.4f}, AvgEFE={avg_efe:.2f}, AvgEpistemic={avg_epistemic:.2f}"
        )

    writer.close()
    env.close()
    if trace_visualizer:
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "aif_gridworld_trace_spectrogram.png")
        trace_visualizer.save_figure(save_path)
        trace_visualizer.close()
    print("Experiment finished. View logs with: tensorboard --logdir=runs")


if __name__ == "__main__":
    main()
