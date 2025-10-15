from typing import Tuple, Protocol

import torch
from torch import Tensor

from sbb.paradigms.policy_gradient.buffer import RLTrajectoryBuffer


class ReturnEstimator(Protocol):
    def compute(
        self, experience_buffer: RLTrajectoryBuffer
    ) -> Tuple[Tensor, Tensor, Tensor]: ...


class GAEReturnEstimator:
    def __init__(
        self,
        gamma: float,
        gae_lambda: float,
        dtype: torch.dtype,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.dtype = dtype

    def compute(
        self, experience_buffer: RLTrajectoryBuffer
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute TD errors (RPE) and GAE advantages separately.

        Returns:
            rpe: TD errors for value function updates
            advantages: GAE advantages for policy updates
            value_targets: Target values for value function
        """
        advantages = torch.zeros_like(experience_buffer.rewards, dtype=self.dtype)
        rpe = torch.zeros_like(experience_buffer.rewards, dtype=self.dtype)
        last_gae_lam = torch.zeros(
            experience_buffer.batch_size, dtype=self.dtype, device=experience_buffer.rewards.device
        )
        for t in reversed(range(experience_buffer.steps)):
            next_value = experience_buffer.state_value_estimates[t + 1]
            not_done = (~experience_buffer.terminations[t]).to(self.dtype)
            delta = (
                experience_buffer.rewards[t]
                + self.gamma * next_value * not_done
                - experience_buffer.state_value_estimates[t]
            )
            rpe[t] = delta  # TD error / RPE
            gae_value = delta + self.gamma * self.gae_lambda * last_gae_lam * not_done
            advantages[t] = gae_value
            last_gae_lam = gae_value
        value_targets = advantages + experience_buffer.state_value_estimates[:-1]
        return rpe, advantages, value_targets


class TD0ReturnEstimator(GAEReturnEstimator):
    def __init__(self, gamma: float, dtype: torch.dtype):
        super().__init__(gamma=gamma, gae_lambda=0.0, dtype=dtype)


class MonteCarloReturnEstimator(GAEReturnEstimator):
    def __init__(self, gamma: float, dtype: torch.dtype):
        super().__init__(gamma=gamma, gae_lambda=1.0, dtype=dtype)
