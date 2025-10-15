from dataclasses import dataclass
from typing import Optional, Tuple, Generic, TypeVar

import torch
from torch import Tensor
from torch.nn import Module

from sbb.base import BaseModel
from sbb.const import DEVICE, EPS
from sbb.heads.feedback import Feedback
from sbb.hyperparameters import BaseConfig
from sbb.paradigms.policy_gradient.buffer import RLTrajectoryBuffer
from sbb.paradigms.policy_gradient.estimator import ReturnEstimator
from sbb.types import SystemStateTuple

from sbb.heads import Value, DiscretePolicyHead, ContinuousPolicyHead


@dataclass
class ReinforcementLearningConfig(BaseConfig):
    """
    Configuration for Reinforcement Learning tasks.

    Uses:
      - Policy head with normalized gradients (implicit lr=1.0)
      - DiscretePolicyHead/ContinuousPolicyHead with normalized gradients
      - CoState head with RLS(diagonal=False) for co-state estimation
    """

    n_rollout_steps: int = 128
    value_lr: float = 1.0
    policy_lr: float = 0.01


PolicyHeadType = TypeVar("PolicyHeadType", DiscretePolicyHead, ContinuousPolicyHead)


class PolicyGradientAgent(Module, Generic[PolicyHeadType]):
    """
    Actor-Critic with unified heads:
      - policy_head: DiscretePolicyHead or ContinuousPolicyHead
      - value_head: ValueHead (rows=1)
      - feedback: maps scalar advantage -> neuron co-state
    """

    def __init__(
        self,
        cfg: ReinforcementLearningConfig,
        policy_head: PolicyHeadType,
        estimator: ReturnEstimator,
    ):
        super().__init__()
        self.cfg = cfg

        self.dtype = cfg.dtype
        self.max_norm = cfg.max_norm
        self.delta_max_norm = cfg.delta_max_norm

        self.recurrent_policy_network = BaseModel(cfg)
        self.policy_head = policy_head
        self.return_estimator = estimator

        self.value_head = Value(
            rows=1,
            cols=self.cfg.total_neurons,
            dtype=self.dtype,
            max_norm=self.max_norm,
            delta_max_norm=self.delta_max_norm,
            initial_scale=self.cfg.initial_weight_scale,
            lr=cfg.policy_lr,
            name="ValueHead",
        )
        self.feedback = Feedback(
            input_dim=1,
            state_dim=self.cfg.total_neurons,
            dtype=self.dtype,
            max_norm=self.max_norm,
            delta_max_norm=self.delta_max_norm,
            lr=1.0,
            name="Feedback",
        )

    def forward(
        self, input_sequence: Tensor, initial_state: SystemStateTuple
    ) -> Tuple[SystemStateTuple, Tensor]:
        return self.recurrent_policy_network.forward(input_sequence, initial_state)

    @torch.no_grad()
    def _estimate_state_value(self, state: Tensor) -> Tensor:
        return self.value_head(state).squeeze(-1)

    @torch.no_grad()
    def act(
        self, state_tuple: SystemStateTuple, action_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if isinstance(self.policy_head, DiscretePolicyHead):
            action, log_prob = self.policy_head.sample(
                state_tuple.activations, action_mask
            )
        else:
            action, log_prob = self.policy_head.sample(state_tuple.activations)
        value = self._estimate_state_value(state_tuple.activations)
        return action, log_prob, value

    def backward(self, experience_buffer: RLTrajectoryBuffer) -> Tensor:
        if not self.training:
            return torch.tensor(0.0, dtype=self.dtype, device=DEVICE)

        # Compute RPE (for value head) and advantages (for policy) separately
        rpe, advantages, value_targets = self.return_estimator.compute(
            experience_buffer
        )

        states_flat = experience_buffer.system_states[:-1].flatten(0, 1)
        actions_flat = experience_buffer.actions.flatten(0, 1)
        advantages_flat = advantages.flatten(0, 1)
        rpe_flat = rpe.flatten(0, 1)

        # Compute inverse norm for normalized gradient descent
        inv_norm = 1.0 / (torch.sum(states_flat**2, dim=-1, keepdim=True) + EPS)

        # Update value head using RPE (TD errors), not advantages
        self.value_head.backward(
            x=states_flat,
            error=rpe_flat.unsqueeze(1),
            inv_norm=inv_norm,
        )
        value_loss = rpe_flat.pow(2).mean()

        # Update policy head using raw advantages (not L2 projected)
        self.policy_head.backward(
            state=states_flat,
            advantage=advantages_flat,
            action=actions_flat,
            inv_norm=inv_norm,
        )

        # Learn feedback projection: RPE -> neuron-level co-state (matching REPO A's structure)
        rpe_local = rpe_flat.unsqueeze(
            1
        )  # Use RPE signal as input to StaticFeedback head

        # Target costate: RPE projected back through the value head weight
        target_costate = (
            rpe_local @ self.value_head.weight
        )  # [T*B, 1] @ [1, N] -> [T*B, N]

        # Update B_err_proj to better map RPE to target co-state
        self.feedback.backward(
            local_signal=rpe_local,  # Use RPE as input
            target_costate=target_costate,
        )

        # Project RPE through learned B_err_proj for recurrent plasticity
        lambda_hat_flat = self.feedback(rpe_local)  # Project RPE
        variational_signal = lambda_hat_flat.view_as(
            experience_buffer.system_states[:-1]
        )

        # Apply plasticity to recurrent weights using projected RPE signal
        self.recurrent_policy_network.backward(
            system_states=experience_buffer.system_states[:-1],
            eligibility_traces=experience_buffer.eligibility_traces[:-1],
            activity_traces=experience_buffer.homeostatic_traces[:-1],
            variational_signal=variational_signal,
            inverse_state_norms=inv_norm,
        )

        return value_loss

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self


class ContinuousPolicyGradientAgent(PolicyGradientAgent[ContinuousPolicyHead]):
    def __init__(self, cfg: ReinforcementLearningConfig, estimator: ReturnEstimator):
        policy_head = ContinuousPolicyHead(
            state_dim=cfg.total_neurons,
            action_dim=cfg.output_features,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            initial_scale=cfg.initial_weight_scale,
            lr=cfg.policy_lr,
            name="ContinuousPolicyHead",
        )
        super().__init__(cfg, policy_head, estimator)


class DiscretePolicyGradientAgent(PolicyGradientAgent[DiscretePolicyHead]):
    def __init__(self, cfg: ReinforcementLearningConfig, estimator: ReturnEstimator):
        policy_head = DiscretePolicyHead(
            state_dim=cfg.total_neurons,
            num_actions=cfg.output_features,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            initial_scale=cfg.initial_weight_scale,
            lr=cfg.policy_lr,
            name="DiscretePolicyHead",
        )
        super().__init__(cfg, policy_head, estimator)
