"""
Active Inference learning paradigm for goal-directed agents.

This module implements active inference, a biologically-inspired framework where
agents minimize expected free energy by selecting actions that simultaneously
satisfy goals (instrumental value) and reduce uncertainty (epistemic value).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.distributions import (
    Distribution,
    Categorical,
    Normal,
    Independent,
    TransformedDistribution,
)

from sbb.base import BaseModel
from sbb.const import DEVICE, EPS
from sbb.heads.feedback import Feedback
from sbb.hyperparameters import BaseConfig
from sbb.types import SystemStateTuple

from sbb.heads import Linear, CoState


@dataclass
class ActiveInferenceHyperparameters(BaseConfig):
    """
    Configuration for Active Inference tasks.

    Uses:
      - Linear head with normalized gradients (implicit lr=1.0)
      - CoState head with RLS(diagonal=False) for co-state estimation
    """

    prediction_horizon: int = 10
    epistemic_weight: float = 0.1

    num_stochastic_policies: int = 64
    include_deterministic_per_action: bool = True
    null_action_index: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.prediction_horizon = max(1, int(self.prediction_horizon))
        self.num_stochastic_policies = max(0, int(self.num_stochastic_policies))


class ActiveInferenceAgent(Module):
    """
    Active Inference agent with generative model and planning.

    This agent implements the active inference framework where action selection
    minimizes expected free energy (EFE). The agent maintains a generative world
    model and uses tree search over candidate action sequences to find actions
    that best achieve goals while reducing uncertainty.

    Components:
    - base: Recurrent dynamics for imagining future trajectories
    - readout: Linear head mapping states to predicted observations
    - feedback: CoState head for backpropagating prediction errors

    The agent balances two objectives:
    1. Instrumental value: Achieving goal observations (pragmatic)
    2. Epistemic value: Reducing uncertainty about hidden states (exploratory)

    Attributes
    ----------
    cfg : ActiveInferenceHyperparameters
        Configuration with planning horizon and epistemic weighting.
    base : BaseModel
        Recurrent network for simulating action outcomes.
    readout : Linear
        Maps hidden states to sensory predictions.
    feedback : CoState
        Backward error projection for learning.

    Methods
    -------
    forward(state_tuple, goal_distribution)
        Plan action by evaluating candidate sequences.
    backward(prior_state, action_taken, true_observation)
        Update world model from observed outcomes.
    """

    def __init__(self, cfg: ActiveInferenceHyperparameters):
        super().__init__()
        self.cfg = cfg

        self.dtype = cfg.dtype

        self.base = BaseModel(cfg)

        self.readout = Linear(
            rows=cfg.output_features,
            cols=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            initial_scale=1.0,
            name="SensoryPredictionHead",
        )
        self.feedback = Feedback(
            input_dim=cfg.output_features,
            state_dim=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            name="CoState",
        )

    def _build_candidate_sequences(self, batch_size: int) -> Tensor:
        T = self.cfg.prediction_horizon
        A = self.cfg.input_features
        assert batch_size == 1, "Planner assumes single-environment planning."

        dtype = self.dtype
        eye = torch.eye(A, dtype=dtype, device=DEVICE)

        # Deterministic per-action candidates: [A, T, A]
        det = None
        if self.cfg.include_deterministic_per_action:
            if (
                self.cfg.null_action_index is not None
                and 0 <= self.cfg.null_action_index < A
            ):
                null_one_hot = eye[self.cfg.null_action_index].unsqueeze(0)  # [1,A]
                tail = (
                    null_one_hot.expand(T - 1, -1)
                    if T > 1
                    else torch.empty(0, A, dtype=dtype, device=DEVICE)
                )
            else:
                tail = (
                    torch.zeros(T - 1, A, dtype=dtype, device=DEVICE)
                    if T > 1
                    else torch.empty(0, A, dtype=dtype, device=DEVICE)
                )
            det_list = []
            for a in range(A):
                first = eye[a].unsqueeze(0)  # [1,A]
                seq = torch.cat([first, tail], dim=0) if T > 1 else first  # [T,A]
                det_list.append(seq)
            det = torch.stack(det_list, dim=0) if det_list else None  # [A,T,A]

        # Stochastic first-step candidates: [K, T, A]
        rand = None
        K = int(self.cfg.num_stochastic_policies)
        if K > 0:
            dist = Categorical(logits=torch.zeros(A, dtype=dtype, device=DEVICE))
            sampled_idx = dist.sample((K,))  # [K]
            first_step = torch.nn.functional.one_hot(sampled_idx, num_classes=A).to(
                dtype
            )  # [K,A]
            if (
                self.cfg.null_action_index is not None
                and 0 <= self.cfg.null_action_index < A
            ):
                null_oh = torch.nn.functional.one_hot(
                    torch.tensor(self.cfg.null_action_index, device=DEVICE),
                    num_classes=A,
                ).to(dtype)
                tail = (
                    null_oh.unsqueeze(0).expand(T - 1, -1)
                    if T > 1
                    else torch.empty(0, A, dtype=dtype, device=DEVICE)
                )
            else:
                tail = (
                    torch.zeros(T - 1, A, dtype=dtype, device=DEVICE)
                    if T > 1
                    else torch.empty(0, A, dtype=dtype, device=DEVICE)
                )
            rand = (
                first_step.unsqueeze(1)
                if T == 1
                else torch.cat(
                    [first_step.unsqueeze(1), tail.unsqueeze(0).expand(K, -1, -1)],
                    dim=1,
                )
            )  # [K,T,A]

        if det is not None and rand is not None:
            return torch.cat([det, rand], dim=0)
        if det is not None:
            return det
        if rand is not None:
            return rand
        # Fallback: a single zero policy, optionally with a null-action first step
        seq = torch.zeros(T, A, dtype=dtype, device=DEVICE)
        if (
            self.cfg.null_action_index is not None
            and 0 <= self.cfg.null_action_index < A
        ):
            seq[0, self.cfg.null_action_index] = 1.0
        return seq.unsqueeze(0)

    def _simulate_policy_outcomes(
        self, initial_state: SystemStateTuple, candidate_policy_actions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Simulate imagined trajectories for each candidate. Uses head.forward for predictions.
        """
        num_policies, horizon, _ = candidate_policy_actions.shape
        st = initial_state
        batched_state = SystemStateTuple(
            activations=st.activations.expand(num_policies, -1).contiguous(),
            eligibility_trace=st.eligibility_trace.expand(
                num_policies, -1
            ).contiguous(),
            homeostatic_trace=st.homeostatic_trace.expand(
                num_policies, -1
            ).contiguous(),
            bias=st.bias.expand(num_policies, -1).contiguous(),
            input_projection=st.input_projection.expand(num_policies, -1).contiguous(),
            noise=st.noise.expand(num_policies).contiguous(),
        )

        inp = candidate_policy_actions.permute(1, 0, 2)
        _, state_trajectory = self.base.forward(inp, batched_state)
        imagined_internal_states = state_trajectory.permute(1, 0, 2)

        flat_states = imagined_internal_states.reshape(num_policies * horizon, -1)
        flat_pred = self.readout(flat_states)
        imagined_sensory_outcomes = torch.tanh(flat_pred).reshape(
            num_policies, horizon, -1
        )
        return imagined_internal_states, imagined_sensory_outcomes

    def _evaluate_expected_free_energy(
        self,
        imagined_internal_states: Tensor,
        imagined_sensory_outcomes: Tensor,
        goal_distribution: Distribution,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Unwrap Independent and TransformedDistribution to get to the base distribution
        base = goal_distribution
        if isinstance(base, Independent):
            base = base.base_dist  # type: ignore[attr-defined]
        if isinstance(base, TransformedDistribution):
            base = base.base_dist  # type: ignore[attr-defined]

        expanded_loc = base.loc.expand_as(imagined_sensory_outcomes)  # type: ignore[attr-defined]
        expanded_scale = base.scale.expand_as(imagined_sensory_outcomes)  # type: ignore[attr-defined]
        aligned_goal = Independent(Normal(loc=expanded_loc, scale=expanded_scale), 1)

        instrumental = -aligned_goal.log_prob(imagined_sensory_outcomes).sum(dim=1)
        final_states = imagined_internal_states[:, -1, :]
        epistemic = torch.var(final_states, dim=0).sum()

        efe = instrumental - self.cfg.epistemic_weight * epistemic
        return efe, instrumental, epistemic

    @torch.no_grad()
    def forward(
        self, state_tuple: SystemStateTuple, goal_distribution: Distribution
    ) -> Tuple[Tensor, Dict]:
        self.eval()
        candidates = self._build_candidate_sequences(batch_size=1)
        if candidates.numel() == 0:
            action = torch.zeros(
                self.cfg.input_features, dtype=self.dtype, device=DEVICE
            )
            return action, {"efe": 0.0, "instrumental": 0.0, "epistemic": 0.0}

        imagined_states, imagined_obs = self._simulate_policy_outcomes(
            state_tuple, candidates
        )
        efe, instrumental, epistemic = self._evaluate_expected_free_energy(
            imagined_states, imagined_obs, goal_distribution
        )
        best = torch.argmin(efe)
        first_action = candidates[best, 0]
        return first_action, {
            "efe": efe[best].item(),
            "instrumental": instrumental[best].item(),
            "epistemic": epistemic.item(),
        }

    def backward(
        self,
        prior_state: SystemStateTuple,
        action_taken: Tensor,
        true_observation: Tensor,
    ) -> Tuple[SystemStateTuple, Dict]:
        action_seq = action_taken.unsqueeze(0)
        next_state, state_traj = self.base.forward(action_seq, prior_state)

        obs_pred = torch.tanh(self.readout(state_traj[0]))

        surprise = obs_pred - true_observation
        s = next_state.activations
        inv_norm = 1.0 / (torch.sum(s**2, dim=1, keepdim=True) + EPS)

        self.readout.backward(x=s, error=surprise, inv_norm=inv_norm)

        lambda_hat = self.feedback(surprise)
        dL_dS = (
            (obs_pred - true_observation) * (1.0 - obs_pred * obs_pred)
        ) @ self.readout.weight
        self.feedback.backward(
            local_signal=surprise,
            target_costate=dL_dS,
        )

        self.base.backward(
            system_states=next_state.activations.unsqueeze(0),
            eligibility_traces=next_state.eligibility_trace.unsqueeze(0),
            activity_traces=next_state.homeostatic_trace.unsqueeze(0),
            variational_signal=lambda_hat.unsqueeze(0),
            inverse_state_norms=inv_norm,
        )

        return next_state, {"surprise_mse": torch.mean(surprise.pow(2)).item()}

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self
