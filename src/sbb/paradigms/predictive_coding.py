import dataclasses
import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

from sbb.base import BaseModel
from sbb.const import DEVICE, EPS
from sbb.heads.feedback import Feedback
from sbb.hyperparameters import BaseConfig
from sbb.types import SystemStateTuple

from sbb.heads import Linear


@dataclass
class SupervisedConfig(BaseConfig):
    """
    Configuration for predictive coding learning paradigm.

    Extends BaseConfig with learning rates for output heads.
    Predictive coding minimizes prediction error via gradient descent on both
    forward (prediction) and backward (costate) pathways.

    Architecture:
    - Linear head: Learns output mapping y = s @ W^T via NLMS (Normalized LMS)
    - CoState head: Learns backward mapping λ̂ = tanh(error @ W^T) via RLS (Recursive Least Squares)

    The costate (λ̂) provides a learned approximation to backpropagation's error
    gradient, avoiding the weight transport problem while maintaining efficiency.

    All learning rates are derived from normalized gradients (implicit lr=1.0).
    """

    def __post_init__(self):
        super().__post_init__()


# ----- Predictive Coding module -----


class PredictiveCoding(Module):
    """
    Predictive coding learning paradigm for sequence prediction tasks.

    This module implements a biologically-inspired learning algorithm that
    minimizes prediction error through bidirectional gradient approximation.
    Unlike backpropagation, it uses locally-computed learning signals and
    avoids weight transport (symmetric feedback connections).

    Components:
    1. Generative model: Recurrent dynamics for temporal prediction
    2. Prediction head: Forward readout y = s @ W_out^T (NLMS adaptation)
    3. Costate head: Backward error mapping λ̂ = tanh(e @ W_back^T) (RLS adaptation)

    Learning flow:
        1. Predict next observation from current state
        2. Compute prediction error: e = y_pred - y_target
        3. Update forward path: W_out ← W_out - η * e * s^T
        4. Compute target costate: λ* = e @ W_out (backprop substitute)
        5. Align costate network: minimize ||λ̂ - λ*||²
        6. Apply plasticity to recurrent weights using λ̂ as error signal

    The costate provides a learned gradient approximation that adapts to the
    task structure, offering a middle ground between random feedback (poor
    performance) and symmetric backprop (biologically implausible).

    Attributes
    ----------
    cfg : SupervisedConfig
        Configuration with learning rates and dimensions.
    base : BaseModel
        Core recurrent network with block-sparse connectivity.
    readout : Linear
        Output layer trained via NLMS (Normalized Least Mean Squares).
    feedback : Feedback
        Backward path trained via slow gradient descent.

    Methods
    -------
    forward(sensory_input, current_state)
        Generate next-step prediction from current observation.
    backward(predictions, targets, state, next_state)
        Update all parameters to reduce prediction error.
    """

    def __init__(self, cfg: SupervisedConfig):
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
            initial_scale=math.sqrt(cfg.output_features / (cfg.total_neurons + EPS)),
            name="PredictionHead",
        )
        self.feedback = Feedback(
            input_dim=cfg.output_features,
            state_dim=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            name="CoState",
        )

    def forward(
        self,
        sensory_input: Tensor,
        current_state: SystemStateTuple,
    ) -> Tuple[Tensor, SystemStateTuple]:
        """
        Generate prediction for next timestep given current observation.

        Runs one step of recurrent dynamics conditioned on sensory input,
        then reads out prediction via the linear output head.

        Parameters
        ----------
        sensory_input : Tensor [B, input_features]
            Current observation to condition state evolution.
        current_state : SystemStateTuple
            Current internal state (from previous step or initialization).

        Returns
        -------
        prediction : Tensor [B, output_features]
            Predicted next observation.
        next_state : SystemStateTuple
            Updated internal state after processing current input.
        """
        input_sequence = sensory_input.unsqueeze(0)
        next_state, state_trajectory = self.base.forward(input_sequence, current_state)
        prediction = self.readout(state_trajectory[0])
        return prediction, next_state

    def backward(
        self,
        sensory_predictions: Tensor,
        sensory_targets: Tensor,
        state_tuple: SystemStateTuple,
        next_state_tuple: SystemStateTuple,
    ) -> Tuple[Tensor, SystemStateTuple]:
        """
        Update all learnable parameters to reduce prediction error.

        Implements the predictive coding learning algorithm:
        1. Compute error: e = prediction - target
        2. Update output weights via NLMS: W_out ← W_out - η * e * s^T / ||s||²
        3. Compute target costate: λ* = e @ W_out (ideal error backprop)
        4. Update costate weights via RLS to minimize ||λ̂ - λ*||²
        5. Apply recurrent plasticity using learned costate λ̂ as error signal

        The costate approximation avoids weight transport while achieving
        performance comparable to backpropagation on many tasks.

        Parameters
        ----------
        sensory_predictions : Tensor [B, output_features]
            Model's prediction from forward().
        sensory_targets : Tensor [B, output_features]
            Ground truth next observation.
        state_tuple : SystemStateTuple
            State before prediction (currently unused).
        next_state_tuple : SystemStateTuple
            State after prediction, used for plasticity updates.

        Returns
        -------
        loss : Tensor (scalar)
            Mean squared prediction error for logging.
        final_state : SystemStateTuple
            Updated state with refreshed homeostatic bias.

        Notes
        -----
        In eval mode, returns zero loss and unchanged state without updates.
        The homeostatic bias is refreshed from base to sync with
        any changes from plasticity updates.
        """
        if not self.training:
            return (
                torch.tensor(0.0, dtype=self.dtype, device=DEVICE),
                next_state_tuple,
            )

        error = sensory_predictions.to(DEVICE, self.dtype) - sensory_targets.to(
            DEVICE, self.dtype
        )

        s_next = next_state_tuple.activations
        inv_norm = 1.0 / (torch.sum(s_next**2, dim=1, keepdim=True) + EPS)

        self.readout.backward(x=s_next, error=error, inv_norm=inv_norm)

        lambda_star = error @ self.readout.weight
        self.feedback.backward(
            local_signal=error,
            target_costate=lambda_star,
        )
        lambda_hat = self.feedback(error)

        self.base.backward(
            system_states=next_state_tuple.activations.unsqueeze(0),
            eligibility_traces=next_state_tuple.eligibility_trace.unsqueeze(0),
            activity_traces=next_state_tuple.homeostatic_trace.unsqueeze(0),
            variational_signal=lambda_hat.unsqueeze(0),
            inverse_state_norms=inv_norm,
        )

        loss = error.pow(2).mean()
        final_state_tuple = dataclasses.replace(
            next_state_tuple,
            bias=self.base.activity_bias.expand(s_next.shape[0], -1).clone(),
        )
        return loss, final_state_tuple

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self
