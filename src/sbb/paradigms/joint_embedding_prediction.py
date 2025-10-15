"""
Joint-Embedding Predictive Architecture (JEPA) for self-supervised learning.

This module implements JEPA, a self-supervised learning framework that learns
representations by predicting latent embeddings of future observations rather
than raw pixels. This approach learns abstract, invariant features that transfer
well to downstream tasks.
"""

import torch

from torch.nn import Module
from torch import Tensor
from typing import Tuple

from dataclasses import dataclass

from sbb.base import BaseModel
from sbb.const import DEVICE, EPS
from sbb.heads.feedback import Feedback
from sbb.hyperparameters import BaseConfig
from sbb.types import SystemStateTuple

from sbb.heads import Linear, CoState


@dataclass
class JEPAHyperparameters(BaseConfig):
    """Configuration for Self-Supervised Learning with a JEPA objective."""

    target_ema_decay: float = 0.996


class JEPA(Module):
    """
    Joint-Embedding Predictive Architecture for representation learning.

    JEPA learns representations by predicting the latent embedding of a target
    sequence from a context sequence. Unlike pixel-based prediction, this approach
    learns high-level, abstract features invariant to low-level details.

    Architecture:
    - online_encoder: Learns from context sequences via gradient descent
    - target_encoder: EMA-updated encoder providing prediction targets
    - predictor_head: Maps online embeddings to target embedding space
    - feedback: CoState head for backpropagating prediction errors

    The target encoder is updated via exponential moving average (EMA) to provide
    stable targets and prevent representation collapse. This follows the momentum
    encoder paradigm from contrastive learning.

    Attributes
    ----------
    cfg : JEPAHyperparameters
        Configuration with EMA decay rate.
    online_encoder : BaseModel
        Actively trained encoder.
    target_encoder : BaseModel
        Momentum-updated encoder for stable targets.
    predictor_head : Linear
        Maps online embeddings to target space.
    feedback : CoState
        Backward error projection for learning.

    Methods
    -------
    learn(context_sequence, target_sequence)
        Update representations from paired sequences.
    """

    def __init__(self, cfg: JEPAHyperparameters):
        super().__init__()
        self.cfg = cfg

        self.dtype = cfg.dtype

        self.online_encoder = BaseModel(cfg)
        self.target_encoder = BaseModel(cfg)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor_head = Linear(
            rows=cfg.total_neurons,
            cols=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            initial_scale=1.0,
            name="PredictorHead",
        )
        self.feedback = Feedback(
            input_dim=cfg.total_neurons,
            state_dim=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            name="CoState",
        )

        self.target_ema_decay = 0.996

    @torch.no_grad()
    def _update_target_network(self):
        decay = self.target_ema_decay
        online_params = self.online_encoder.state_dict()
        target_params = self.target_encoder.state_dict()
        for name, param in online_params.items():
            if name in target_params and isinstance(target_params[name], torch.Tensor):
                target_params[name].data.mul_((decay)).add_(param.data, alpha=1 - decay)
        self.target_encoder.load_state_dict(target_params)

    def _get_representations(
        self, context_sequence: Tensor, target_sequence: Tensor
    ) -> Tuple[Tensor, Tensor, SystemStateTuple]:
        batch_size = context_sequence.shape[1]
        initial_state = self.online_encoder.new_state(batch_size)

        online_final_state, _ = self.online_encoder.forward(
            context_sequence, initial_state
        )
        context_embedding = online_final_state.activations

        with torch.no_grad():
            target_initial_state = self.target_encoder.new_state(batch_size)
            target_final_state, _ = self.target_encoder.forward(
                target_sequence, target_initial_state
            )
            target_embedding = target_final_state.activations

        return context_embedding, target_embedding, online_final_state

    def learn(
        self, context_sequence: Tensor, target_sequence: Tensor
    ) -> Tuple[Tensor, SystemStateTuple]:
        if not self.training:
            batch_size = context_sequence.shape[1]
            initial_state = self.online_encoder.new_state(batch_size)
            final_state, _ = self.online_encoder.forward(
                context_sequence, initial_state
            )
            return torch.tensor(0.0, dtype=self.dtype, device=DEVICE), final_state

        s_context, s_target, online_state_tuple = self._get_representations(
            context_sequence, target_sequence
        )

        s_pred = self.predictor_head(s_context)
        error = s_pred - s_target.detach()
        loss = error.pow(2).mean()

        inv_norm = 1.0 / (torch.sum(s_context**2, dim=1, keepdim=True) + EPS)
        self.predictor_head.backward(
            x=s_context,
            error=error,
            inv_norm=inv_norm,
        )

        lambda_hat = self.feedback(error)
        lambda_star = error @ self.predictor_head.weight
        self.feedback.backward(
            local_signal=error,
            target_costate=lambda_star,
        )

        self.online_encoder.backward(
            system_states=online_state_tuple.activations.unsqueeze(0),
            eligibility_traces=online_state_tuple.eligibility_trace.unsqueeze(0),
            activity_traces=online_state_tuple.homeostatic_trace.unsqueeze(0),
            variational_signal=lambda_hat.unsqueeze(0),
            inverse_state_norms=inv_norm,
        )

        return loss, online_state_tuple

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self
