"""
Active Learning paradigm combining sensory grounding with latent consistency.

This module implements a hybrid learning approach that combines Active Inference
(sensory prediction) with JEPA-style representation learning (latent prediction).
The dual objective provides both perceptual grounding and abstract structure learning.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.distributions import Distribution

from sbb.base import BaseModel
from sbb.const import EPS
from sbb.heads.feedback import Feedback
from sbb.hyperparameters import BaseConfig
from sbb.types import SystemStateTuple

from sbb.heads import Linear, CoState


@dataclass
class ActiveLearningHyperparameters(BaseConfig):
    prediction_horizon: int = 10
    epistemic_weight: float = 0.1

    num_stochastic_policies: int = 64
    include_deterministic_per_action: bool = True
    null_action_index: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.prediction_horizon = max(1, int(self.prediction_horizon))
        self.num_stochastic_policies = max(0, int(self.num_stochastic_policies))


class ActiveLearningAgent(Module):
    """
    Hybrid agent combining sensory prediction and latent representation learning.

    This agent implements a dual learning objective that trains representations
    to both predict sensory observations (Active Inference) and maintain temporal
    consistency in latent space (JEPA). The combination provides grounded abstractions
    that generalize better than either objective alone.

    Architecture:
    - online_encoder: Learns representations from current observations
    - target_encoder: EMA-updated copy providing stable prediction targets
    - readout: Predicts observations from latent states (AIF objective)
    - predictor_head: Predicts future latent states (JEPA objective)
    - feedback_sensory: Backprop approximation for sensory errors
    - feedback_latent: Backprop approximation for representation errors

    The target encoder prevents representation collapse and provides stable
    learning signals, following the self-supervised learning paradigm.

    Attributes
    ----------
    cfg : ActiveLearningHyperparameters
        Configuration with EMA decay rate.
    online_encoder : BaseModel
        Actively trained encoder network.
    target_encoder : BaseModel
        Momentum-updated encoder for stable targets.
    sensory_readout : Linear
        Maps states to sensory predictions.
    predictor_head : Linear
        Predicts target encoder outputs from online encoder states.
    feedback_sensory : CoState
        Backward projection for sensory errors.
    feedback_latent : CoState
        Backward projection for representation errors.

    Methods
    -------
    learn(prior_state, action_taken, current_observation, goal_distribution)
        Update all parameters from a single transition.
    """

    def __init__(self, cfg: ActiveLearningHyperparameters):
        super().__init__()
        self.cfg = cfg

        self.dtype = cfg.dtype

        self.online_encoder = BaseModel(cfg)
        self.base = self.online_encoder

        self.target_encoder = BaseModel(cfg)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.sensory_readout = Linear(
            rows=cfg.output_features,
            cols=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            initial_scale=1.0,
            name="SensoryPredictionHead",
        )
        self.predictor_head = Linear(
            rows=cfg.total_neurons,
            cols=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            initial_scale=1.0,
            name="PredictorHead",
        )
        self.feedback_sensory = Feedback(
            input_dim=cfg.output_features,
            state_dim=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            name="CoStateSensory",
        )
        self.feedback_latent = Feedback(
            input_dim=cfg.total_neurons,
            state_dim=cfg.total_neurons,
            dtype=cfg.dtype,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
            name="CoStateLatent",
        )

        self.target_ema_decay = 0.996

    @torch.no_grad()
    def _update_target_encoder(self):
        decay = self.target_ema_decay
        online = self.online_encoder.state_dict()
        target = self.target_encoder.state_dict()
        for k, v in online.items():
            if k in target and isinstance(target[k], torch.Tensor):
                target[k].mul_(decay).add_(v, alpha=1.0 - decay)
        self.target_encoder.load_state_dict(target)

    def _update_sensory_head_and_core(
        self, next_state: SystemStateTuple, obs_pred: Tensor, obs_true: Tensor
    ) -> float:
        surprise = obs_pred - obs_true
        s = next_state.activations
        inv_norm = 1.0 / (torch.sum(s**2, dim=1, keepdim=True) + EPS)

        self.sensory_readout.backward(x=s, error=surprise, inv_norm=inv_norm)

        lambda_hat = self.feedback_sensory(surprise)
        dL_dS = (
            (obs_pred - obs_true) * (1.0 - obs_pred * obs_pred)
        ) @ self.sensory_readout.weight
        self.feedback_sensory.backward(
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
        return torch.mean(surprise.pow(2)).item()

    def _update_predictor_head_and_core(
        self, prior_state: SystemStateTuple, current_observation: Tensor
    ) -> float:
        B = current_observation.shape[0]

        with torch.no_grad():
            init_tgt = self.target_encoder.new_state(B)
            final_tgt, _ = self.target_encoder.forward(
                current_observation.unsqueeze(0), init_tgt
            )
            s_target = final_tgt.activations

        s_prior = prior_state.activations
        s_pred = self.predictor_head(s_prior)
        err = s_pred - s_target
        inv_norm = 1.0 / (torch.sum(s_prior**2, dim=1, keepdim=True) + EPS)

        self.predictor_head.backward(
            x=s_prior,
            error=err,
            inv_norm=inv_norm,
        )

        lambda_hat = self.feedback_latent(err)
        dL_dSprior = err @ self.predictor_head.weight
        self.feedback_latent.backward(
            local_signal=err,
            target_costate=dL_dSprior,
        )

        self.base.backward(
            system_states=prior_state.activations.unsqueeze(0),
            eligibility_traces=prior_state.eligibility_trace.unsqueeze(0),
            activity_traces=prior_state.homeostatic_trace.unsqueeze(0),
            variational_signal=lambda_hat.unsqueeze(0),
            inverse_state_norms=inv_norm,
        )
        return torch.mean(err.pow(2)).item()

    def learn(
        self,
        prior_state: SystemStateTuple,
        action_taken: Tensor,
        current_observation: Tensor,
        goal_distribution: Optional[Distribution] = None,
    ) -> Tuple[Dict[str, float], SystemStateTuple]:
        action_seq = action_taken.unsqueeze(0)
        next_state, state_traj = self.base.forward(action_seq, prior_state)
        obs_pred = torch.tanh(self.sensory_readout(state_traj[0]))

        aif_mse = self._update_sensory_head_and_core(
            next_state, obs_pred, current_observation
        )
        jepa_mse = self._update_predictor_head_and_core(
            prior_state, current_observation
        )

        self._update_target_encoder()

        metrics = {
            "aif_surprise_mse": aif_mse,
            "jepa_representation_mse": jepa_mse,
        }
        return metrics, next_state

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self
