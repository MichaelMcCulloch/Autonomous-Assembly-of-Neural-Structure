"""
Policy heads for reinforcement learning agents.

This module provides discrete and continuous policy parameterizations with
online gradient-based adaptation. Both use projected SGD with normalized
gradients for stable learning in non-stationary environments.
"""

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.distributions import Normal, Categorical

from sbb.util import _orthogonal, _project_l2_norm
from sbb.const import DEVICE, EPS


class DiscretePolicyHead(Module):
    """
    Discrete action policy with softmax distribution.

    Implements a linear softmax policy: π(a|s) = softmax(s @ W^T). The weights
    are updated via projected SGD using policy gradient with advantages. The
    three-factor learning rule (state × advantage × action-probability-error)
    provides credit assignment without full backpropagation.

    Parameters
    ----------
    state_dim : int
        Dimension of input state representation.
    num_actions : int
        Number of discrete actions.
    dtype : torch.dtype
        Parameter precision.
    max_norm : float
        Maximum L2 norm for weight matrix.
    delta_max_norm : float
        Maximum L2 norm for weight updates.
    initial_scale : float, optional
        Scale for orthogonal initialization. Default 1.0.
    lr : float, optional
        Learning rate. Default 1.0 (implicit due to normalization).
    name : str, optional
        Identifier for logging. Default "DiscretePolicyHead".

    Attributes
    ----------
    logit_weight : Parameter [num_actions, state_dim]
        Policy weight matrix, initialized orthogonally.

    Methods
    -------
    forward(state, action_mask=None)
        Compute action distribution and logits.
    sample(state, action_mask=None)
        Sample action and return log probability.
    backward(state, advantage, action, inv_norm)
        Update policy weights via policy gradient.
    """

    logit_weight: Parameter

    def __init__(
        self,
        *,
        state_dim: int,
        num_actions: int,
        dtype: torch.dtype,
        max_norm: float,
        delta_max_norm: float,
        initial_scale: float = 1.0,
        lr: float = 1.0,
        name: str = "DiscretePolicyHead",
    ):
        super().__init__()

        self.dtype = dtype
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.max_norm = float(max_norm)
        self.delta_max_norm = float(delta_max_norm)
        self.lr = float(lr)
        self.name = name

        self.register_parameter(
            "logit_weight",
            Parameter(
                _orthogonal(
                    dtype=dtype,
                    rows=self.num_actions,
                    columns=self.state_dim,
                )
                * initial_scale,
                requires_grad=False,
            ),
        )

    @property
    def weight(self):
        """Alias for logit_weight for consistency with other heads."""
        return self.logit_weight

    @torch.no_grad()
    def forward(
        self, state: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Categorical, Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor [B x state_dim]
            action_mask: Optional boolean mask [B x num_actions]

        Returns:
            (distribution, logits)
        """

        logits = state @ self.logit_weight.T

        if action_mask is not None:
            if action_mask.shape != logits.shape:
                action_mask = action_mask.expand_as(logits)
            logits[~action_mask] = -torch.inf
        return Categorical(logits=logits), logits

    @torch.no_grad()
    def sample(
        self, state: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sample action and log probability."""
        dist, _ = self.forward(state, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    @torch.no_grad()
    def backward(
        self,
        state: Tensor,
        advantage: Tensor,
        action: Tensor,
        inv_norm: Tensor,
    ):
        """
        Update policy weights using projected SGD with normalized gradients.

        The gradient is pre-scaled by inverse state norm, so learning rate = 1.0 (implicit).

        Args:
            state: State tensor [B x state_dim]
            advantage: Advantage values [B]
            action: Actions taken [B] or [B x 1]
            inv_norm: Inverse normalization factor [B x 1]
        """
        state.shape[0]

        dist, logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)

        action_2d = action if action.ndim == 2 else action.unsqueeze(1)
        target = torch.zeros_like(probs).scatter_(1, action_2d, 1.0)

        # Per-sample gradient with normalization (reference uses einsum for outer product per sample)
        error_term = advantage.unsqueeze(1) * (
            target - probs.detach()
        )  # [B, num_actions]
        # Compute batched outer product: [B, state_dim, num_actions]
        grad_batched = torch.einsum(
            "bi,bj->bij", state, error_term
        ) * inv_norm.unsqueeze(-1)
        # Average over batch
        grad = grad_batched.mean(dim=0).T  # [num_actions, state_dim]

        delta = _project_l2_norm(grad, self.delta_max_norm)

        self.logit_weight.data.add_(delta, alpha=self.lr)
        self.logit_weight.data = _project_l2_norm(self.logit_weight.data, self.max_norm)


class ContinuousPolicyHead(Module):
    """
    Continuous action policy with Gaussian distribution.

    Implements a diagonal Gaussian policy: π(a|s) = N(tanh(s @ W^T), exp(log_std)).
    The mean is bounded by tanh, and the standard deviation is learned per
    action dimension. Both are updated via projected SGD with policy gradient.

    The tanh bounds the mean to [-1, 1], making it suitable for control tasks
    with normalized action spaces (e.g., torque limits). The log_std is clamped
    to prevent numerical instability from extreme exploration or exploitation.

    Parameters
    ----------
    state_dim : int
        Dimension of input state representation.
    action_dim : int
        Dimension of continuous action space.
    dtype : torch.dtype
        Parameter precision.
    max_norm : float
        Maximum L2 norm for mean weight matrix.
    delta_max_norm : float
        Maximum L2 norm for weight updates.
    initial_scale : float, optional
        Scale for orthogonal initialization. Default 1.0.
    lr : float, optional
        Learning rate. Default 1.0 (implicit due to normalization).
    name : str, optional
        Identifier for logging. Default "ContinuousPolicyHead".

    Attributes
    ----------
    mean_weight : Parameter [action_dim, state_dim]
        Weight matrix for policy mean.
    log_std : Parameter [1, action_dim]
        Log standard deviation for each action dimension.

    Methods
    -------
    forward(state)
        Compute action distribution and mean.
    sample(state)
        Sample action and return log probability.
    backward(state, advantage, action, inv_norm)
        Update policy parameters via policy gradient.
    """

    mean_weight: Parameter
    log_std: Parameter

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        dtype: torch.dtype,
        max_norm: float,
        delta_max_norm: float,
        initial_scale: float = 1.0,
        lr: float = 1.0,
        name: str = "ContinuousPolicyHead",
    ):
        super().__init__()

        self.dtype = dtype
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_norm = float(max_norm)
        self.delta_max_norm = float(delta_max_norm)
        self.lr = float(lr)
        self.name = name

        self.register_parameter(
            "mean_weight",
            Parameter(
                _orthogonal(
                    dtype=dtype,
                    rows=self.action_dim,
                    columns=self.state_dim,
                )
                * initial_scale,
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "log_std",
            Parameter(
                torch.zeros(1, action_dim, dtype=dtype, device=DEVICE),
                requires_grad=False,
            ),
        )

    @property
    def weight(self):
        """Alias for mean_weight for consistency with other heads."""
        return self.mean_weight

    @torch.no_grad()
    def forward(self, state: Tensor) -> tuple[Normal, Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor [B x state_dim]

        Returns:
            (distribution, mean)
        """
        mean = torch.tanh(state @ self.mean_weight.T)
        std = torch.exp(self.log_std.expand(state.shape[0], -1))
        return Normal(mean, std), mean

    @torch.no_grad()
    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Sample action and log probability."""
        dist, _ = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    @torch.no_grad()
    def backward(
        self,
        state: Tensor,
        advantage: Tensor,
        action: Tensor,
        inv_norm: Tensor,
    ):
        """
        Update policy weights using projected SGD with normalized gradients.

        The gradient is pre-scaled by inverse state norm, so learning rate = 1.0 (implicit).

        Args:
            state: State tensor [B x state_dim]
            advantage: Advantage values [B]
            action: Actions taken [B x action_dim]
            inv_norm: Inverse normalization factor [B x 1]
        """
        dist, mean = self.forward(state)
        B = state.shape[0]

        # Update mean weights - per-sample gradient with normalization
        grad_log_prob_mean = (action - mean) / (dist.scale.pow(2) + EPS)
        error_term = advantage.unsqueeze(1) * grad_log_prob_mean  # [B, action_dim]
        # Compute batched outer product: [B, state_dim, action_dim]
        grad_batched = torch.einsum(
            "bi,bj->bij", state, error_term
        ) * inv_norm.unsqueeze(-1)
        # Average over batch
        grad = grad_batched.mean(dim=0).T  # [action_dim, state_dim]

        delta = _project_l2_norm(grad, self.delta_max_norm)

        self.mean_weight.data.add_(delta, alpha=self.lr)
        self.mean_weight.data = _project_l2_norm(self.mean_weight.data, self.max_norm)

        # Update log_std (with normalized gradient, implicit lr=1.0)
        grad_log_prob_log_std = (action - mean).pow(2) / (dist.scale.pow(2) + EPS) - 1.0
        d_log_std = inv_norm.squeeze(-1) * (
            advantage.unsqueeze(1) * grad_log_prob_log_std
        ).sum(dim=1)
        self.log_std.data.add_(d_log_std.mean() / max(1, B))
        self.log_std.data.clamp_(-2.0, 0.5)
