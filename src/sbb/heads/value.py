"""
Value estimation head for reinforcement learning.

This module provides state-value (V) or action-value (Q) estimation via linear
readout with online gradient-based updates.
"""

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sbb.util import _orthogonal, _project_l2_norm


class Value(Module):
    """
    Value function estimator with temporal difference learning.

    Implements a linear value function V(s) = s @ W^T or Q(s) = s @ W^T for
    multiple outputs. Updated via projected SGD with temporal difference (TD)
    errors, providing stable online learning for value estimation.

    For state-value functions, set rows=1. For action-value functions or
    distributional RL, set rows to the number of outputs needed.

    Parameters
    ----------
    rows : int
        Number of value outputs (1 for V, num_actions for Q).
    cols : int
        Dimension of input state representation.
    dtype : torch.dtype
        Parameter precision.
    max_norm : float
        Maximum L2 norm for weight matrix.
    delta_max_norm : float
        Maximum L2 norm for weight updates.
    initial_scale : float, optional
        Scale for orthogonal initialization. Default 1.0.
    lr : float, optional
        Learning rate for TD updates. Default 0.1.
    name : str, optional
        Identifier for logging. Default "Policy" (legacy name).

    Attributes
    ----------
    weight : Parameter [rows, cols]
        Value function weight matrix, initialized orthogonally.

    Methods
    -------
    forward(x)
        Compute value estimate V = x @ W^T.
    backward(x, error, inv_norm)
        Update weights via TD error with projected SGD.
    """

    weight: Parameter

    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        max_norm: float,
        delta_max_norm: float,
        initial_scale: float = 1.0,
        lr: float = 0.1,
        name: str = "Policy",
    ):
        super().__init__()

        self.dtype = dtype
        self.rows = int(rows)
        self.cols = int(cols)
        self.max_norm = float(max_norm)
        self.delta_max_norm = float(delta_max_norm)
        self.lr = float(lr)
        self.name = name

        self.register_parameter(
            "weight",
            Parameter(
                _orthogonal(dtype=dtype, rows=self.rows, columns=self.cols)
                * initial_scale,
                requires_grad=False,
            ),
        )

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute value estimate.

        Parameters
        ----------
        x : Tensor [B, cols]
            State representation.

        Returns
        -------
        Tensor [B, rows]
            Value estimates V = x @ W^T.
        """
        return x @ self.weight.T

    @torch.no_grad()
    def backward(
        self,
        x: Tensor,
        error: Tensor,
        inv_norm: Tensor,
    ):
        """
        Update weights using projected SGD with normalized gradients.

        The gradient is pre-scaled by inverse state norm, so learning rate = 1.0 (implicit).

        Args:
            x: Input tensor [B x cols]
            error: Error signal [B x rows]
            inv_norm: Inverse normalization factor [B x 1]
        """
        if x.numel() == 0 or error.numel() == 0:
            return

        # Per-sample gradient with normalization (reference: lr * inv_norm * x * error)
        # error: [B, rows], x: [B, cols], inv_norm: [B, 1]
        grad_batched = inv_norm * x * error  # [B, cols]
        # Average over batch
        grad = grad_batched.mean(
            dim=0, keepdim=True
        )  # [1, cols] for rows=1, or [rows, cols] general

        delta = _project_l2_norm(grad * self.lr, self.delta_max_norm)

        self.weight.data.add_(delta)
        self.weight.data = _project_l2_norm(self.weight.data, self.max_norm)
