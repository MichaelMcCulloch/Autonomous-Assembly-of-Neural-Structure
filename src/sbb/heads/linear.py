import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sbb.util import _orthogonal, _project_l2_norm
from sbb.const import EPS


class Linear(Module):
    """
    Linear readout layer with NLMS (Normalized Least Mean Squares) adaptation.

    Implements a simple linear transformation y = x @ W^T with online learning
    via the NLMS algorithm. NLMS is a variant of gradient descent that normalizes
    updates by input power, providing robustness to input scale variations.

    Update rule (per sample):
        W ← W - η * (error ⊗ x) / (||x||² + ε)

    where ⊗ denotes outer product. The normalization by ||x||² ensures stable
    learning when input magnitudes vary across time or features.

    Both the weights and per-sample updates are projected to maximum L2 norms
    to prevent parameter explosion during online learning.

    Parameters
    ----------
    rows : int
        Output dimension (number of readout features).
    cols : int
        Input dimension (hidden state size).
    dtype : torch.dtype
        Parameter precision.
    lr : float
        Base learning rate (scaled by 1/batch_size during updates).
    max_norm : float
        Maximum L2 norm for weight matrix.
    delta_max_norm : float
        Maximum L2 norm for weight updates.
    initial_scale : float, optional
        Scale factor for orthogonal initialization. Default 1.0.
    eps : float, optional
        Numerical stability constant. Default EPS.
    name : str, optional
        Identifier for logging/debugging. Default "Linear".

    Attributes
    ----------
    weight : Parameter [rows, cols]
        Weight matrix, initialized orthogonally.

    Methods
    -------
    forward(x) -> y
        Compute linear readout: y = x @ W^T.
    update_from_error(x, error, inv_norm, lr)
        Apply NLMS update from prediction error.
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
        eps: float = EPS,
        name: str = "Linear",
    ):
        super().__init__()

        self.dtype = dtype
        self.rows = int(rows)
        self.cols = int(cols)
        self.max_norm = float(max_norm)
        self.delta_max_norm = float(delta_max_norm)
        self.eps = float(eps)
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
        Compute linear readout.

        Parameters
        ----------
        x : Tensor [B, cols]
            Input features (hidden states).

        Returns
        -------
        Tensor [B, rows]
            Linear transformation y = x @ W^T.
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
        Update weights using NLMS (Normalized Least Mean Squares) rule.

        Computes the gradient of squared error with respect to weights and
        applies it with input-power normalization for stability. The gradient
        is automatically normalized, so learning rate = 1.0 (implicit).

        Update formula:
            ΔW = -(1/B) * Σ_b (error_b ⊗ x_b) * inv_norm_b
            W ← project(W + ΔW, max_norm)

        Parameters
        ----------
        x : Tensor [B, cols]
            Input features (hidden states).
        error : Tensor [B, rows]
            Prediction error (y_pred - y_target).
        inv_norm : Tensor [B, 1]
            Precomputed inverse input power: 1 / (||x||² + ε).
            Can be None, in which case it's computed internally.

        Notes
        -----
        The gradient is computed as x^T @ scaled_error, then transposed.
        Both the update and final weights are L2-norm projected for stability.
        """
        if x.numel() == 0 or error.numel() == 0:
            return

        B = max(1, x.shape[0])

        # NLMS: normalize by input power
        if inv_norm is None:
            denom = torch.sum(x * x, dim=1, keepdim=True) + self.eps
            scaled_err = error / denom
        else:
            scaled_err = error * inv_norm

        # Compute gradient
        grad = x.T @ scaled_err
        grad = grad.T
        grad = grad / B

        delta = _project_l2_norm(grad, self.delta_max_norm)

        self.weight.data.sub_(delta)
        self.weight.data = _project_l2_norm(self.weight.data, self.max_norm)
