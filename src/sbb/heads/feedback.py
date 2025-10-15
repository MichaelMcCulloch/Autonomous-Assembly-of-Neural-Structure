import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sbb.util import _orthogonal, _project_l2_norm
from sbb.const import EPS


class Feedback(Module):
    """
    A simple, slowly adapting linear feedback projection for credit assignment.

    This module implements a stable feedback mechanism that projects a local
    error/advantage signal into the high-dimensional state space of the neurons.
    It serves as a simpler, more stable alternative to the fast-adapting CoState
    head, making it particularly suitable for high-variance environments like RL.

    The projection is a simple linear transformation:
        variational_signal = local_signal @ W^T

    The weight matrix `W` (named `B_err_proj` for consistency with other models)
    is adapted slowly via a simple normalized SGD rule. This prevents the feedback
    signal from oscillating wildly in response to noisy targets, promoting stable
    learning in the recurrent network.

    Parameters
    ----------
    input_dim : int
        Dimension of local error signal (typically 1 for scalar advantage).
    state_dim : int
        Dimension of the neural state space (N).
    dtype : torch.dtype
        Parameter precision.
    lr : float
        Learning rate for adapting the projection matrix.
    max_norm : float
        Maximum L2 norm for the projection matrix.
    delta_max_norm : float
        Maximum L2 norm for weight updates.
    initial_scale : float
        Scale factor for orthogonal initialization.
    eps : float, optional
        Numerical stability constant. Default EPS.
    name : str, optional
        Identifier for logging/debugging. Default "StaticFeedback".
    """

    weight: Parameter

    def __init__(
        self,
        *,
        input_dim: int,
        state_dim: int,
        dtype: torch.dtype,
        lr: float = 1.0,
        max_norm: float,
        delta_max_norm: float,
        initial_scale: float = 1.0,
        eps: float = EPS,
        name: str = "StaticFeedback",
    ):
        super().__init__()

        self.dtype = dtype
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.lr = float(lr)
        self.max_norm = float(max_norm)
        self.delta_max_norm = float(delta_max_norm)
        self.eps = float(eps)
        self.name = name

        # Initialize the projection matrix (equivalent to B_err_proj)
        self.register_parameter(
            "weight",
            Parameter(
                _orthogonal(
                    dtype=dtype,
                    rows=self.state_dim,
                    columns=self.input_dim,
                )
                * initial_scale,
                requires_grad=False,
            ),
        )

    @torch.no_grad()
    def reset(self):
        """
        Reinitialize the projection matrix to orthogonal state.

        Resets weights while preserving the current norm, useful for
        recovering from numerical instability or restarting adaptation.
        """
        initial_weights = _orthogonal(
            dtype=self.dtype,
            rows=self.state_dim,
            columns=self.input_dim,
        ) * torch.linalg.norm(
            self.weight.data
        )  # Preserve norm
        self.weight.data.copy_(initial_weights)

    @torch.no_grad()
    def forward(self, local_signal: Tensor) -> Tensor:
        """
        Projects the local signal into the state space.
        Equivalent to: advantage.unsqueeze(1) @ self.weight.T
        """
        return local_signal @ self.weight.T

    @torch.no_grad()
    def backward(
        self,
        local_signal: Tensor,
        target_costate: Tensor,
    ):
        """
        Slowly adapts the projection matrix towards a target direction.

        This method is named `align` to match the CoState API, but it performs
        a simple, slow SGD update rather than a fast RLS alignment. It updates
        the projection `W` to better map `local_signal` to `target_costate`.

        Update rule:
            error = (local_signal @ W^T) - target_costate
            grad = error^T @ local_signal
            ΔW = -lr * grad

        Parameters
        ----------
        local_signal : Tensor [B, input_dim]
            The input to the projection (e.g., advantage signal).
        target_costate : Tensor [B, state_dim]
            The target output vector (e.g., value_error * value_head.weight).
        """
        if local_signal.numel() == 0 or self.lr <= 0:
            return

        B = max(1, local_signal.shape[0])

        # Get the current projection
        pred_costate = self.forward(local_signal)

        # Calculate the error in the state space
        error = pred_costate - target_costate.detach()

        # Calculate the gradient for the projection weights
        # grad = error^T @ local_signal
        grad = error.T @ local_signal
        grad = grad / B

        # Project the update to prevent explosions
        delta = _project_l2_norm(grad * self.lr, self.delta_max_norm)

        # Apply the update
        self.weight.data.sub_(delta)

        # Project the final weights to maintain stability
        self.weight.data = _project_l2_norm(self.weight.data, self.max_norm)
