import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sbb.util import _orthogonal, _project_l2_norm
from sbb.const import DEVICE, EPS


class CoState(Module):
    """
    Costate (adjoint) estimator for backpropagation-free credit assignment.

    Learns a nonlinear backward mapping from local error signals to hidden
    state gradients: λ̂ = tanh(error @ W^T). This provides a learned approximation
    to backpropagation's error gradient without requiring weight transport
    (symmetric backward connections).

    The mapping is trained via RLS (Recursive Least Squares) with full precision
    matrix tracking (non-diagonal), providing fast adaptation to task structure.
    RLS is an optimal online algorithm for linear regression under the assumption
    of Gaussian noise.

    Update rule (RLS with forgetting):
        K = P @ x / (λ + x^T @ P @ x)                 [Kalman gain]
        W ← W + η * error @ K^T                        [weight update]
        P ← (P - K @ x^T @ P) / λ                      [precision update]

    where λ is the forgetting factor (typically 0.995), allowing adaptation to
    non-stationary tasks while maintaining numerical stability.

    Parameters
    ----------
    input_dim : int
        Dimension of local error signal.
    state_dim : int
        Dimension of hidden state (costate dimension).

    dtype : torch.dtype
        Parameter precision.
    lr : float
        Learning rate for weight updates (scaled by batch size).
    max_norm : float
        Maximum L2 norm for weight matrix.
    delta_max_norm : float
        Maximum L2 norm for weight updates (currently unused).
    forgetting_factor : float, optional
        RLS forgetting factor λ ∈ (0, 1]. Values < 1 allow non-stationarity.
        Default 0.995 (half-life ≈ 138 updates).
    ridge_lambda : float, optional
        Ridge regularization for precision matrix initialization. Default 1e-2.
    eps : float, optional
        Numerical stability constant. Default EPS.
    name : str, optional
        Identifier for logging/debugging. Default "CoState".

    Attributes
    ----------
    weight : Parameter [state_dim, input_dim]
        Costate mapping weights, initialized orthogonally.
    P_full : Tensor [state_dim, input_dim, input_dim]
        RLS precision matrices (one per output dimension).

    Methods
    -------
    forward(local_signal) -> costate
        Compute costate estimate: λ̂ = tanh(error @ W^T).
    align(local_signal, target_costate, inv_norm, lr)
        Update weights via RLS to match target costate (from backprop).
    reset()
        Reinitialize precision matrices to initial state.

    Notes
    -----
    The tanh nonlinearity bounds the costate magnitude, preventing gradient
    explosion. This differs from standard backprop which can amplify errors
    through deep networks. The RLS precision matrix tracks second-order
    statistics, providing faster convergence than first-order methods like SGD.
    """

    weight: Parameter
    P_full: Tensor

    def __init__(
        self,
        *,
        input_dim: int,
        state_dim: int,
        dtype: torch.dtype,
        max_norm: float,
        delta_max_norm: float,
        forgetting_factor: float = 0.995,
        ridge_lambda: float = 1e-2,
        eps: float = EPS,
        name: str = "CoState",
    ):
        super().__init__()
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must be in (0,1].")

        self.dtype = dtype
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.max_norm = float(max_norm)
        self.delta_max_norm = float(delta_max_norm)
        self.lambda_ff = float(forgetting_factor)
        self.ridge_lambda = float(ridge_lambda)
        self.eps = float(eps)
        self.name = name

        # Initialize weight matrix
        self.register_parameter(
            "weight",
            Parameter(
                _orthogonal(
                    dtype=dtype,
                    rows=self.state_dim,
                    columns=self.input_dim,
                ),
                requires_grad=False,
            ),
        )

        # Initialize RLS precision matrix
        eye = torch.eye(self.input_dim, dtype=dtype, device=DEVICE)
        P0 = (1.0 / max(self.ridge_lambda, self.eps)) * eye
        P = P0.unsqueeze(0).expand(self.state_dim, -1, -1).contiguous()
        self.register_buffer("P_full", P.clone(), persistent=True)

    @torch.no_grad()
    def reset(self):
        """
        Reset the RLS precision matrix to initial state.

        Reinitializes the precision matrices to ridge-regularized identity,
        useful for recovering from numerical instability or restarting
        adaptation in a new task.
        """
        self.P_full.zero_()
        self.P_full.view(self.state_dim, self.input_dim, self.input_dim)[:] = torch.eye(
            self.input_dim, dtype=self.P_full.dtype, device=DEVICE
        ) * (1.0 / max(self.ridge_lambda, self.eps))

    @torch.no_grad()
    def forward(self, local_signal: Tensor) -> Tensor:
        """
        Compute costate estimate from local error signal.

        Parameters
        ----------
        local_signal : Tensor [B, input_dim]
            Local error or advantage signal.

        Returns
        -------
        Tensor [B, state_dim]
            Costate estimate λ̂ = tanh(local_signal @ W^T).
        """
        mod = local_signal @ self.weight.T
        return torch.tanh(mod)

    @torch.no_grad()
    def backward(
        self,
        local_signal: Tensor,
        target_costate: Tensor,
    ):
        """
        Align costate prediction to target via RLS (Recursive Least Squares).

        Updates the costate mapping to minimize ||λ̂ - λ*||², where λ* is the
        target costate (typically from backpropagation or a learned backward model).
        The RLS algorithm provides optimal online updates under Gaussian noise,
        adapting the precision matrix to second-order statistics.

        The update accounts for the tanh nonlinearity via its derivative:
            linear_error = (λ̂ - λ*) * (1 - λ̂²)

        This ensures proper credit flow through the saturating nonlinearity.

        Parameters
        ----------
        local_signal : Tensor [B, input_dim]
            Error signal to map backward (e.g., prediction error).
        target_costate : Tensor [B, state_dim]
            Target costate values (λ* = error @ W_forward).
        inv_norm : Tensor [B, 1], optional
            Inverse normalization factor (currently unused). Can be None.

        Notes
        -----
        The method processes each batch element sequentially to update the
        per-output-dimension precision matrices. This is necessary because
        RLS updates depend on previous precision states. The forgetting factor
        λ allows the precision matrix to track non-stationary statistics.

        After updates, weights are projected to max_norm for stability.
        """
        if local_signal.numel() == 0:
            return

        B = max(1, local_signal.shape[0])

        # Forward pass
        pred = self.forward(local_signal)
        err = pred - target_costate
        dpred_dlinear = 1.0 - pred * pred
        lin_error = err * dpred_dlinear

        # RLS update (full matrix) - MUST be sequential, not batched
        # RLS updates are inherently sequential because each update depends on
        # the precision matrix from the previous step. Batching leads to numerical
        # instability and incorrect updates.
        x = local_signal.to(device=DEVICE, dtype=self.dtype)  # [B, input_dim]
        error = (-lin_error).to(device=DEVICE, dtype=self.dtype)  # [B, state_dim]

        # Process each batch element sequentially
        for b in range(B):
            x_b = x[b]  # [input_dim]
            error_b = error[b]  # [state_dim]

            # Compute P @ x for all state dimensions: [state_dim, input_dim]
            Px = torch.einsum("sij,j->si", self.P_full, x_b)

            # Compute x^T @ P @ x for all state dimensions: [state_dim]
            xPx = torch.einsum("i,si->s", x_b, Px)

            # Kalman gain denominator: λ + x^T @ P @ x
            denom = self.lambda_ff + xPx + self.eps  # [state_dim]

            # Kalman gain: K = P @ x / denom -> [state_dim, input_dim]
            K = Px / denom.unsqueeze(1)

            # Weight update: W += error @ K^T
            # For each state dimension s: weight[s, :] += error[s] * K[s, :]
            weight_update = torch.einsum("s,si->si", error_b, K)
            self.weight.data.add_(weight_update)

            # Precision matrix update: P = (P - K @ x^T @ P) / λ
            # x^T @ P: [input_dim] for each state dimension
            xP = torch.einsum("j,sjk->sk", x_b, self.P_full)  # [state_dim, input_dim]

            # K @ xP: outer product for each state dimension
            # P[s] -= K[s, :] @ xP[s, :].T
            KxP = torch.einsum("si,sk->sik", K, xP)  # [state_dim, input_dim, input_dim]

            # Update: P = (P - K @ x^T @ P) / λ
            self.P_full.sub_(KxP).div_(self.lambda_ff)

        # Project weights to max_norm
        self.weight.data = _project_l2_norm(self.weight.data, self.max_norm)
