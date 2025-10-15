from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


@dataclass
class ESNConfig:
    input_size: int
    reservoir_size: int
    output_size: int

    # Reservoir generation
    density: float = 0.02  # fraction of nonzeros in W
    spectral_radius: float = 0.9  # target rho(W)
    input_scale: float = 0.5
    bias_scale: float = 0.0
    leak: float = 0.3  # α in leaky update
    use_bias: bool = False
    add_input_to_state: bool = False  # augment readout with input features

    # Training / numerics
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32

    # Runtime knobs
    clip_state: Optional[float] = None  # optional state clipping (e.g., 5.0)
    state_nonlinearity: str = "tanh"  # "tanh" or "relu"


class ESN(nn.Module):
    """
    Leaky ESN with sparse random reservoir and linear readout.
    State update:
      x_{t+1} = (1-α) x_t + α φ( W x_t + W_in u_t + b ), φ=tanh by default.
    Readout y_t = Wout [x_t, u_t, 1] if add_input_to_state or use_bias is true; else y_t = Wout x_t.
    """

    def __init__(self, cfg: ESNConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        N, Fi = cfg.reservoir_size, cfg.input_size
        self.register_buffer(
            "W",
            self._make_reservoir(
                N=N,
                density=cfg.density,
                target_rho=cfg.spectral_radius,
                device=cfg.device,
                dtype=cfg.dtype,
                seed=cfg.seed,
            ),
            persistent=False,
        )
        self.register_buffer(
            "Win",
            (torch.randn(N, Fi, device=cfg.device, dtype=cfg.dtype) * cfg.input_scale),
            persistent=False,
        )
        if cfg.use_bias:
            self.register_buffer(
                "b",
                (torch.randn(N, device=cfg.device, dtype=cfg.dtype) * cfg.bias_scale),
                persistent=False,
            )
        else:
            self.register_buffer(
                "b",
                torch.zeros(N, device=cfg.device, dtype=cfg.dtype),
                persistent=False,
            )

        # Readout is learned by the trainer; we keep it here for inference convenience.
        aug_dim = N + (Fi if cfg.add_input_to_state else 0) + (1 if cfg.use_bias else 0)
        self.Wout = nn.Parameter(
            torch.zeros(aug_dim, cfg.output_size, device=cfg.device, dtype=cfg.dtype),
            requires_grad=False,
        )

        self._nonlin = torch.tanh if cfg.state_nonlinearity == "tanh" else torch.relu
        self.to(cfg.device, cfg.dtype)

    @staticmethod
    def _make_reservoir(
        N: int,
        density: float,
        target_rho: float,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> Tensor:
        """
        Generate a sparse random reservoir and scale to target spectral radius.
        """
        gen = torch.Generator(device=device).manual_seed(seed)
        mask = torch.rand(N, N, generator=gen, device=device) < float(density)
        W = torch.zeros(N, N, device=device, dtype=dtype)
        if mask.any():
            W[mask] = torch.randn(int(mask.sum().item()), device=device, dtype=dtype)
        # Remove self-loops (optional, but common)
        W.fill_diagonal_(0.0)

        # Scale to target spectral radius (power iteration)
        def _power_radius(A: Tensor, iters: int = 50) -> float:
            v = torch.randn(A.shape[0], device=A.device, dtype=A.dtype)
            v = v / (v.norm() + 1e-12)
            for _ in range(iters):
                v = A @ v
                n = v.norm() + 1e-12
                v = v / n
            lam = (v @ (A @ v)) / (v @ v + 1e-12)
            return float(lam.abs().item())

        rho = _power_radius(W) if mask.any() else 0.0
        if rho > 1e-9:
            W = W * (target_rho / rho)
        return W

    def reset_state(self, batch_size: int) -> Tensor:
        return torch.zeros(
            batch_size,
            self.cfg.reservoir_size,
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def _augment(self, x_t: Tensor, u_t: Optional[Tensor]) -> Tensor:
        """
        Build readout feature vector per batch:
          φ_t = [x_t, u_t? , 1?]
        """
        feats = [x_t]
        if self.cfg.add_input_to_state and u_t is not None:
            feats.append(u_t)
        if self.cfg.use_bias:
            feats.append(
                torch.ones(x_t.shape[0], 1, device=x_t.device, dtype=x_t.dtype)
            )
        return torch.cat(feats, dim=1)  # [B, Daug]

    def step(self, x_t: Tensor, u_t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One-step update. x_t: [B,N], u_t: [B,Fi]
        Returns (x_{t+1}, y_{t+1})
        """
        cfg = self.cfg

        pre = x_t @ self.W.t()  # type: ignore[operator]  # [B,N]
        pre = pre + (u_t @ self.Win.t())  # type: ignore[operator]  # add input drive
        pre = pre + self.b  # add bias (broadcasts)

        x_nonlin = self._nonlin(pre)
        x_next = (1.0 - cfg.leak) * x_t + cfg.leak * x_nonlin
        if cfg.clip_state is not None:
            x_next = torch.clamp(x_next, -cfg.clip_state, cfg.clip_state)
        y = self._augment(x_next, u_t) @ self.Wout  # [B,Fo]
        return x_next, y

    @torch.no_grad()
    def forward_sequence(
        self, U: Tensor, x0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        U: [T,B,Fi] -> states X: [T,B,N], outputs Y: [T,B,Fo] using current Wout.
        """
        T, B, Fi = U.shape
        x = self.reset_state(B) if x0 is None else x0
        X = []
        Y = []
        for t in range(T):
            x, y = self.step(x, U[t])
            X.append(x)
            Y.append(y)
        return torch.stack(X, dim=0), torch.stack(Y, dim=0)

    @torch.no_grad()
    def predict_autoregressive(self, Y0: Tensor, steps: int) -> Tensor:
        """
        Closed-loop rollout when Fi==Fo==Y dims: start from Y0 [B, Fo] or [Fo]
        and feed predictions back as inputs. Returns [steps,B,Fo].
        """
        assert (
            self.cfg.input_size == self.cfg.output_size
        ), "predict_autoregressive requires Fi==Fo"

        if Y0.ndim == 1:
            Y0 = Y0.unsqueeze(0)
        elif Y0.ndim != 2:
            raise ValueError(f"Y0 must be [B, Fo] or [Fo]; got shape {tuple(Y0.shape)}")

        B, Fo = Y0.shape
        x = self.reset_state(B)
        u_t = Y0
        preds = []
        for _ in range(steps):
            x, y = self.step(x, u_t)
            preds.append(y)
            u_t = y
        return torch.stack(preds, dim=0)  # [steps,B,Fo]
