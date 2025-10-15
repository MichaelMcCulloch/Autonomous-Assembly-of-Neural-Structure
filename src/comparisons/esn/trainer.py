from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor

from .model import ESN


@dataclass
class RidgeConfig:
    ridge_lambda: float = 1e-4
    washout: int = 100
    chunk: int = 4096  # chunk T to avoid big temporary mats


@dataclass
class RLSConfig:
    lambda_reg: float = 1e-2  # initial P0 = (1/lambda_reg) I
    forgetting: float = 1.0  # <=1; <1 for forgetting
    washout: int = 100
    max_cond: float = 1e8  # numeric guard


class ESNTrainer:
    def __init__(self, model: ESN):
        self.model = model
        self.cfg = model.cfg

    @torch.no_grad()
    def collect_features_targets(
        self, U: Tensor, Y: Tensor, washout: int
    ) -> Tuple[Tensor, Tensor]:
        """
        U: [T,B,Fi], Y: [T,B,Fo]
        Returns Φ: [T_eff*B, Daug], T: [T_eff*B, Fo]
        """
        T, B, Fi = U.shape
        X_states, _ = self.model.forward_sequence(U)  # [T,B,N]
        feats_list = []
        tgts_list = []
        for t in range(washout, T):
            x_t = X_states[t]  # [B,N]
            u_t = U[t]
            phi = self.model._augment(x_t, u_t)  # [B,Daug]
            feats_list.append(phi)
            tgts_list.append(Y[t])
        Φ = (
            torch.cat(feats_list, dim=0)
            if feats_list
            else torch.empty(0, device=self.cfg.device, dtype=self.cfg.dtype)
        )
        Tgt = (
            torch.cat(tgts_list, dim=0)
            if tgts_list
            else torch.empty(0, device=self.cfg.device, dtype=self.cfg.dtype)
        )
        return Φ, Tgt

    @torch.no_grad()
    def fit_ridge(self, U: Tensor, Y: Tensor, *, rc: RidgeConfig) -> Dict[str, float]:
        """
        Closed-form ridge on readout using a robust solver (float64 accumulations, jittered Cholesky).
        Used for driven tasks (NARMA).
        """
        Φ, Tgt = self.collect_features_targets(U, Y, washout=rc.washout)
        if Φ.numel() == 0:
            return {"status": 0, "solved": 0, "reason": "empty features"}  # type: ignore[dict-item]

        device = self.cfg.device
        Φ64 = Φ.to(device=device, dtype=torch.float64, copy=True)
        T64 = Tgt.to(device=device, dtype=torch.float64, copy=True)
        n = Φ64.shape[0]
        d = Φ64.shape[1]
        lam = float(max(rc.ridge_lambda, 1e-12))

        def _cholesky_solve_spd(A: Tensor, B: Tensor) -> Tensor:
            jitter = 0.0
            for k in range(7):
                try:
                    A_j = A.clone() if jitter > 0 else A
                    if jitter > 0:
                        A_j.diagonal().add_(jitter)
                    A_j = 0.5 * (A_j + A_j.T)
                    L = torch.linalg.cholesky(A_j)
                    X = torch.cholesky_solve(B, L)
                    return X
                except RuntimeError:
                    jitter = 10.0 ** (k - 8)  # 1e-8..1e-2
                    continue
            try:
                return torch.linalg.solve(0.5 * (A + A.T), B)
            except RuntimeError:
                return torch.linalg.pinv(0.5 * (A + A.T)) @ B

        if n >= d:
            # PRIMAL: (ΦᵀΦ + λI)W = ΦᵀT
            G = torch.zeros(d, d, device=device, dtype=torch.float64)
            R = torch.zeros(d, self.cfg.output_size, device=device, dtype=torch.float64)
            chunk = max(1, int(rc.chunk))
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                Φs = Φ64[s:e]
                G.add_(Φs.T @ Φs)
                R.add_(Φs.T @ T64[s:e])
            G.diagonal().add_(lam)
            W64 = _cholesky_solve_spd(G, R)
        else:
            # DUAL: A = (ΦΦᵀ + λI)^{-1} T, W = Φᵀ A
            K = Φ64 @ Φ64.T
            K = 0.5 * (K + K.T)
            K.diagonal().add_(lam)
            A = _cholesky_solve_spd(K, T64)  # [n,Fo]
            W64 = Φ64.T @ A  # [d,Fo]

        W = W64.to(self.cfg.dtype)
        self.model.Wout.data.copy_(W)
        return {"status": 1, "solved": 1, "mode": ("primal" if n >= d else "dual")}  # type: ignore[dict-item]

    @torch.no_grad()
    def fit_rls_streaming(
        self, U: Tensor, Y: Tensor, *, rc: RLSConfig
    ) -> Dict[str, float]:
        """
        Recursive Least Squares on readout; processes timesteps sequentially (after washout),
        optionally with forgetting factor < 1.0. Uses teacher forcing inputs U.
        """
        T, B, _ = U.shape
        Daug = self.model.Wout.shape[0]
        W = self.model.Wout.data  # [Daug,Fo]
        P = (
            torch.eye(Daug, device=self.cfg.device, dtype=self.cfg.dtype)
            / rc.lambda_reg
        )

        X_states, _ = self.model.forward_sequence(U)  # [T,B,N]
        for t in range(rc.washout, T):
            x_t = X_states[t]
            u_t = U[t]
            y_t = Y[t]
            phi = self.model._augment(x_t, u_t)  # [B, Daug]
            for b in range(B):
                f = phi[b].unsqueeze(1)  # [D,1]
                y = y_t[b].unsqueeze(1)  # [Fo,1]
                Pf = P @ f
                denom = rc.forgetting + (f.t() @ Pf).item()
                if denom <= 0 or not torch.isfinite(torch.tensor(denom)):
                    continue
                K = Pf / denom
                err = y - (W.t() @ f)
                W.add_(K @ err.t())
                P = (P - K @ f.t() @ P) / rc.forgetting
                if torch.linalg.cond(P) > rc.max_cond or not torch.isfinite(P).all():
                    P = (
                        torch.eye(Daug, device=self.cfg.device, dtype=self.cfg.dtype)
                        / rc.lambda_reg
                    )
        self.model.Wout.data.copy_(W)
        return {"status": 1, "solved": 1}

    @torch.no_grad()
    def fit_rls_scheduled_sampling(
        self,
        U: Tensor,  # [T,B,Fi] with Fi == Fo
        Y: Tensor,  # [T,B,Fo]
        *,
        rc: RLSConfig,
        p_init: float = 1.0,
        p_final: float = 0.1,
        schedule: str = "exponential",
        jitter: float = 0.0,
    ) -> Dict[str, float]:
        """
        Streaming RLS with scheduled sampling for Fi==Fo tasks (MG, Lorenz).
        At each step t, feed either ground-truth u[t] or the model's last prediction
        (with prob 1-p_tf) into the reservoir, then update Wout on the target Y[t].
        """
        T, B, Fi = U.shape
        _, _, Fo = Y.shape
        assert (
            Fi == Fo
        ), "fit_rls_scheduled_sampling requires Fi == Fo (closed-loop tasks)."

        device, dtype = self.cfg.device, self.cfg.dtype
        U = U.to(device, dtype)
        Y = Y.to(device, dtype)

        Daug = self.model.Wout.shape[0]
        W = self.model.Wout.data  # [Daug,Fo]
        P = torch.eye(Daug, device=device, dtype=dtype) / rc.lambda_reg

        x = self.model.reset_state(B)  # [B,N]
        y_prev = torch.zeros(B, Fo, device=device, dtype=dtype)  # seed if needed

        total_steps = max(1, T - rc.washout)

        def p_tf_for_step(g: int) -> float:
            if schedule == "exponential":
                if p_init <= 0:
                    base = 0.0
                else:
                    decay = (p_final / max(1e-12, p_init)) ** (g / total_steps)
                    base = p_init * decay
            else:
                base = max(p_final, p_init * (1.0 - g / total_steps))
            if jitter > 0:
                base = float(
                    torch.clamp(
                        torch.tensor(base)
                        + torch.empty((), device=device, dtype=dtype).uniform_(
                            -jitter, jitter
                        ),
                        0.0,
                        1.0,
                    )
                )
            return float(base)

        for t in range(T):
            if t == 0:
                u_in = U[t]
            else:
                p_tf = p_tf_for_step(max(0, t - rc.washout))
                mask = (
                    torch.rand(B, 1, device=device, dtype=dtype)
                    < torch.tensor(p_tf, device=device, dtype=dtype)
                ).to(dtype)
                u_in = mask * U[t] + (1.0 - mask) * y_prev

            x, y_hat = self.model.step(x, u_in)  # y_hat corresponds to target Y[t]
            if t >= rc.washout:
                phi = self.model._augment(x, u_in)  # [B,Daug]
                y_t = Y[t]
                for b in range(B):
                    f = phi[b].unsqueeze(1)  # [D,1]
                    y_true = y_t[b].unsqueeze(1)  # [Fo,1]
                    Pf = P @ f
                    denom = rc.forgetting + (f.t() @ Pf).item()
                    if denom <= 0 or not torch.isfinite(torch.tensor(denom)):
                        continue
                    K = Pf / denom
                    err = y_true - (W.t() @ f)
                    W.add_(K @ err.t())
                    P = (P - K @ f.t() @ P) / rc.forgetting
                    if (
                        torch.linalg.cond(P) > rc.max_cond
                        or not torch.isfinite(P).all()
                    ):
                        P = torch.eye(Daug, device=device, dtype=dtype) / rc.lambda_reg

            y_prev = y_hat.detach()

        self.model.Wout.data.copy_(W)
        return {"status": 1, "solved": 1}
