from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from torch import nn, Tensor
from torch.optim import AdamW

from .model import LSTMRegressor


@dataclass
class TrainSnapshot:
    W_init: torch.Tensor
    W_early: torch.Tensor
    W_final: torch.Tensor


class SequenceTrainer:
    """
    Trainer for LSTMRegressor with:
      - Teacher forcing (Driven tasks)
      - Multi-step scheduled sampling (Closed-loop tasks)
      - AdamW + gradient clipping
    """

    def __init__(self, model: LSTMRegressor):
        self.model = model
        self.cfg = model.cfg
        self.crit = nn.MSELoss(reduction="mean")
        self.opt = AdamW(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    def fit_teacher_forcing(
        self,
        inputs: Tensor,  # [T,B,Fi]
        targets: Tensor,  # [T,B,Fo]
        *,
        n_epochs: Optional[int] = None,
        washout: Optional[int] = None,
        tbptt_steps: Optional[int] = None,
        early_fraction: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[TrainSnapshot, Dict[str, float]]:
        cfg = self.cfg
        T, B, Fi = inputs.shape
        _, _, Fo = targets.shape
        n_epochs = cfg.n_epochs if n_epochs is None else n_epochs
        wash = cfg.washout if washout is None else max(0, int(washout))
        K = cfg.tbptt_steps if tbptt_steps is None else max(1, int(tbptt_steps))
        self.model.train()

        hc: Optional[Tuple[Tensor, Tensor]] = None
        W_init = self.model.get_recurrent_weights().detach().clone()

        total_steps = n_epochs * T
        early_at = max(1, int(early_fraction * total_steps))
        seen_steps = 0
        W_early = None

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            t = 0
            while t < T:
                t_end = min(T, t + K)
                x_chunk = inputs[t:t_end].to(cfg.device, cfg.dtype)
                y_chunk = targets[t:t_end].to(cfg.device, cfg.dtype)

                self.opt.zero_grad(set_to_none=True)
                y_pred, hc_next = self.model.forward(x_chunk, hc)
                hc = (hc_next[0].detach(), hc_next[1].detach())

                losses = []
                for i_rel in range(y_pred.shape[0]):
                    t_global = t + i_rel
                    if t_global >= wash:
                        losses.append(self.crit(y_pred[i_rel], y_chunk[i_rel]))
                if losses:
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.grad_clip
                        )
                    self.opt.step()
                    epoch_loss += float(loss.detach().cpu().item())

                seen_steps += t_end - t
                if (W_early is None) and (seen_steps >= early_at):
                    W_early = self.model.get_recurrent_weights().detach().clone()

                t = t_end

            if verbose:
                try:
                    from tqdm.auto import tqdm

                    tqdm.write(
                        f"[LSTM-BPTT] Epoch {epoch+1}/{n_epochs}  loss={epoch_loss:.6f}"
                    )
                except ImportError:
                    print(
                        f"[LSTM-BPTT] Epoch {epoch+1}/{n_epochs}  loss={epoch_loss:.6f}"
                    )

        if W_early is None:
            W_early = self.model.get_recurrent_weights().detach().clone()
        W_final = self.model.get_recurrent_weights().detach().clone()

        snap = TrainSnapshot(W_init=W_init, W_early=W_early, W_final=W_final)
        stats = {"final_epoch_loss": epoch_loss}
        return snap, stats

    def fit_multistep_scheduled_sampling(
        self,
        inputs: Tensor,  # [T,B,Fi]
        targets: Tensor,  # [T,B,Fo]
        *,
        n_epochs: Optional[int] = None,
        washout: Optional[int] = None,
        tbptt_steps: Optional[int] = None,
        p_init: float = 1.0,
        p_final: float = 0.1,
        schedule: str = "exponential",
        jitter: float = 0.0,
        H_max: int = 16,
        verbose: bool = True,
    ) -> Dict[str, float]:
        cfg = self.cfg
        T, B, Fi = inputs.shape
        _, _, Fo = targets.shape
        assert Fi == Fo, "fit_multistep_scheduled_sampling requires Fi == Fo."
        device, dtype = cfg.device, cfg.dtype
        base_inputs = inputs.to(device, dtype)
        base_targets = targets.to(device, dtype)

        n_epochs = cfg.n_epochs if n_epochs is None else n_epochs
        wash = cfg.washout if washout is None else max(0, int(washout))
        K = cfg.tbptt_steps if tbptt_steps is None else max(1, int(tbptt_steps))

        total_steps_all = n_epochs * T

        def p_tf_for_step(g: int) -> float:
            if schedule == "exponential":
                if p_init <= 0:
                    return 0.0
                decay = (p_final / p_init) ** (g / max(1, total_steps_all))
                base = p_init * decay
            else:
                base = max(p_final, p_init * (1.0 - g / max(1, total_steps_all)))
            if jitter > 0:
                base = float(
                    torch.clamp(
                        torch.tensor(base) + torch.empty(()).uniform_(-jitter, jitter),
                        0.0,
                        1.0,
                    )
                )
            return float(base)

        epoch_last = 0.0
        self.model.train(True)
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            loss_bucket = []
            hc: Optional[Tuple[Tensor, Tensor]] = None
            for t in range(T):
                H = 1 + int(torch.randint(low=0, high=max(1, H_max), size=(1,)).item())

                L = min(K, t + 1)
                start = t + 1 - L
                ctx_inputs = base_inputs[start : t + 1].clone()

                last_pred: Optional[Tensor] = None
                for h in range(H):
                    step_idx = t + h
                    if step_idx >= T:
                        break

                    if step_idx > 0 and last_pred is not None and step_idx >= wash:
                        p_tf = p_tf_for_step(epoch * T + t)
                        use_truth = torch.rand((), device=device).item() < p_tf
                        if not use_truth:
                            ctx_inputs[-1] = last_pred.detach()

                    y_step, hc = self.model.one_step(ctx_inputs[-1], hc)

                    if step_idx >= wash:
                        loss = self.crit(y_step, base_targets[step_idx])
                        loss_bucket.append(loss)

                    last_pred = y_step
                    if h < H - 1:
                        next_in = base_inputs[min(step_idx + 1, T - 1)]
                        if L < K:
                            ctx_inputs = torch.cat(
                                [ctx_inputs, next_in.unsqueeze(0)], dim=0
                            )
                            L += 1
                        else:
                            ctx_inputs = torch.cat(
                                [ctx_inputs[1:], next_in.unsqueeze(0)], dim=0
                            )

                if ((t + 1) % K == 0) or (t == T - 1):
                    if loss_bucket:
                        self.opt.zero_grad(set_to_none=True)
                        train_loss = torch.stack(loss_bucket).mean()
                        train_loss.backward()
                        if cfg.grad_clip and cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), cfg.grad_clip
                            )
                        self.opt.step()
                        epoch_loss += float(train_loss.detach().cpu().item())
                        loss_bucket.clear()
                    if hc is not None:
                        hc = (hc[0].detach(), hc[1].detach())

            epoch_last = epoch_loss
            if verbose:
                try:
                    from tqdm.auto import tqdm

                    tqdm.write(
                        f"[LSTM-BPTT MS] Epoch {epoch+1}/{n_epochs} loss={epoch_loss:.6f}"
                    )
                except ImportError:
                    print(
                        f"[LSTM-BPTT MS] Epoch {epoch+1}/{n_epochs} loss={epoch_loss:.6f}"
                    )

        return {"final_epoch_loss": epoch_last}
