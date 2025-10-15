from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn, Tensor
from torch.optim import AdamW

from .model import (
    CausalTransformerEncoder,
    TransformerEncoderDecoder,
    TransformerConfig,
)


@dataclass
class TrainSnapshot:
    dummy: int = 0  # placeholder for parity with other baselines


class TransformerSequenceTrainer:
    """
    - For Fi==Fo tasks (MG, Lorenz): multi-step scheduled sampling with causal encoder-only transformer.
    - For NARMA10 (Driven): teacher forcing with causal encoder.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        model_enc: Optional[CausalTransformerEncoder] = None,
        model_encdec: Optional[TransformerEncoderDecoder] = None,
    ):
        self.cfg = cfg
        self.model_enc = model_enc  # used for Fi==Fo tasks & driven NARMA
        self.model_encdec = model_encdec  # unused in this configuration
        self.crit = nn.MSELoss(reduction="mean")
        params = []
        if self.model_enc is not None:
            params += list(self.model_enc.parameters())
        if self.model_encdec is not None:
            params += list(self.model_encdec.parameters())
        self.opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------- Encoder-only: Teacher Forcing (Driven tasks like NARMA) ----------

    def fit_teacher_forcing(
        self,
        inputs: Tensor,  # [T,B,Fi]
        targets: Tensor,  # [T,B,Fo] (aligned: inputs[t] -> targets[t])
        *,
        n_epochs: Optional[int] = None,
        washout: Optional[int] = None,
        tbptt_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        assert self.model_enc is not None, "Encoder-only transformer not initialized."
        cfg = self.cfg
        T, B, Fi = inputs.shape
        _, _, Fo = targets.shape
        n_epochs = cfg.n_epochs if n_epochs is None else n_epochs
        wash = cfg.washout if washout is None else max(0, int(washout))
        K = cfg.tbptt_steps if tbptt_steps is None else max(1, int(tbptt_steps))
        self.model_enc.train(True)

        epoch_last = 0.0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            loss_bucket = []
            for t in range(T):
                L = min(K, t + 1)
                start = t + 1 - L
                ctx = inputs[start : t + 1].to(cfg.device, cfg.dtype)  # [L,B,Fi]
                y_pred = self.model_enc.forward_last(ctx)  # [B,Fo]
                if t >= wash:
                    loss = self.crit(y_pred, targets[t].to(cfg.device, cfg.dtype))
                    loss_bucket.append(loss)
                if ((t + 1) % K == 0) or (t == T - 1):
                    if loss_bucket:
                        self.opt.zero_grad(set_to_none=True)
                        train_loss = torch.stack(loss_bucket).mean()
                        train_loss.backward()
                        if cfg.grad_clip and cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model_enc.parameters(), cfg.grad_clip
                            )
                        self.opt.step()
                        epoch_loss += float(train_loss.detach().cpu().item())
                        loss_bucket.clear()
            epoch_last = epoch_loss
            if verbose:
                try:
                    from tqdm.auto import tqdm

                    tqdm.write(
                        f"[Transformer TF] Epoch {epoch+1}/{n_epochs} loss={epoch_loss:.6f}"
                    )
                except ImportError:
                    print(
                        f"[Transformer TF] Epoch {epoch+1}/{n_epochs} loss={epoch_loss:.6f}"
                    )

        return {"final_epoch_loss": epoch_last}

    # ---------- Encoder-only: Multi-step scheduled sampling (Fi==Fo) ----------

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
        assert self.model_enc is not None, "Encoder-only transformer not initialized."
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
        self.model_enc.train(True)
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            loss_bucket = []
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

                    y_step = self.model_enc.forward_last(ctx_inputs)  # [B,Fo]
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
                                self.model_enc.parameters(), cfg.grad_clip
                            )
                        self.opt.step()
                        epoch_loss += float(train_loss.detach().cpu().item())
                        loss_bucket.clear()

            epoch_last = epoch_loss
            if verbose:
                try:
                    from tqdm.auto import tqdm

                    tqdm.write(
                        f"[Transformer MS] Epoch {epoch+1}/{n_epochs} loss={epoch_loss:.6f}"
                    )
                except ImportError:
                    print(
                        f"[Transformer MS] Epoch {epoch+1}/{n_epochs} loss={epoch_loss:.6f}"
                    )

        return {"final_epoch_loss": epoch_last}

    # ---------- Encoder-Decoder: NARMA Teacher Forcing (UNUSED in this setup) ----------

    def fit_narma_encdec_teacher_forcing(
        self,
        u_series: Tensor,  # [T,B,1]
        y_series: Tensor,  # [T,B,1]
        *,
        n_epochs: Optional[int] = None,
        washout: Optional[int] = None,
        tbptt_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        # This method is kept for completeness but is not used in the unified exogenous NARMA setup.
        raise NotImplementedError(
            "Encoder-Decoder training is disabled for unified NARMA benchmark."
        )
