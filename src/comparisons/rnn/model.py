from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


@dataclass
class RNNBPTTConfig:
    input_size: int
    hidden_size: int
    output_size: int
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    n_epochs: int = 1
    tbptt_steps: int = 128
    washout: int = 100
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    use_tanh_output: bool = True


class SimpleRNNRegressor(nn.Module):
    def __init__(self, cfg: RNNBPTTConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rnn = nn.RNN(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=False,
            bias=True,
        )
        self.readout = nn.Linear(cfg.hidden_size, cfg.output_size)
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.to(cfg.device, cfg.dtype)

    def forward(
        self, seq: Tensor, h0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        out, hT = self.rnn(seq.to(self.cfg.device, self.cfg.dtype), h0)
        y = self.readout(out)
        if self.cfg.use_tanh_output:
            y = torch.tanh(y)
        return y, hT

    def one_step(
        self, x_t: Tensor, h_t: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        y_seq, hT = self.forward(x_t.unsqueeze(0), h_t)
        return y_seq[0], hT

    @torch.no_grad()
    def rollout_autoregressive(
        self, x0: Tensor, steps: int, h0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, Fi = x0.shape  # must be [B, Fi]
        y_list = []
        h = h0
        x_t = x0.to(self.cfg.device, self.cfg.dtype)
        for _ in range(steps):
            y_t, h = self.one_step(x_t, h)
            y_list.append(y_t)
            x_t = y_t
        Y = torch.stack(y_list, dim=0)  # [steps, B, Fo]
        return Y, h

    def get_recurrent_weights(self) -> Tensor:
        return getattr(self.rnn, "weight_hh_l0").to(self.cfg.device, self.cfg.dtype)
