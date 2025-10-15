from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


@dataclass
class LSTMBPTTConfig:
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


class LSTMRegressor(nn.Module):
    """
    1-layer LSTM + linear readout; outputs y_t = tanh(W h_t) by default (configurable).
    API mirrors the SimpleRNNRegressor used in the RNN baseline.
    """

    def __init__(self, cfg: LSTMBPTTConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=False,
            bias=True,
        )
        self.readout = nn.Linear(cfg.hidden_size, cfg.output_size)

        # Initialize LSTM weights: input with Xavier; recurrent with orthogonal; set forget bias to +1
        for name, p in self.lstm.named_parameters():
            if "weight_ih_l0" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh_l0" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias_ih_l0" in name:
                nn.init.zeros_(p)
                # Set forget gate bias (+1) to help remembering
                # LSTM bias layout: [i, f, g, o] chunks of hidden_size
                hidden = cfg.hidden_size
                p.data[hidden : 2 * hidden].fill_(1.0)
            elif "bias_hh_l0" in name:
                nn.init.zeros_(p)

        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.to(cfg.device, cfg.dtype)

    def forward(
        self, seq: Tensor, hc0: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        seq: [T, B, Fi]; hc0: (h0,c0) each [1, B, H] or None
        returns:
          y: [T, B, Fo], (hT, cT)
        """
        out, (hT, cT) = self.lstm(seq.to(self.cfg.device, self.cfg.dtype), hc0)
        y = self.readout(out)
        if self.cfg.use_tanh_output:
            y = torch.tanh(y)
        return y, (hT, cT)

    def one_step(
        self, x_t: Tensor, hc_t: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        x_t: [B, Fi]
        returns y_t [B, Fo], (h,c) each [1,B,H]
        """
        y_seq, (hT, cT) = self.forward(x_t.unsqueeze(0), hc_t)
        return y_seq[0], (hT, cT)

    @torch.no_grad()
    def rollout_autoregressive(
        self, x0: Tensor, steps: int, hc0: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Start from x0 [B, Fi], autoregress for 'steps' using own predictions.
        Returns Y [steps, B, Fo], (h,c).
        """
        B, Fi = x0.shape
        y_list = []
        hc = hc0
        x_t = x0.to(self.cfg.device, self.cfg.dtype)
        for _ in range(steps):
            y_t, hc = self.one_step(x_t, hc)
            y_list.append(y_t)
            x_t = y_t
        Y = torch.stack(y_list, dim=0)
        return Y, hc

    def get_recurrent_weights(self) -> Tensor:
        # Return the main recurrent matrix for inspection
        return getattr(self.lstm, "weight_hh_l0").to(self.cfg.device, self.cfg.dtype)
