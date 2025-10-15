import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor


def _sinusoidal_positional_encoding(max_len: int, d_model: int, device, dtype):
    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
    position = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, d_model]


def _alibi_slopes(n_heads: int, device, dtype):
    # From ALiBi paper/code: head-specific slopes spread roughly geometrically
    def get_slopes(n):
        def power_of_2(n):
            return 2 ** math.floor(math.log2(n))

        m = power_of_2(n)
        slopes = [2 ** (-8.0 * i / m) for i in range(m)]
        if m < n:
            extra = [slopes[-1] * (0.5 ** (i + 1)) for i in range(n - m)]
            slopes = slopes + extra
        return torch.tensor(slopes, device=device, dtype=dtype)

    return get_slopes(n_heads).view(n_heads, 1, 1)  # [H,1,1]


def _build_causal_mask(L: int, device, dtype, fill_value: float = float("-inf")):
    # Upper triangular (future) masked to -inf; allowed entries are 0
    mask = torch.zeros(L, L, device=device, dtype=dtype)
    mask[
        (torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1))
    ] = fill_value
    return mask  # [L, L], additive mask


def _build_alibi_bias(L: int, n_heads: int, device, dtype):
    # For allowed positions j<=i: bias[h,i,j] = -slope[h]*(i-j); future positions handled by causal mask
    i_idx = torch.arange(L, device=device, dtype=dtype).view(L, 1)
    j_idx = torch.arange(L, device=device, dtype=dtype).view(1, L)
    dist = (i_idx - j_idx).clamp(min=0)  # [L,L], >=0 for allowed
    slopes = _alibi_slopes(n_heads, device, dtype)  # [H,1,1]
    bias = -slopes * dist  # [H,L,L]
    return bias


@dataclass
class TransformerConfig:
    # Shared
    input_size: int
    output_size: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float = 0.0
    pos_encoding: str = "sinusoidal"  # 'sinusoidal' | 'alibi'
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    n_epochs: int = 1
    tbptt_steps: int = 128
    washout: int = 100
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    use_tanh_output: bool = True
    max_len: int = 16384

    # Encoder-Decoder (for NARMA10)
    enc_input_size: int = 1  # u_t dimension
    dec_input_size: int = 1  # y_t dimension (Fi==Fo==1 for narma)
    use_encdec_for_narma: bool = True


class CausalTransformerEncoder(nn.Module):
    """
    Encoder-only causal transformer with either sinusoidal PE or ALiBi bias.
    Expects x: [L, B, Fi]; returns y: [L, B, Fo].
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.in_proj = nn.Linear(cfg.input_size, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=False,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.out_proj = nn.Linear(cfg.d_model, cfg.output_size)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        if cfg.pos_encoding == "sinusoidal":
            self.register_buffer(
                "pe",
                _sinusoidal_positional_encoding(
                    cfg.max_len, cfg.d_model, cfg.device, cfg.dtype
                ),
                persistent=False,
            )
        else:
            self.register_buffer("pe", None, persistent=False)

        self.to(cfg.device, cfg.dtype)

    def forward(self, seq: Tensor) -> Tensor:
        """
        seq: [L, B, Fi]
        """
        L, B, _ = seq.shape
        x = self.in_proj(seq.to(self.cfg.device, self.cfg.dtype))  # [L,B,d]
        if self.cfg.pos_encoding == "sinusoidal":
            x = x + self.pe[:L].unsqueeze(1)  # type: ignore[index]
            attn_mask = _build_causal_mask(L, self.cfg.device, x.dtype)
        else:
            # ALiBi: no token PE; provide additive per-head bias in attn_mask
            base = _build_causal_mask(L, self.cfg.device, x.dtype)
            bias = _build_alibi_bias(
                L, self.cfg.nhead, self.cfg.device, x.dtype
            )  # [H,L,L]
            attn_mask = bias.repeat(B, 1, 1) + base.unsqueeze(0).repeat(
                B * self.cfg.nhead, 1, 1
            )

        h = self.encoder(x, mask=attn_mask)
        y = self.out_proj(h)
        if self.cfg.use_tanh_output:
            y = torch.tanh(y)
        return y

    def forward_last(self, seq: Tensor) -> Tensor:
        # seq: [L,B,Fi] -> return last step [B,Fo]
        y = self.forward(seq)
        return y[-1]


class TransformerEncoderDecoder(nn.Module):
    """
    Encoder-Decoder transformer for exogenous-input forecasting (e.g., NARMA10).
    Encoder ingests u context; decoder autoregresses y with cross-attention to encoder memory.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.enc_in = nn.Linear(cfg.enc_input_size, cfg.d_model)
        self.dec_in = nn.Linear(cfg.dec_input_size, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=False,
            norm_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=False,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_layers)
        self.out_proj = nn.Linear(cfg.d_model, cfg.output_size)

        for lin in (self.enc_in, self.dec_in, self.out_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        if cfg.pos_encoding == "sinusoidal":
            self.register_buffer(
                "pe_enc",
                _sinusoidal_positional_encoding(
                    cfg.max_len, cfg.d_model, cfg.device, cfg.dtype
                ),
                persistent=False,
            )
            self.register_buffer(
                "pe_dec",
                _sinusoidal_positional_encoding(
                    cfg.max_len, cfg.d_model, cfg.device, cfg.dtype
                ),
                persistent=False,
            )
        else:
            self.register_buffer("pe_enc", None, persistent=False)
            self.register_buffer("pe_dec", None, persistent=False)

        self.to(cfg.device, cfg.dtype)

    def encode(self, src: Tensor) -> Tensor:
        # src: [Ls,B,Fe]
        Ls, B, _ = src.shape
        s = self.enc_in(src.to(self.cfg.device, self.cfg.dtype))
        if self.cfg.pos_encoding == "sinusoidal":
            s = s + self.pe_enc[:Ls].unsqueeze(1)  # type: ignore[index]
            mask_s = _build_causal_mask(Ls, self.cfg.device, s.dtype)
            mem = self.encoder(s, mask=mask_s)
        else:
            base = _build_causal_mask(Ls, self.cfg.device, s.dtype)
            bias = _build_alibi_bias(
                Ls, self.cfg.nhead, self.cfg.device, s.dtype
            )  # [H,Ls,Ls]
            attn_mask = bias.repeat(B, 1, 1) + base.unsqueeze(0).repeat(
                B * self.cfg.nhead, 1, 1
            )
            mem = self.encoder(s, mask=attn_mask)
        return mem  # [Ls,B,d]

    def decode_last(self, mem: Tensor, tgt_ctx: Tensor) -> Tensor:
        # mem: [Ls,B,d], tgt_ctx: [Lt,B,Fo_in] (e.g., previous y tokens)
        Lt, B, _ = tgt_ctx.shape
        t = self.dec_in(tgt_ctx.to(self.cfg.device, self.cfg.dtype))
        if self.cfg.pos_encoding == "sinusoidal":
            t = t + self.pe_dec[:Lt].unsqueeze(1)  # type: ignore[index]
            mask_t = _build_causal_mask(Lt, self.cfg.device, t.dtype)
            out = self.decoder(t, mem, tgt_mask=mask_t)
        else:
            base = _build_causal_mask(Lt, self.cfg.device, t.dtype)
            bias = _build_alibi_bias(Lt, self.cfg.nhead, self.cfg.device, t.dtype)
            attn_mask = bias.repeat(B, 1, 1) + base.unsqueeze(0).repeat(
                B * self.cfg.nhead, 1, 1
            )
            out = self.decoder(t, mem, tgt_mask=attn_mask)
        y = self.out_proj(out[-1])  # [B,Fo]
        if self.cfg.use_tanh_output:
            y = torch.tanh(y)
        return y
