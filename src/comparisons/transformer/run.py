from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Set
import os
import tempfile

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import (
    TransformerConfig,
    CausalTransformerEncoder,
)
from .trainer import TransformerSequenceTrainer

from src.ablation.tasks import (
    compute_sequence_metrics,
    mackey_glass_generator,
    narma10_generator,
    lorenz_generator,
)

ALL_TASKS = ("mackey", "narma", "lorenz")


def _log(msg: str, verbose: bool):
    if not verbose:
        return
    try:
        from tqdm.auto import tqdm

        tqdm.write(str(msg))
    except Exception:
        print(str(msg))


@dataclass
class TFBenchmarkConfig:
    n_epochs: int = 5
    tbptt_steps: int = 256
    washout: int = 100
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.0
    pos_encoding: str = "sinusoidal"  # 'sinusoidal' | 'alibi'
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    out_dir: str = os.path.join(tempfile.gettempdir(), "transformer_bptt_plots")
    scheduled_sampling_p: float = 1.0
    sampling_decay_schedule: str = "exponential"
    sampling_jitter: float = 0.0
    produce_plots: bool = True
    narma_mode: str = "exogenous"  # Driven for all models
    verbose: bool = False


def _ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _plot_1d(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str, enable: bool
):
    if not enable:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Target", lw=2)
    plt.plot(y_pred, label="Prediction", lw=2, ls="--")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_3d(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str, enable: bool
):
    if not enable:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    dims = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(y_true[:, i], label=f"Target {dims[i]}", lw=1.5)
        ax.plot(y_pred[:, i], label=f"Pred {dims[i]}", lw=1.5, ls="--")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax.set_ylabel(dims[i])
    axes[-1].set_xlabel("Time step")
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_tasks_arg(text: str) -> Set[str]:
    allowed = set(ALL_TASKS)
    if not text or text.strip().lower() == "all":
        return allowed
    s = text.strip()
    if s.startswith("-"):
        excl = {t.strip().lower() for t in s[1:].split(",") if t.strip()}
        return allowed.difference(excl)
    inc = {t.strip().lower() for t in s.split(",") if t.strip()}
    unknown = inc.difference(allowed)
    if unknown:
        raise ValueError(
            f"Unknown task(s): {sorted(list(unknown))} (allowed: {ALL_TASKS})"
        )
    return inc


def _train_eval_mackey(
    cfg: TFBenchmarkConfig,
) -> Tuple[float, np.ndarray, np.ndarray, str]:
    N_TRAIN, N_ROLL, W = 1000, 50, cfg.washout
    data, _ = mackey_glass_generator(
        n_points=N_TRAIN + N_ROLL + W + 1, tau=17, seed=cfg.seed
    )
    x_np, y_np = data[:-1], data[1:]
    x = (
        torch.tensor(x_np, dtype=cfg.dtype, device=cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )
    y = (
        torch.tensor(y_np, dtype=cfg.dtype, device=cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )

    tcfg = TransformerConfig(
        input_size=1,
        output_size=1,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.ff_dim,
        dropout=cfg.dropout,
        pos_encoding=cfg.pos_encoding,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        n_epochs=cfg.n_epochs,
        tbptt_steps=cfg.tbptt_steps,
        washout=cfg.washout,
        seed=cfg.seed,
        device=cfg.device,
        dtype=cfg.dtype,
        use_tanh_output=False,  # Linear output for comparison
    )
    enc = CausalTransformerEncoder(tcfg)
    trainer = TransformerSequenceTrainer(tcfg, model_enc=enc)
    trainer.fit_multistep_scheduled_sampling(
        x[: W + N_TRAIN],
        y[: W + N_TRAIN],
        washout=W,
        tbptt_steps=cfg.tbptt_steps,
        n_epochs=cfg.n_epochs,
        p_init=cfg.scheduled_sampling_p,
        p_final=0.1,
        schedule=cfg.sampling_decay_schedule,
        jitter=cfg.sampling_jitter,
        H_max=16,
        verbose=cfg.verbose,
    )

    enc.eval()
    with torch.no_grad():
        idx = W + N_TRAIN - 1
        start_input = x[idx, 0].unsqueeze(0)
        ctx = x[max(0, idx - cfg.tbptt_steps + 1) : idx]
        outs = []
        seq = torch.cat([ctx, start_input.unsqueeze(0)], dim=0)
        for _ in range(N_ROLL):
            y_h = enc.forward_last(seq)
            outs.append(y_h)
            seq = torch.cat([seq[-(cfg.tbptt_steps - 1) :], y_h.unsqueeze(0)], dim=0)
        preds = torch.stack(outs, dim=0)

    preds_np = preds.squeeze(-1).squeeze(-1).float().cpu().numpy()
    tgt_np = y_np[W + N_TRAIN : W + N_TRAIN + N_ROLL]
    m = compute_sequence_metrics(preds_np, tgt_np, normalization="std")
    out_path = os.path.join(cfg.out_dir, "mackey_rollout.png")
    _plot_1d(
        tgt_np,
        preds_np,
        f"Mackey-Glass Autonomous Rollout (NRMSE: {m.nrmse:.4f})",
        out_path,
        cfg.produce_plots,
    )
    return m.nrmse, preds_np, tgt_np, out_path


def _train_eval_narma(
    cfg: TFBenchmarkConfig,
) -> Tuple[float, np.ndarray, np.ndarray, str]:
    N_TRAIN, N_EVAL, W = 1000, 50, cfg.washout
    u, y_sym, _ = narma10_generator(n_points=N_TRAIN + N_EVAL + W + 1, seed=cfg.seed)
    x_u, tgt = u[:-1], y_sym[1:]
    t = torch.tensor(tgt, dtype=cfg.dtype, device=cfg.device).unsqueeze(-1).unsqueeze(1)

    # Use Encoder-only, exogenous input (driven)
    x = (
        torch.tensor(x_u, dtype=cfg.dtype, device=cfg.device).unsqueeze(-1).unsqueeze(1)
    )  # [T,1,1]
    in_size = 1

    tcfg = TransformerConfig(
        input_size=in_size,
        output_size=1,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.ff_dim,
        dropout=cfg.dropout,
        pos_encoding=cfg.pos_encoding,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        n_epochs=cfg.n_epochs,
        tbptt_steps=cfg.tbptt_steps,
        washout=cfg.washout,
        seed=cfg.seed,
        device=cfg.device,
        dtype=cfg.dtype,
        use_tanh_output=False,  # Linear output for comparison
    )
    enc = CausalTransformerEncoder(tcfg)
    trainer = TransformerSequenceTrainer(tcfg, model_enc=enc)
    trainer.fit_teacher_forcing(
        x[: W + N_TRAIN],
        t[: W + N_TRAIN],
        washout=W,
        tbptt_steps=cfg.tbptt_steps,
        n_epochs=cfg.n_epochs,
        verbose=cfg.verbose,
    )

    enc.eval()
    with torch.no_grad():
        x_test = x[W + N_TRAIN : W + N_TRAIN + N_EVAL]
        preds_full = []
        for i in range(x_test.shape[0]):
            L = min(cfg.tbptt_steps, W + N_TRAIN + i + 1)
            start = W + N_TRAIN + i + 1 - L
            ctx = x[start : W + N_TRAIN + i + 1]
            y_i = enc.forward_last(ctx)
            preds_full.append(y_i)
        preds = torch.stack(preds_full, dim=0)

    preds_np = preds.squeeze(-1).squeeze(-1).float().cpu().numpy()
    tgt_np = tgt[W + N_TRAIN : W + N_TRAIN + N_EVAL]
    m = compute_sequence_metrics(preds_np, tgt_np, normalization="std")
    out_path = os.path.join(cfg.out_dir, "narma10_prediction.png")
    _plot_1d(
        tgt_np,
        preds_np,
        f"NARMA10 One-Step Prediction (NRMSE: {m.nrmse:.4f})",
        out_path,
        cfg.produce_plots,
    )
    return m.nrmse, preds_np, tgt_np, out_path


def _train_eval_lorenz(
    cfg: TFBenchmarkConfig,
) -> Tuple[float, np.ndarray, np.ndarray, str]:
    N_TRAIN, N_ROLL, W = 1000, 50, cfg.washout
    data, _ = lorenz_generator(n_points=N_TRAIN + N_ROLL + W + 1, seed=cfg.seed)
    x_np, y_np = data[:-1], data[1:]
    x = torch.tensor(x_np, dtype=cfg.dtype, device=cfg.device).unsqueeze(1)
    y = torch.tensor(y_np, dtype=cfg.dtype, device=cfg.device).unsqueeze(1)

    tcfg = TransformerConfig(
        input_size=3,
        output_size=3,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.ff_dim,
        dropout=cfg.dropout,
        pos_encoding=cfg.pos_encoding,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        n_epochs=cfg.n_epochs,
        tbptt_steps=cfg.tbptt_steps,
        washout=cfg.washout,
        seed=cfg.seed,
        device=cfg.device,
        dtype=cfg.dtype,
        use_tanh_output=False,  # Linear output for comparison
    )
    enc = CausalTransformerEncoder(tcfg)
    trainer = TransformerSequenceTrainer(tcfg, model_enc=enc)
    trainer.fit_multistep_scheduled_sampling(
        x[: W + N_TRAIN],
        y[: W + N_TRAIN],
        washout=W,
        tbptt_steps=cfg.tbptt_steps,
        n_epochs=cfg.n_epochs,
        p_init=cfg.scheduled_sampling_p,
        p_final=0.1,
        schedule=cfg.sampling_decay_schedule,
        jitter=cfg.sampling_jitter,
        H_max=16,
        verbose=cfg.verbose,
    )

    enc.eval()
    with torch.no_grad():
        idx = W + N_TRAIN - 1
        start_input = x[idx, 0].unsqueeze(0)
        ctx = x[max(0, idx - cfg.tbptt_steps + 1) : idx]
        outs = []
        seq = torch.cat([ctx, start_input.unsqueeze(0)], dim=0)
        for _ in range(N_ROLL):
            y_h = enc.forward_last(seq)
            outs.append(y_h)
            seq = torch.cat([seq[-(cfg.tbptt_steps - 1) :], y_h.unsqueeze(0)], dim=0)
        preds = torch.stack(outs, dim=0)

    preds_np = preds.squeeze(1).float().cpu().numpy()
    tgt_np = y_np[W + N_TRAIN : W + N_TRAIN + N_ROLL]
    m = compute_sequence_metrics(preds_np, tgt_np, normalization="std")
    out_path = os.path.join(cfg.out_dir, "lorenz_rollout.png")
    _plot_3d(
        tgt_np,
        preds_np,
        f"Lorenz Autonomous Rollout (NRMSE: {m.nrmse:.4f})",
        out_path,
        cfg.produce_plots,
    )
    return m.nrmse, preds_np, tgt_np, out_path


def run_transformer_benchmark(
    *,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.0,
    pos_encoding: str = "sinusoidal",
    n_epochs: int = 5,
    tbptt_steps: int = 256,
    washout: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str | torch.device = "auto",
    out_dir: str | None = None,
    scheduled_sampling_p: float = 1.0,
    sampling_decay_schedule: str = "exponential",
    sampling_jitter: float = 0.0,
    tasks: str | Iterable[str] | None = "all",
    produce_plots: bool = True,
    narma_mode: str = "exogenous",
    verbose: bool = False,
) -> Dict[str, float]:
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    task_set = (
        _parse_tasks_arg(tasks) if isinstance(tasks, str) else set(tasks or ALL_TASKS)
    )
    if not task_set:
        raise ValueError("No tasks selected after parsing '--tasks'")

    cfg = TFBenchmarkConfig(
        n_epochs=n_epochs,
        tbptt_steps=tbptt_steps,
        washout=washout,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        pos_encoding=pos_encoding,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        seed=seed,
        device=device,
        dtype=torch.float32,
        out_dir=(
            _ensure_out_dir(out_dir)
            if out_dir
            else _ensure_out_dir(TFBenchmarkConfig().out_dir)
        ),
        scheduled_sampling_p=scheduled_sampling_p,
        sampling_decay_schedule=sampling_decay_schedule,
        sampling_jitter=sampling_jitter,
        produce_plots=produce_plots,
        narma_mode=narma_mode,
        verbose=verbose,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    _log(
        f"[Transformer] Device: {cfg.device}, Seed: {cfg.seed}, pos_enc={cfg.pos_encoding}, narma_mode={cfg.narma_mode}",
        verbose=cfg.verbose,
    )
    _log(f"[Transformer] Tasks: {sorted(list(task_set))}", verbose=cfg.verbose)
    _log(
        f"[Transformer] Plots dir: {cfg.out_dir} | produce_plots={cfg.produce_plots}",
        verbose=cfg.verbose,
    )

    results: Dict[str, float] = {}

    if "mackey" in task_set:
        nrmse_mg, _, _, p_mg = _train_eval_mackey(cfg)
        if cfg.produce_plots:
            _log(f"  -> Plot: {p_mg}", verbose=cfg.verbose)
        results["nrmse_mg"] = float(nrmse_mg)

    if "narma" in task_set:
        nrmse_narma, _, _, p_na = _train_eval_narma(cfg)
        if cfg.produce_plots:
            _log(f"  -> Plot: {p_na}", verbose=cfg.verbose)
        results["nrmse_narma"] = float(nrmse_narma)

    if "lorenz" in task_set:
        nrmse_lorenz, _, _, p_lo = _train_eval_lorenz(cfg)
        if cfg.produce_plots:
            _log(f"  -> Plot: {p_lo}", verbose=cfg.verbose)
        results["nrmse_lorenz"] = float(nrmse_lorenz)

    return results
