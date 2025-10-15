from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Set
import os
import tempfile

import numpy as np
import torch
from torch import Tensor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import RNNBPTTConfig, SimpleRNNRegressor
from .trainer import SequenceTrainer

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
class RNNBenchmarkConfig:
    n_epochs: int = 5
    tbptt_steps: int = 256
    washout: int = 100
    hidden_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    out_dir: str = os.path.join(tempfile.gettempdir(), "rnn_bptt_plots")
    scheduled_sampling_p: float = 1.0
    sampling_decay_schedule: str = "exponential"
    sampling_jitter: float = 0.05
    ms_horizon_max: int = 16
    narma_mode: str = "exogenous"  # Driven for all models
    produce_plots: bool = True
    use_cosine_schedule: bool = True
    warmup_frac: float = 0.02
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


def train_eval_mackey(
    cfg: RNNBenchmarkConfig,
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

    mcfg = RNNBPTTConfig(
        input_size=1,
        hidden_size=cfg.hidden_size,
        output_size=1,
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
    model = SimpleRNNRegressor(mcfg)
    trainer = SequenceTrainer(
        model, use_cosine_schedule=cfg.use_cosine_schedule, warmup_frac=cfg.warmup_frac
    )
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
        H_max=cfg.ms_horizon_max,
        verbose=cfg.verbose,
    )

    model.eval()
    with torch.no_grad():
        idx = W + N_TRAIN - 1
        start_input = x[idx, 0].unsqueeze(0)
        preds_list: list[Tensor] = []
        inp = start_input
        for _ in range(N_ROLL):
            y_t, _ = model.one_step(inp, None)
            preds_list.append(y_t)
            inp = y_t
        preds = torch.stack(preds_list, dim=0)

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


def train_eval_narma(
    cfg: RNNBenchmarkConfig,
) -> Tuple[float, np.ndarray, np.ndarray, str]:
    N_TRAIN, N_EVAL, W = 1000, 50, cfg.washout
    u, y_sym, _ = narma10_generator(n_points=N_TRAIN + N_EVAL + W + 1, seed=cfg.seed)
    # Align to predict y[t+1]
    u_feats = u[:-1]  # length T
    t_np = y_sym[1:]  # length T (target)

    # Exogenous: predict y[t+1] from u[t]
    x = (
        torch.tensor(u_feats, dtype=cfg.dtype, device=cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )  # [T,1,1]
    in_size = 1

    t = (
        torch.tensor(t_np, dtype=cfg.dtype, device=cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )
    assert (
        x.shape[0] == t.shape[0]
    ), f"Length mismatch: x={x.shape[0]} vs t={t.shape[0]}"

    mcfg = RNNBPTTConfig(
        input_size=in_size,
        hidden_size=cfg.hidden_size,
        output_size=1,
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
    model = SimpleRNNRegressor(mcfg)
    trainer = SequenceTrainer(
        model, use_cosine_schedule=cfg.use_cosine_schedule, warmup_frac=cfg.warmup_frac
    )
    trainer.fit_teacher_forcing(
        x[: W + N_TRAIN],
        t[: W + N_TRAIN],
        washout=W,
        tbptt_steps=cfg.tbptt_steps,
        n_epochs=cfg.n_epochs,
        verbose=cfg.verbose,
    )

    model.eval()
    with torch.no_grad():
        x_test = x[W + N_TRAIN : W + N_TRAIN + N_EVAL]
        preds_full = []
        for i in range(x_test.shape[0]):
            y_i, _ = model.one_step(x_test[i, 0], None)
            preds_full.append(y_i)
        preds = torch.stack(preds_full, dim=0)

    preds_np = preds.squeeze(-1).squeeze(-1).float().cpu().numpy()
    tgt_np = t_np[W + N_TRAIN : W + N_TRAIN + N_EVAL]
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


def train_eval_lorenz(
    cfg: RNNBenchmarkConfig,
) -> Tuple[float, np.ndarray, np.ndarray, str]:
    N_TRAIN, N_ROLL, W = 1000, 50, cfg.washout
    data, _ = lorenz_generator(n_points=N_TRAIN + N_ROLL + W + 1, seed=cfg.seed)
    x_np, y_np = data[:-1], data[1:]
    x = torch.tensor(x_np, dtype=cfg.dtype, device=cfg.device).unsqueeze(1)  # [T,1,3]
    y = torch.tensor(y_np, dtype=cfg.dtype, device=cfg.device).unsqueeze(1)

    mcfg = RNNBPTTConfig(
        input_size=3,
        hidden_size=cfg.hidden_size,
        output_size=3,
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
    model = SimpleRNNRegressor(mcfg)
    trainer = SequenceTrainer(
        model, use_cosine_schedule=cfg.use_cosine_schedule, warmup_frac=cfg.warmup_frac
    )
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
        H_max=cfg.ms_horizon_max,
        verbose=cfg.verbose,
    )

    model.eval()
    with torch.no_grad():
        idx = W + N_TRAIN - 1
        start_input = x[idx, 0].unsqueeze(0)
        preds_list: list[Tensor] = []
        inp = start_input
        for _ in range(N_ROLL):
            y_t, _ = model.one_step(inp, None)
            preds_list.append(y_t)
            inp = y_t
        preds = torch.stack(preds_list, dim=0)

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


def run_rnn_bptt_benchmark(
    *,
    hidden_size: int = 128,
    n_epochs: int = 5,
    tbptt_steps: int = 256,
    washout: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str | torch.device = "auto",
    out_dir: str | None = None,
    scheduled_sampling_p: float = 1.0,
    sampling_decay_schedule: str = "exponential",
    sampling_jitter: float = 0.05,
    ms_horizon_max: int = 16,
    narma_mode: str = "exogenous",
    tasks: str | Iterable[str] | None = "all",
    produce_plots: bool = True,
    use_cosine_schedule: bool = True,
    warmup_frac: float = 0.02,
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

    cfg = RNNBenchmarkConfig(
        n_epochs=n_epochs,
        tbptt_steps=tbptt_steps,
        washout=washout,
        hidden_size=hidden_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        seed=seed,
        device=device,
        dtype=torch.float32,
        out_dir=(
            _ensure_out_dir(out_dir)
            if out_dir
            else _ensure_out_dir(RNNBenchmarkConfig().out_dir)
        ),
        scheduled_sampling_p=scheduled_sampling_p,
        sampling_decay_schedule=sampling_decay_schedule,
        sampling_jitter=sampling_jitter,
        ms_horizon_max=ms_horizon_max,
        narma_mode=narma_mode,
        produce_plots=produce_plots,
        use_cosine_schedule=use_cosine_schedule,
        warmup_frac=warmup_frac,
        verbose=verbose,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    _log(f"[RNN-BPTT] Device: {cfg.device}, Seed: {cfg.seed}", verbose=cfg.verbose)
    _log(f"[RNN-BPTT] Tasks: {sorted(list(task_set))}", verbose=cfg.verbose)
    _log(
        f"[RNN-BPTT] Regime: narma_mode={cfg.narma_mode}, multi-step H_max={cfg.ms_horizon_max}",
        verbose=cfg.verbose,
    )
    _log(
        f"[RNN-BPTT] Plots dir: {cfg.out_dir} | produce_plots={cfg.produce_plots}",
        verbose=cfg.verbose,
    )

    results: Dict[str, float] = {}

    if "mackey" in task_set:
        nrmse_mg, _, _, p_mg = train_eval_mackey(cfg)
        if cfg.produce_plots:
            _log(f"  -> Plot: {p_mg}", verbose=cfg.verbose)
        results["nrmse_mg"] = float(nrmse_mg)

    if "narma" in task_set:
        nrmse_narma, _, _, p_na = train_eval_narma(cfg)
        if cfg.produce_plots:
            _log(f"  -> Plot: {p_na}", verbose=cfg.verbose)
        results["nrmse_narma"] = float(nrmse_narma)

    if "lorenz" in task_set:
        nrmse_lorenz, _, _, p_lo = train_eval_lorenz(cfg)
        if cfg.produce_plots:
            _log(f"  -> Plot: {p_lo}", verbose=cfg.verbose)
        results["nrmse_lorenz"] = float(nrmse_lorenz)

    return results
