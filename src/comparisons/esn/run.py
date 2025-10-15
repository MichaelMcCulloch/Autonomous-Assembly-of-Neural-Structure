from dataclasses import dataclass
from typing import Dict, Iterable, Set
import os
import tempfile
import uuid

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import ESN, ESNConfig
from .trainer import ESNTrainer, RidgeConfig, RLSConfig

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
        # Fallback if tqdm not available
        print(str(msg))


@dataclass
class ESNBenchmarkConfig:
    reservoir_size: int = 1500
    density: float = 0.02
    spectral_radius: float = 0.9
    input_scale: float = 0.5
    leak: float = 0.3
    use_bias: bool = False
    add_input_to_state: bool = True
    state_nonlinearity: str = "tanh"  # 'tanh'|'relu'
    clip_state: float = 5.0

    ridge_lambda: float = 1e-4
    washout: int = 100

    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32

    out_dir: str = os.path.join(tempfile.gettempdir(), "esn_plots")
    produce_plots: bool = True
    narma_mode: str = "exogenous"  # Driven for all models
    verbose: bool = False


def _ensure_out_dir(path: str | None) -> str:
    """
    Returns a valid directory path; if path is None/empty, create a temp folder.
    """
    if not path or not str(path).strip():
        path = os.path.join(tempfile.gettempdir(), f"esn_plots_{uuid.uuid4().hex[:8]}")
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


def _cfg_to_esn(esn_cfg: ESNBenchmarkConfig, Fi: int, Fo: int) -> ESN:
    """
    Build an ESN instance from the benchmark config and specified input/output sizes.
    """
    cfg = ESNConfig(
        input_size=Fi,
        reservoir_size=esn_cfg.reservoir_size,
        output_size=Fo,
        density=esn_cfg.density,
        spectral_radius=esn_cfg.spectral_radius,
        input_scale=esn_cfg.input_scale,
        bias_scale=0.0,
        leak=esn_cfg.leak,
        use_bias=esn_cfg.use_bias,
        add_input_to_state=esn_cfg.add_input_to_state,
        seed=esn_cfg.seed,
        device=esn_cfg.device,
        dtype=esn_cfg.dtype,
        clip_state=esn_cfg.clip_state,
        state_nonlinearity=esn_cfg.state_nonlinearity,
    )
    return ESN(cfg)


def run_mackey(esn_cfg: ESNBenchmarkConfig) -> Dict[str, float]:
    N_TRAIN, N_ROLL, W = 1000, 50, esn_cfg.washout
    data, _ = mackey_glass_generator(
        n_points=N_TRAIN + N_ROLL + W + 1, tau=17, seed=esn_cfg.seed
    )
    x_np, y_np = data[:-1], data[1:]
    U = (
        torch.tensor(x_np, dtype=esn_cfg.dtype, device=esn_cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )  # [T,1,1]
    Y = (
        torch.tensor(y_np, dtype=esn_cfg.dtype, device=esn_cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )

    esn = _cfg_to_esn(esn_cfg, Fi=1, Fo=1)
    tr = ESNTrainer(esn)
    # Use RLS with Scheduled Sampling for closed-loop tasks
    tr.fit_rls_scheduled_sampling(
        U[: W + N_TRAIN],
        Y[: W + N_TRAIN],
        rc=RLSConfig(lambda_reg=1e-2, forgetting=1.0, washout=W, max_cond=1e8),
        p_init=1.0,
        p_final=0.1,
        schedule="exponential",
        jitter=0.05,
    )

    with torch.no_grad():
        idx = W + N_TRAIN - 1
        y0 = Y[idx, 0]
        preds = esn.predict_autoregressive(y0, steps=N_ROLL)  # [N_ROLL,1,1]
    preds_np = preds.squeeze(-1).squeeze(-1).cpu().numpy()
    tgt_np = y_np[W + N_TRAIN : W + N_TRAIN + N_ROLL]
    m = compute_sequence_metrics(preds_np, tgt_np, normalization="std")

    plot_dir = _ensure_out_dir(esn_cfg.out_dir) if esn_cfg.produce_plots else ""
    plot_path = (
        os.path.join(plot_dir, "mackey_esn.png") if esn_cfg.produce_plots else ""
    )
    _plot_1d(
        tgt_np,
        preds_np,
        f"ESN Mackey Rollout (NRMSE {m.nrmse:.4f})",
        plot_path,
        esn_cfg.produce_plots,
    )
    return {"nrmse_mg": float(m.nrmse)}


def run_narma(esn_cfg: ESNBenchmarkConfig) -> Dict[str, float]:
    N_TRAIN, N_EVAL, W = 1000, 50, esn_cfg.washout
    u, y_sym, _ = narma10_generator(
        n_points=N_TRAIN + N_EVAL + W + 1, seed=esn_cfg.seed
    )
    targets = y_sym[1:]
    # Exogenous input only (driven)
    U = (
        torch.tensor(u[:-1], dtype=esn_cfg.dtype, device=esn_cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )  # [T,1,1]
    Fi = 1
    Y = (
        torch.tensor(targets, dtype=esn_cfg.dtype, device=esn_cfg.device)
        .unsqueeze(-1)
        .unsqueeze(1)
    )

    esn = _cfg_to_esn(esn_cfg, Fi=Fi, Fo=1)
    tr = ESNTrainer(esn)
    # Use Ridge Regression for driven tasks
    tr.fit_ridge(
        U[: W + N_TRAIN],
        Y[: W + N_TRAIN],
        rc=RidgeConfig(ridge_lambda=esn_cfg.ridge_lambda, washout=W),
    )

    with torch.no_grad():
        _, y_preds = esn.forward_sequence(U[W + N_TRAIN : W + N_TRAIN + N_EVAL])
    preds_np = y_preds.squeeze(-1).squeeze(-1).cpu().numpy()
    tgt_np = targets[W + N_TRAIN : W + N_TRAIN + N_EVAL]
    m = compute_sequence_metrics(preds_np, tgt_np, normalization="std")

    plot_dir = _ensure_out_dir(esn_cfg.out_dir) if esn_cfg.produce_plots else ""
    plot_path = (
        os.path.join(plot_dir, "narma10_esn.png") if esn_cfg.produce_plots else ""
    )
    _plot_1d(
        tgt_np,
        preds_np,
        f"ESN NARMA10 One-Step (NRMSE {m.nrmse:.4f})",
        plot_path,
        esn_cfg.produce_plots,
    )
    return {"nrmse_narma": float(m.nrmse)}


def run_lorenz(esn_cfg: ESNBenchmarkConfig) -> Dict[str, float]:
    N_TRAIN, N_ROLL, W = 1000, 50, esn_cfg.washout
    data, _ = lorenz_generator(n_points=N_TRAIN + N_ROLL + W + 1, seed=esn_cfg.seed)
    x_np, y_np = data[:-1], data[1:]
    U = torch.tensor(x_np, dtype=esn_cfg.dtype, device=esn_cfg.device).unsqueeze(
        1
    )  # [T,1,3]
    Y = torch.tensor(y_np, dtype=esn_cfg.dtype, device=esn_cfg.device).unsqueeze(1)

    esn = _cfg_to_esn(esn_cfg, Fi=3, Fo=3)
    tr = ESNTrainer(esn)
    tr.fit_rls_scheduled_sampling(
        U[: W + N_TRAIN],
        Y[: W + N_TRAIN],
        rc=RLSConfig(lambda_reg=1e-2, forgetting=1.0, washout=W, max_cond=1e8),
        p_init=1.0,
        p_final=0.1,
        schedule="exponential",
        jitter=0.05,
    )

    with torch.no_grad():
        idx = W + N_TRAIN - 1
        y0 = Y[idx, 0]
        preds = esn.predict_autoregressive(y0, steps=N_ROLL)  # [N_ROLL,1,3]
    preds_np = preds.squeeze(1).cpu().numpy()
    tgt_np = y_np[W + N_TRAIN : W + N_TRAIN + N_ROLL]
    m = compute_sequence_metrics(preds_np, tgt_np, normalization="std")

    plot_dir = _ensure_out_dir(esn_cfg.out_dir) if esn_cfg.produce_plots else ""
    plot_path = (
        os.path.join(plot_dir, "lorenz_esn.png") if esn_cfg.produce_plots else ""
    )
    _plot_3d(
        tgt_np,
        preds_np,
        f"ESN Lorenz Rollout (NRMSE {m.nrmse:.4f})",
        plot_path,
        esn_cfg.produce_plots,
    )
    return {"nrmse_lorenz": float(m.nrmse)}


def run_esn_benchmark(tasks: Iterable[str] | str = "all", **kwargs) -> Dict[str, float]:
    if isinstance(tasks, str):
        task_set = _parse_tasks_arg(tasks)
    else:
        task_set = set(tasks)

    cfg = ESNBenchmarkConfig(**kwargs)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    _log(f"[ESN] Device: {cfg.device}, Seed: {cfg.seed}", verbose=cfg.verbose)

    results: Dict[str, float] = {}
    if "mackey" in task_set:
        results.update(run_mackey(cfg))
    if "narma" in task_set:
        results.update(run_narma(cfg))
    if "lorenz" in task_set:
        results.update(run_lorenz(cfg))
    return results
