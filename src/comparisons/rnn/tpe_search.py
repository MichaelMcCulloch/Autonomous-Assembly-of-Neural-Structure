import argparse
import os
import yaml
import numpy as np
import torch
from typing import Dict, Any

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from .run import run_rnn_bptt_benchmark


def _log(msg: str, verbose: bool):
    if not verbose:
        return
    try:
        from tqdm.auto import tqdm

        tqdm.write(str(msg))
    except Exception:
        print(str(msg))


def _to_py(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _build_space():
    return {
        "hidden_size": hp.choice("hidden_size", [64, 96, 128, 192, 256]),
        "n_epochs": hp.choice("n_epochs", [3, 4, 5]),
        "tbptt_steps": hp.choice("tbptt_steps", [64, 96, 128, 192, 256]),
        "washout": hp.choice("washout", [50, 75, 100]),
        "lr": hp.loguniform("lr", np.log(3e-5), np.log(2e-3)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(5e-4)),
        "grad_clip": hp.choice("grad_clip", [0.5, 1.0, 2.0]),
        "scheduled_sampling_p": hp.uniform("scheduled_sampling_p", 0.8, 1.0),
        "sampling_decay_schedule": hp.choice(
            "sampling_decay_schedule", ["exponential", "linear"]
        ),
        "sampling_jitter": hp.choice("sampling_jitter", [0.0, 0.05, 0.1]),
        "ms_horizon_max": hp.choice("ms_horizon_max", [8, 12, 16]),
        # NARMA is driven uniformly across models
        "narma_mode": hp.choice("narma_mode", ["exogenous"]),
    }


def _gm(vals):
    arr = [float(v) for v in vals if np.isfinite(v)]
    if not arr:
        raise ValueError("No finite values for GM.")
    prod = 1.0
    for v in arr:
        prod *= max(1e-12, v)
    return prod ** (1.0 / len(arr))


def _objective(
    params: Dict[str, Any], device: torch.device, seed: int, tasks: str
) -> Dict[str, Any]:
    p = {k: _to_py(v) for k, v in params.items()}
    results = run_rnn_bptt_benchmark(
        hidden_size=int(p["hidden_size"]),
        n_epochs=int(p["n_epochs"]),
        tbptt_steps=int(p["tbptt_steps"]),
        washout=int(p["washout"]),
        lr=float(p["lr"]),
        weight_decay=float(p["weight_decay"]),
        grad_clip=float(p["grad_clip"]),
        seed=seed,
        device=device,
        out_dir=None,
        scheduled_sampling_p=float(p["scheduled_sampling_p"]),
        sampling_decay_schedule=str(p["sampling_decay_schedule"]),
        sampling_jitter=float(p["sampling_jitter"]),
        ms_horizon_max=int(p["ms_horizon_max"]),
        narma_mode="exogenous",
        tasks=tasks,
        produce_plots=False,
        use_cosine_schedule=True,
        warmup_frac=0.02,
        verbose=False,
    )
    selected = []
    if "nrmse_mg" in results:
        selected.append(results["nrmse_mg"])
    if "nrmse_narma" in results:
        selected.append(results["nrmse_narma"])
    if "nrmse_lorenz" in results:
        selected.append(results["nrmse_lorenz"])
    gm = _gm(selected)
    return {"status": STATUS_OK, "loss": gm, "fitness": -gm, **results}


def _save_yaml(
    best_params: Dict[str, Any], out_yaml: str, seed: int, device: str, tasks: str
):
    cfg = {
        "hidden_size": int(best_params["hidden_size"]),
        "n_epochs": int(best_params["n_epochs"]),
        "tbptt_steps": int(best_params["tbptt_steps"]),
        "washout": int(best_params["washout"]),
        "lr": float(best_params["lr"]),
        "weight_decay": float(best_params["weight_decay"]),
        "grad_clip": float(best_params["grad_clip"]),
        "seed": int(seed),
        "device": device,
        "scheduled_sampling_p": float(best_params["scheduled_sampling_p"]),
        "sampling_decay_schedule": str(best_params["sampling_decay_schedule"]),
        "sampling_jitter": float(best_params["sampling_jitter"]),
        "ms_horizon_max": int(best_params["ms_horizon_max"]),
        "narma_mode": "exogenous",
        "tasks": tasks,
    }
    os.makedirs(os.path.dirname(out_yaml) or ".", exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def run_search(task: str, max_evals: int = 30, seed: int = 42, verbose: bool = True):
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials = Trials()
    space = _build_space()

    _log(f"[RNN-TPE] task={task} max_evals={max_evals} device={device}", verbose)

    best = fmin(
        fn=lambda p: _objective(p, device=device, seed=seed, tasks=task),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )

    decode = {
        "hidden_size": [64, 96, 128, 192, 256, 384, 512],
        "n_epochs": [3, 4, 5, 6],
        "tbptt_steps": [64, 96, 128, 192, 256, 384, 512],
        "washout": [50, 75, 100, 150, 200],
        "grad_clip": [0.5, 1.0, 2.0],
        "sampling_decay_schedule": ["exponential", "linear"],
        "ms_horizon_max": [8, 12, 16, 24, 32],
        "narma_mode": ["exogenous"],
        "sampling_jitter": [0.0, 0.05, 0.1, 0.2],
    }
    best_params = dict(best)
    for k in decode:
        best_params[k] = decode[k][best_params[k]]  # type: ignore[index]

    comp_dir = Path("comparisons")
    comp_dir.mkdir(exist_ok=True)

    yaml_path = comp_dir / f"{task}_rnn.yaml"
    plot_dir = comp_dir / f"{task}_rnn_plots"

    cfg = {k: _to_py(v) for k, v in best_params.items()}
    cfg["seed"] = seed
    cfg["out_dir"] = str(plot_dir)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    _log(f"[RNN-TPE] Saved: {yaml_path}", verbose)
    _log("[RNN-TPE] Generating plots...", verbose)

    run_rnn_bptt_benchmark(
        tasks=task,
        device=device,
        out_dir=str(plot_dir),
        produce_plots=True,
        verbose=verbose,
        **{k: v for k, v in cfg.items() if k != "out_dir"},
    )


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--max-evals", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run_search(
        task=args.task, max_evals=args.max_evals, seed=args.seed, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
