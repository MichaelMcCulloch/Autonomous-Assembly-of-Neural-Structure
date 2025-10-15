from pathlib import Path
import numpy as np
import yaml
import torch
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from .run import run_esn_benchmark


def _log(msg: str, verbose: bool):
    if not verbose:
        return
    try:
        from tqdm.auto import tqdm

        tqdm.write(str(msg))
    except Exception:
        print(str(msg))


def _to_py(x):
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _build_space():
    return {
        "reservoir_size": hp.choice("reservoir_size", [500, 800, 1000, 1500]),
        "density": hp.uniform("density", 0.01, 0.03),
        "spectral_radius": hp.uniform("spectral_radius", 0.7, 1.1),
        "input_scale": hp.loguniform("input_scale", np.log(0.1), np.log(1.0)),
        "leak": hp.uniform("leak", 0.1, 0.6),
        "use_bias": hp.choice("use_bias", [False, True]),
        "add_input_to_state": hp.choice("add_input_to_state", [True, False]),
        "ridge_lambda": hp.loguniform("ridge_lambda", np.log(1e-6), np.log(5e-3)),
        "washout": hp.choice("washout", [50, 75, 100]),
        "state_nonlinearity": hp.choice("state_nonlinearity", ["tanh", "relu"]),
        "narma_mode": hp.choice("narma_mode", ["exogenous"]),
    }


def _objective(params, device, seed, task):
    p = {k: _to_py(v) for k, v in params.items()}
    res = run_esn_benchmark(
        tasks=task,
        device=device,
        seed=seed,
        out_dir=None,
        produce_plots=False,
        verbose=False,
        **p,
    )
    val = res.get(f"nrmse_{task}" if task != "mackey" else "nrmse_mg", 1e9)
    return {"status": STATUS_OK, "loss": val, **res}


def run_search(task: str, max_evals: int = 50, seed: int = 42, verbose: bool = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials = Trials()
    space = _build_space()

    _log(f"[ESN-TPE] task={task} max_evals={max_evals} device={device}", verbose)

    best = fmin(
        fn=lambda p: _objective(p, device=device, seed=seed, task=task),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )

    decode = {
        "reservoir_size": [500, 800, 1000, 1500, 2000, 2500, 3000],
        "washout": [50, 75, 100, 150, 200],
        "use_bias": [False, True],
        "add_input_to_state": [True, False],
        "state_nonlinearity": ["tanh", "relu"],
        "narma_mode": ["exogenous"],
    }
    best_params = dict(best)
    for k in decode:
        best_params[k] = decode[k][best_params[k]]  # type: ignore[index]

    best_params["density"] = float(best_params.get("density", 0.02))
    best_params["spectral_radius"] = float(best_params.get("spectral_radius", 0.9))
    best_params["input_scale"] = float(best_params.get("input_scale", 0.5))
    best_params["leak"] = float(best_params.get("leak", 0.3))
    best_params["ridge_lambda"] = float(best_params.get("ridge_lambda", 1e-4))

    comp_dir = Path("comparisons")
    comp_dir.mkdir(exist_ok=True)

    yaml_path = comp_dir / f"{task}_esn.yaml"
    plot_dir = comp_dir / f"{task}_esn_plots"

    cfg = {k: _to_py(v) for k, v in best_params.items()}
    cfg["seed"] = seed
    cfg["out_dir"] = str(plot_dir)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    _log(f"[ESN-TPE] Saved: {yaml_path}", verbose)
    _log("[ESN-TPE] Generating plots...", verbose)

    run_esn_benchmark(
        tasks=task,
        device=device,
        out_dir=str(plot_dir),
        produce_plots=True,
        verbose=verbose,
        **{k: v for k, v in cfg.items() if k != "out_dir"},
    )


def main():
    import argparse

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
