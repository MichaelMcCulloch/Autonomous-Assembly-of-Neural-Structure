#!/usr/bin/env python3
"""
Unified benchmark runner.
Usage: python run.py <architecture> <task>
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

ARCHS = ["esn", "rnn", "lstm", "transformer"]
TASKS = ["mackey", "narma", "lorenz"]


def load_config(task: str, arch: str):
    """Load comparisons/{task}_{arch}.yaml"""
    path = Path("comparisons") / f"{task}_{arch}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\nRun: python run.py tune {arch} {task}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def run_esn(task: str):
    from src.comparisons.esn.run import run_esn_benchmark

    cfg = load_config(task, "esn")
    return run_esn_benchmark(
        tasks=task,
        verbose=True,
        produce_plots=True,
        **{k: v for k, v in cfg.items() if k not in ["tasks", "device"]},
    )


def run_rnn(task: str):
    from src.comparisons.rnn.run import run_rnn_bptt_benchmark

    cfg = load_config(task, "rnn")
    return run_rnn_bptt_benchmark(
        tasks=task,
        verbose=True,
        produce_plots=True,
        **{k: v for k, v in cfg.items() if k not in ["tasks", "device"]},
    )


def run_lstm(task: str):
    from src.comparisons.lstm.run import run_lstm_benchmark

    cfg = load_config(task, "lstm")
    return run_lstm_benchmark(
        tasks=task,
        verbose=True,
        produce_plots=True,
        **{k: v for k, v in cfg.items() if k not in ["tasks", "device"]},
    )


def run_transformer(task: str):
    from src.comparisons.transformer.run import run_transformer_benchmark

    cfg = load_config(task, "transformer")
    return run_transformer_benchmark(
        tasks=task,
        verbose=True,
        produce_plots=True,
        **{k: v for k, v in cfg.items() if k not in ["tasks", "device"]},
    )


def tune(arch: str, task: str):
    """Run TPE search for arch on task"""

    if arch == "esn":
        from src.comparisons.esn.tpe_search import run_search
    elif arch == "rnn":
        from src.comparisons.rnn.tpe_search import run_search
    elif arch == "lstm":
        from src.comparisons.lstm.tpe_search import run_search
    else:  # transformer
        from src.comparisons.transformer.tpe_search import run_search

    run_search(task=task, verbose=True)


def main():
    args = sys.argv[1:]  # Skip program name

    if len(args) < 2:
        print("Usage: python run.py <arch> <task>")
        print("       python run.py tune <arch> <task>")
        print(f"Architectures: {', '.join(ARCHS)}")
        print(f"Tasks: {', '.join(TASKS)}")
        sys.exit(1)

    if args[0] == "tune":
        if len(args) != 3:
            print("Usage: python run.py tune <arch> <task>")
            sys.exit(1)
        arch, task = args[1], args[2]
        if arch not in ARCHS or task not in TASKS:
            print(f"Invalid: arch must be in {ARCHS}, task must be in {TASKS}")
            sys.exit(1)
        tune(arch, task)
    else:
        if len(args) != 2:
            print("Usage: python run.py <arch> <task>")
            sys.exit(1)
        arch, task = args[0], args[1]
        if arch not in ARCHS or task not in TASKS:
            print(f"Invalid: arch must be in {ARCHS}, task must be in {TASKS}")
            sys.exit(1)
        runner = {
            "esn": run_esn,
            "rnn": run_rnn,
            "lstm": run_lstm,
            "transformer": run_transformer,
        }[arch]
        results = runner(task)
        print(f"\n{arch.upper()} on {task}: {results}")


if __name__ == "__main__":
    main()
