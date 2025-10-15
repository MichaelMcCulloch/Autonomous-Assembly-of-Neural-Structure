import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import solve_ivp

from sbb import (
    PredictiveCoding,
    SupervisedConfig,
)
from sbb.const import EPS, DEVICE


def mackey_glass_generator(n_points=1000, tau=17, seed=None):
    """Generates the Mackey-Glass time series."""
    if seed is not None:
        np.random.seed(seed)
    tau = int(tau)
    series = np.zeros(n_points)
    series[:tau] = 1.2
    for i in range(tau, n_points):
        series[i] = series[i - 1] + (
            0.2 * series[i - tau] / (1 + series[i - tau] ** 10) - 0.1 * series[i - 1]
        )
    s_min, s_max = np.min(series), np.max(series)
    normalized_series = (series - s_min) / (s_max - s_min + EPS)
    return normalized_series


def narma10_generator(n_points=1000, seed=None):
    """Generates the NARMA10 time series data."""
    if seed is not None:
        np.random.seed(seed)
    u = np.random.rand(n_points) * 0.5
    y = np.zeros(n_points)
    for k in range(9, n_points - 1):
        y[k + 1] = (
            0.3 * y[k]
            + 0.05 * y[k] * np.sum(y[k - 9 : k + 1])
            + 1.5 * u[k - 9] * u[k]
            + 0.1
        )
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_normalized = (y - y_min) / (y_range if y_range > EPS else 1.0)
    return y_normalized, u, y_range


def lorenz_generator(n_points=1000, dt=0.01, seed=None):
    """Generates the Lorenz attractor time series."""

    def lorenz_system(t, state):
        x, y, z = state
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    sol = solve_ivp(
        lorenz_system,
        [0, n_points * dt],
        (0.0, 1.0, 1.05),
        t_eval=np.arange(0, n_points * dt, dt),
    )
    data = sol.y.T
    min_v, max_v = data.min(axis=0), data.max(axis=0)
    return (data - min_v) / (max_v - min_v + EPS)


def get_agent_config(args, task_name):
    """Creates the hyperparameter configuration for the agent based on the task."""
    input_features, output_features = (3, 3) if task_name == "lorenz" else (1, 1)

    return SupervisedConfig(
        num_blocks=args.num_blocks,
        neurons_per_block=args.neurons_per_block,
        input_features=input_features,
        output_features=output_features,
        batch_size=1,
        dtype=torch.float32,
        seed=args.seed,
    )


def compute_nrmse(preds_np, targets_np, data_range=None):
    """Compute NRMSE with proper normalization.

    Args:
        preds_np: Predictions
        targets_np: Targets
        data_range: Range of the original (unnormalized) data. If None, uses std of targets.
    """
    if targets_np.ndim == 1:
        mse = np.mean((preds_np - targets_np) ** 2)
        rmse = np.sqrt(mse)
        if data_range is not None:
            denom = data_range if data_range > EPS else 1.0
        else:
            denom = np.std(targets_np)
            denom = denom if denom > EPS else 1.0
        nrmse = rmse / denom
    else:
        err = preds_np - targets_np
        per_dim_mse = np.mean(err**2, axis=0)
        per_dim_rmse = np.sqrt(per_dim_mse)
        if data_range is not None:
            denom = np.where(data_range > EPS, data_range, 1.0)
        else:
            denom = np.std(targets_np, axis=0)
            denom = np.where(denom > EPS, denom, 1.0)
        per_dim_nrmse = per_dim_rmse / denom
        nrmse = float(np.mean(per_dim_nrmse))
    return nrmse


def evaluate_persistence_baseline(targets_np, data_range=None):
    """Evaluate naive persistence: y[t+1] = y[t]."""
    preds_persistence = targets_np[:-1]
    targets_shifted = targets_np[1:]
    return compute_nrmse(preds_persistence, targets_shifted, data_range)


def process_and_log_results(
    preds_np, targets_np, task_name, log_dir, writer, args, data_range=None
):
    """Calculates metrics, prints results, and saves plots for a single task run."""
    mse = np.mean((preds_np - targets_np) ** 2)
    np.sqrt(mse)
    nrmse = compute_nrmse(preds_np, targets_np, data_range)

    # Compute persistence baseline for comparison
    nrmse_persistence = evaluate_persistence_baseline(targets_np, data_range)
    skill_score = 1 - (nrmse / nrmse_persistence)

    print(f"\n--- Evaluation Results for {task_name.upper()} ---")
    print(f"Final MSE: {mse:.6f}")
    print(f"Final NRMSE: {nrmse:.4f}")
    print(f"Persistence Baseline NRMSE: {nrmse_persistence:.4f}")
    print(
        f"Skill Score: {skill_score:.4f} ({'better' if skill_score > 0 else 'worse'} than persistence)"
    )

    writer.add_hparams(
        vars(args),
        {
            f"hparam/Test_NRMSE_{task_name}": nrmse,
            f"hparam/Skill_Score_{task_name}": skill_score,
        },
    )

    fig = plt.figure(figsize=(15, 8))
    if task_name == "lorenz":
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            targets_np[:, 0], targets_np[:, 1], targets_np[:, 2], lw=0.8, label="Target"
        )
        ax.plot(
            preds_np[:, 0],
            preds_np[:, 1],
            preds_np[:, 2],
            lw=0.8,
            linestyle="--",
            label="Prediction",
        )
        ax.set_title(f"Lorenz Attractor Prediction (NRMSE: {nrmse:.4f})")
    else:
        plt.plot(targets_np, label="Target")
        plt.plot(preds_np, label="Prediction", linestyle="--")
        plt.title(f"{task_name.upper()} Prediction (NRMSE: {nrmse:.4f})")
    plt.legend()
    plt.savefig(os.path.join(log_dir, f"{task_name}_prediction.png"))
    plt.close()

    return {"nrmse": nrmse, "mse": mse}


def evaluate_multistep_narma(
    model, inputs_np, targets_np, train_steps, data_range=None, horizons=[1, 5, 10, 20]
):
    """Evaluate multi-step-ahead prediction for NARMA10."""
    results = {}
    test_steps = min(200, len(inputs_np) - train_steps - max(horizons))

    for horizon in horizons:
        predictions, targets = [], []

        for start_idx in range(train_steps, train_steps + test_steps):
            # Reset state for each prediction sequence
            eval_state = model.base.new_state(1)

            with torch.no_grad():
                # Feed history to build up state
                for h in range(max(0, start_idx - 50), start_idx):
                    inp = (
                        torch.from_numpy(np.atleast_1d(inputs_np[h]))
                        .float()
                        .to(DEVICE)
                        .unsqueeze(0)
                    )
                    _, eval_state = model.forward(inp, eval_state)

                # Predict horizon steps ahead
                for h in range(horizon):
                    inp = (
                        torch.from_numpy(np.atleast_1d(inputs_np[start_idx + h]))
                        .float()
                        .to(DEVICE)
                        .unsqueeze(0)
                    )
                    pred, eval_state = model.forward(inp, eval_state)

                # Store final prediction
                predictions.append(pred.squeeze(0).cpu().numpy())
                targets.append(np.atleast_1d(targets_np[start_idx + horizon - 1]))

        nrmse = compute_nrmse(np.array(predictions), np.array(targets), data_range)
        results[f"{horizon}-step"] = nrmse

    return results


def run_narma_task(args, cfg, log_dir, writer):
    """Runs the NARMA10 benchmark (one-step-ahead prediction)."""
    train_steps, test_steps, washout = args.steps, 1000, 200
    total_points = train_steps + test_steps + washout + 1
    y_norm, u, y_range = narma10_generator(total_points, seed=cfg.seed)
    inputs_np, targets_np = u[:-1], y_norm[1:]

    model = PredictiveCoding(cfg=cfg).to(DEVICE)
    state_tuple = model.base.new_state(cfg.batch_size)

    print("Starting training (100% Teacher Forcing)...")
    for step in tqdm(range(train_steps)):
        input_t = (
            torch.from_numpy(np.atleast_1d(inputs_np[step]))
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )
        target_t = (
            torch.from_numpy(np.atleast_1d(targets_np[step]))
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )
        prediction, next_state_tuple = model.forward(input_t, state_tuple)
        if step >= washout:
            _, state_tuple = model.backward(
                prediction, target_t, state_tuple, next_state_tuple
            )
        else:
            state_tuple = next_state_tuple

        with torch.no_grad():
            active_slots = model.base.active_blocks.nonzero().squeeze(-1)
            if active_slots.numel() > 0:
                active_weights = model.base.weight_values[active_slots]
                block_norms = torch.linalg.norm(
                    active_weights.flatten(start_dim=-2), dim=-1
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_min",
                    block_norms.min().item(),
                    step,
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_max",
                    block_norms.max().item(),
                    step,
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_mean",
                    block_norms.mean().item(),
                    step,
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_std",
                    block_norms.std().item(),
                    step,
                )

            # Log input projection norms
            input_norm = torch.linalg.norm(model.base.weight_in)
            writer.add_scalar("Norms/weight_input_projection", input_norm.item(), step)

            # Log prediction head norms
            prediction_norm = torch.linalg.norm(model.readout.weight)
            writer.add_scalar("Norms/weight_readout", prediction_norm.item(), step)

    print("Starting evaluation (Lifelong Learning - One-Step-Ahead)...")
    # Keep model in training mode, continue learning during evaluation
    predictions, targets = [], []
    eval_state = state_tuple

    # No torch.no_grad() - we continue learning
    for step in range(test_steps):
        idx = train_steps + step
        input_t = (
            torch.from_numpy(np.atleast_1d(inputs_np[idx]))
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )
        target_t = (
            torch.from_numpy(np.atleast_1d(targets_np[idx]))
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )

        prediction, next_state = model.forward(input_t, eval_state)

        # Continue learning during evaluation (lifelong)
        _, eval_state = model.backward(prediction, target_t, eval_state, next_state)

        predictions.append(prediction.squeeze(0).cpu().numpy())
        targets.append(target_t.squeeze(0).cpu().numpy())

    # Multi-step evaluation
    if args.multistep_eval:
        print("\nEvaluating multi-step-ahead prediction...")
        multistep_results = evaluate_multistep_narma(
            model, inputs_np, targets_np, train_steps, y_range
        )
        print("\nMulti-step NRMSE:")
        for horizon, nrmse in multistep_results.items():
            print(f"  {horizon}: {nrmse:.4f}")
            writer.add_scalar(f"Multistep/{horizon}_NRMSE", nrmse, 0)

    return np.array(predictions), np.array(targets), y_range


def run_generative_task(args, cfg, log_dir, writer, task_name):
    """Runs a generative benchmark (Mackey-Glass or Lorenz) with scheduled sampling."""
    train_steps, test_steps, washout = args.steps, 1000, 200
    total_points = train_steps + test_steps + washout + 1

    if task_name == "mackey":
        data = mackey_glass_generator(total_points, seed=cfg.seed)
        inputs_np, targets_np = data[:-1], data[1:]
    elif task_name == "lorenz":
        data = lorenz_generator(total_points, seed=cfg.seed)
        inputs_np, targets_np = data[:-1], data[1:]

    model = PredictiveCoding(cfg=cfg).to(DEVICE)
    state_tuple = model.base.new_state(cfg.batch_size)

    print(
        f"Starting training (Scheduled Sampling: {args.sampling_decay_schedule} decay)..."
    )
    last_prediction = None

    if args.sampling_decay_schedule == "exponential":
        p_initial = args.scheduled_sampling_p
        p_final = 0.01
        if train_steps > 0 and p_initial > 0:
            decay_rate = (p_final / p_initial) ** (1 / train_steps)
        else:
            decay_rate = 1.0

    for step in tqdm(range(train_steps)):
        if args.sampling_decay_schedule == "exponential":
            p_teacher_force_deterministic = args.scheduled_sampling_p * (
                decay_rate**step
            )
        else:
            p_teacher_force_deterministic = max(
                0.0, args.scheduled_sampling_p * (1 - step / train_steps)
            )

        if args.sampling_jitter > 0:
            jitter = np.random.uniform(-args.sampling_jitter, args.sampling_jitter)
            p_teacher_force = np.clip(p_teacher_force_deterministic + jitter, 0.0, 1.0)
        else:
            p_teacher_force = p_teacher_force_deterministic

        if (
            step > washout
            and last_prediction is not None
            and torch.rand(1).item() > p_teacher_force
        ):
            input_t = last_prediction.detach()
        else:
            input_t = (
                torch.from_numpy(np.atleast_1d(inputs_np[step]))
                .float()
                .to(DEVICE)
                .unsqueeze(0)
            )

        target_t = (
            torch.from_numpy(np.atleast_1d(targets_np[step]))
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )
        prediction, next_state_tuple = model.forward(input_t, state_tuple)
        last_prediction = prediction

        if step >= washout:
            _, state_tuple = model.backward(
                prediction, target_t, state_tuple, next_state_tuple
            )
            writer.add_scalar("Training/Teacher_Force_P", p_teacher_force, step)
        else:
            state_tuple = next_state_tuple

        with torch.no_grad():
            active_slots = model.base.active_blocks.nonzero().squeeze(-1)
            if active_slots.numel() > 0:
                active_weights = model.base.weight_values[active_slots]
                block_norms = torch.linalg.norm(
                    active_weights.flatten(start_dim=-2), dim=-1
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_min",
                    block_norms.min().item(),
                    step,
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_max",
                    block_norms.max().item(),
                    step,
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_mean",
                    block_norms.mean().item(),
                    step,
                )
                writer.add_scalar(
                    "Norms/weight_recurrent_std",
                    block_norms.std().item(),
                    step,
                )

            # Log input projection norms
            input_norm = torch.linalg.norm(model.base.weight_in)
            writer.add_scalar("Norms/weight_input_projection", input_norm.item(), step)

            # Log prediction head norms
            prediction_norm = torch.linalg.norm(model.readout.weight)
            writer.add_scalar("Norms/weight_readout", prediction_norm.item(), step)

    print("Starting evaluation (Lifelong Learning - Autonomous Rollout)...")
    # Keep model in training mode, continue learning during autonomous generation
    predictions, targets = [], []
    eval_state = state_tuple

    # Start with last training prediction (autoregressive)
    current_input = (
        last_prediction.detach()
        if last_prediction is not None
        else torch.from_numpy(np.atleast_1d(inputs_np[train_steps - 1]))
        .float()
        .to(DEVICE)
        .unsqueeze(0)
    )

    # No torch.no_grad() - continue learning
    for step in range(test_steps):
        target_t = (
            torch.from_numpy(np.atleast_1d(targets_np[train_steps + step]))
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )

        prediction, next_state = model.forward(current_input, eval_state)

        # Continue learning during autonomous generation (lifelong)
        _, eval_state = model.backward(prediction, target_t, eval_state, next_state)

        predictions.append(prediction.squeeze(0).cpu().numpy())
        targets.append(target_t.squeeze(0).cpu().numpy())

        # Use prediction as next input (autoregressive)
        current_input = prediction.detach()

    return np.array(predictions), np.array(targets)


def main():
    parser = argparse.ArgumentParser(
        description="Train AANS on a time-series benchmark."
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=["mackey", "lorenz", "narma", "all"]
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--neurons-per-block", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scheduled-sampling-p",
        type=float,
        default=1.0,
        help="Initial probability for teacher forcing.",
    )
    parser.add_argument(
        "--sampling-decay-schedule",
        type=str,
        default="exponential",
        choices=["linear", "exponential"],
    )

    parser.add_argument(
        "--sampling-jitter",
        type=float,
        default=0.0,
        help="Amount of uniform noise to add to the sampling probability (e.g., 0.1).",
    )
    parser.add_argument(
        "--multistep-eval",
        action="store_true",
        help="Enable multi-step-ahead evaluation for NARMA task.",
    )

    args = parser.parse_args()

    if args.task == "all":
        all_results = {}
        task_list = ["narma", "mackey", "lorenz"]
        for task_name in task_list:
            print(f"\n{'='*20} RUNNING TASK: {task_name.upper()} {'='*20}")
            cfg = get_agent_config(args, task_name)
            log_dir = os.path.expanduser(
                f"~/.tensorboard/runs/{task_name.upper()}_AANS"
            )
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs: {log_dir}")

            if task_name == "narma":
                preds_np, targets_np, data_range = run_narma_task(args, cfg, log_dir, writer)
            else:
                preds_np, targets_np = run_generative_task(
                    args, cfg, log_dir, writer, task_name
                )
                data_range = None

            metrics = process_and_log_results(
                preds_np, targets_np, task_name, log_dir, writer, args, data_range
            )
            all_results[task_name] = metrics
            writer.close()

        print(f"\n\n{'='*20} FINAL BENCHMARK SUMMARY {'='*20}")
        print(f"{'Task':<15} {'NRMSE':<10} {'MSE':<10}")
        print("-" * 37)
        for task, metrics in all_results.items():
            print(
                f"{task.upper():<15} {metrics['nrmse']:<10.4f} {metrics['mse']:<10.6f}"
            )
        print("=" * 57)

    else:

        cfg = get_agent_config(args, args.task)
        log_dir = os.path.expanduser(f"~/.tensorboard/runs/{args.task.upper()}_AANS")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")

        if args.task == "narma":
            preds_np, targets_np, data_range = run_narma_task(args, cfg, log_dir, writer)
        else:
            preds_np, targets_np = run_generative_task(
                args, cfg, log_dir, writer, args.task
            )
            data_range = None

        process_and_log_results(preds_np, targets_np, args.task, log_dir, writer, args, data_range)
        writer.close()


if __name__ == "__main__":
    main()
