import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import copy

from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import (
    PredictiveCoding,
    SupervisedConfig,
)
from tests.common import mackey_glass_generator


def get_config(N, BS):
    """
    Creates a config that precisely matches the OLD test environment's effective parameters.
    """
    if N % BS != 0:
        raise ValueError("N must be divisible by BS")

    cfg = SupervisedConfig(
        num_blocks=N // BS,
        neurons_per_block=BS,
        dtype=torch.float32,
        seed=42,
        noise=0.005,
        input_features=1,
        output_features=1,
        batch_size=1,
    )

    return cfg


def test_generative_rollout():
    """
    Tests if the model can generate a sequence autonomously after training.
    """
    print("\n--- [Generative Rollout] Phase 1: Training model ---")
    n_train_steps = 2000
    n_rollout_steps = 50
    washout = 100

    gen_config = get_config(
        N=1024,
        BS=32,
    )

    total_points = n_train_steps + n_rollout_steps + 1
    data = mackey_glass_generator(n_points=total_points, seed=42)
    data_ts = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)

    model = PredictiveCoding(cfg=gen_config)
    assert model is not None, "Model creation failed"

    state_tuple = model.base.new_state(gen_config.batch_size)

    training_predictions = []
    training_losses = []

    model.train()
    for step in range(n_train_steps):
        input_t = data_ts[step].unsqueeze(0).to(DEVICE, model.dtype)
        target_t = data_ts[step + 1].unsqueeze(0).to(DEVICE, model.dtype)

        prediction, next_state_tuple = model.forward(input_t, state_tuple)

        if step >= washout:
            training_predictions.append(prediction.squeeze().cpu().item())

        if step >= washout:
            loss, state_tuple = model.backward(
                prediction, target_t, state_tuple, next_state_tuple
            )
            training_losses.append(loss.item())
        else:
            state_tuple = next_state_tuple

    with torch.no_grad():
        final_pred, _ = model.forward(
            data_ts[n_train_steps - 1].unsqueeze(0).to(DEVICE, model.dtype),
            state_tuple,
        )
        train_mse = (
            (final_pred - data_ts[n_train_steps].unsqueeze(0).to(DEVICE, model.dtype))
            .pow(2)
            .mean()
            .item()
        )
        train_targets = data_ts[washout + 1 : n_train_steps + 1].squeeze().to(DEVICE)
        train_std = float(torch.std(train_targets).item())
        train_rmse = math.sqrt(train_mse)
        train_nrmse = train_rmse / (train_std if train_std > 0 else 1.0)

    print(f"Final training NRMSE: {train_nrmse:.6f}")
    assert train_mse < 0.02, "Model failed to learn the task before rollout"

    plot_dir = "perturbation_plots"
    os.makedirs(plot_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    plot_start = max(0, len(training_predictions) - 200)
    plot_predictions = training_predictions[plot_start:]
    plot_ground_truth = (
        data_ts[washout + plot_start + 1 : washout + len(training_predictions) + 1]
        .squeeze()
        .numpy()[-200:]
    )

    ax1.plot(plot_ground_truth, label="Ground Truth", alpha=0.7, linewidth=2)
    ax1.plot(plot_predictions, label="Model Predictions", alpha=0.7, linewidth=2)
    ax1.set_title("Training Phase: Last 200 Steps")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(training_losses, alpha=0.7)
    ax2.set_title("Training Loss (MSE) Over Time")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("MSE (log scale)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    training_plot_path = os.path.join(plot_dir, "training_phase.png")
    plt.savefig(training_plot_path)
    print(f"Saved training phase plot to {training_plot_path}")
    plt.close()

    print("\n--- [Generative Rollout] Phase 2: Autonomous 50-step rollout ---")
    eval_model = copy.deepcopy(model).eval()
    generated_sequence = []

    current_input = data_ts[n_train_steps - 1].unsqueeze(0).to(DEVICE, gen_config.dtype)

    rollout_state = copy.deepcopy(state_tuple)

    with torch.no_grad():
        for _ in range(n_rollout_steps):
            prediction, rollout_state = eval_model.forward(current_input, rollout_state)
            generated_sequence.append(prediction.squeeze().item())
            current_input = prediction

    generated_np = np.array(generated_sequence)
    ground_truth_np = (
        data_ts[n_train_steps : n_train_steps + n_rollout_steps].squeeze().numpy()
    )

    rollout_mse = np.mean((generated_np - ground_truth_np) ** 2)
    rollout_rmse = math.sqrt(rollout_mse)
    gt_std = float(np.std(ground_truth_np))
    rollout_nrmse = rollout_rmse / (gt_std if gt_std > 0 else 1.0)
    generated_std = np.std(generated_np)

    if generated_std > 0 and np.std(ground_truth_np) > 0:
        correlation = np.corrcoef(generated_np, ground_truth_np)[0, 1]
    else:
        correlation = 0.0

    print(f"Rollout NRMSE over 50 steps: {rollout_nrmse:.4f}")
    print(f"Std dev of generated sequence: {generated_std:.6f}")
    print(f"Correlation between generated and ground truth: {correlation:.4f}")

    rollout_plot_path = os.path.join(plot_dir, "generative.png")

    plt.figure(figsize=(12, 6))
    plt.plot(
        ground_truth_np, label="Ground Truth", marker="o", linestyle="--", markersize=4
    )
    plt.plot(
        generated_np, label="Generated Rollout", marker="x", linestyle="-", markersize=4
    )
    plt.title(
        f"Autonomous Rollout vs. Ground Truth (MSE: {rollout_mse:.4f}, Correlation: {correlation:.4f})"
    )
    plt.xlabel("Rollout Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(rollout_plot_path)
    print(f"Saved rollout plot to {rollout_plot_path}")
    plt.close()

    assert rollout_mse < 0.15, "Generative rollout error is too high."

    assert generated_std > 0.05, "Generated sequence is nearly constant."
