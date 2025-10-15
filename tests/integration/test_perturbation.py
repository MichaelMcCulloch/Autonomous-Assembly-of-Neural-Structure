import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd

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
        noise=0.00,
        input_features=1,
        output_features=1,
        batch_size=1,
    )
    return cfg


@pytest.fixture
def default_config():
    """Provides a default, small configuration for most tests in this file."""
    return get_config(
        N=256,
        BS=16,
    )


def run_learning_loop(model, data_ts, n_steps, washout, train=True):
    """
    Runs a learning or evaluation loop and returns:
      - mean_mse: mean of the last 100 (or fewer if not available) MSE errors
      - nrmse: sqrt(mean_mse) normalized by std of the aligned target segment
    """
    state_tuple = model.base.new_state(model.cfg.batch_size)
    errors = []
    target_vals_for_errors = []
    model.train(train)
    with torch.set_grad_enabled(train):
        for step in range(n_steps - 1):
            input_t = data_ts[step].unsqueeze(0).to(DEVICE, model.dtype)
            target_t = data_ts[step + 1].unsqueeze(0).to(DEVICE, model.dtype)

            prediction, next_state_tuple = model.forward(input_t, state_tuple)

            if step >= washout:
                if train:
                    loss, state_tuple = model.backward(
                        prediction, target_t, state_tuple, next_state_tuple
                    )
                    errors.append(loss.item())
                    target_vals_for_errors.append(float(target_t.squeeze().item()))
                else:
                    loss = torch.mean((prediction - target_t) ** 2)
                    errors.append(loss.item())
                    target_vals_for_errors.append(float(target_t.squeeze().item()))
                    state_tuple = next_state_tuple
            else:
                state_tuple = next_state_tuple

    if not errors:
        return 0.0, 0.0

    tail_errors = errors[-100:]
    tail_targets = target_vals_for_errors[-100:] if target_vals_for_errors else []
    mean_mse = float(np.mean(tail_errors))
    rmse = math.sqrt(mean_mse)
    std_targets = float(np.std(tail_targets)) if len(tail_targets) > 0 else 0.0
    nrmse = rmse / (std_targets if std_targets > 0 else 1.0)

    return mean_mse, nrmse


def run_and_collect_loop(
    model, data_ts, n_steps, washout, train=True, initial_state_tuple=None
):
    """
    Runs a learning or evaluation loop and collects detailed history.
    Returns:
      - all_errors: list of MSE at each step after washout
      - all_predictions: list of predictions at each step after washout
      - all_targets: list of targets at each step after washout
      - final_state_tuple: the state of the model at the end
    """
    if initial_state_tuple is None:
        state_tuple = model.base.new_state(model.cfg.batch_size)
    else:
        state_tuple = initial_state_tuple

    all_errors = []
    all_predictions = []
    all_targets = []

    model.train(train)
    with torch.set_grad_enabled(train):
        for step in range(n_steps - 1):
            input_t = data_ts[step].unsqueeze(0).to(DEVICE, model.dtype)
            target_t = data_ts[step + 1].unsqueeze(0).to(DEVICE, model.dtype)

            prediction, next_state_tuple = model.forward(input_t, state_tuple)

            if step >= washout:
                if train:
                    loss, state_tuple = model.backward(
                        prediction, target_t, state_tuple, next_state_tuple
                    )
                else:
                    loss = torch.mean((prediction - target_t) ** 2)
                    state_tuple = next_state_tuple

                all_errors.append(loss.item())
                all_predictions.append(prediction.squeeze().item())
                all_targets.append(target_t.squeeze().item())
            else:
                state_tuple = next_state_tuple

    return all_errors, all_predictions, all_targets, state_tuple


def test_concept_drift_adaptation(default_config):
    """
    Tests if the model can adapt to a change in the data-generating process.
    """
    print("\n--- [Concept Drift] Phase 1: Training on tau=17 ---")
    n_phase1_steps = 1500
    n_phase2_steps = 1500
    washout = 100
    transient_discard = 500
    plot_dir = "perturbation_plots"
    os.makedirs(plot_dir, exist_ok=True)

    data_tau17 = mackey_glass_generator(n_points=n_phase1_steps, tau=17, seed=42)
    full_data_tau30 = mackey_glass_generator(
        n_points=n_phase2_steps + transient_discard, tau=30, seed=43
    )
    data_tau30 = full_data_tau30[transient_discard:]

    data_ts_tau17 = torch.tensor(data_tau17, dtype=torch.float32).unsqueeze(-1)
    data_ts_tau30 = torch.tensor(data_tau30, dtype=torch.float32).unsqueeze(-1)

    model = PredictiveCoding(cfg=default_config)
    assert model is not None, "Model creation failed"

    errors_p1, preds_p1, targets_p1, final_state_p1 = run_and_collect_loop(
        model, data_ts_tau17, n_phase1_steps, washout, train=True
    )
    final_mse_phase1 = float(np.mean(errors_p1[-100:]))
    print(f"Stable pre-perturbation MSE: {final_mse_phase1:.6f}")

    print("\n--- [Concept Drift] Phase 2: Adapting to tau=30 ---")
    errors_p2, preds_p2, targets_p2, _ = run_and_collect_loop(
        model,
        data_ts_tau30,
        n_phase2_steps,
        0,
        train=True,
        initial_state_tuple=final_state_p1,
    )
    initial_shock_mse = float(np.mean(errors_p2[:100]))
    final_mse_phase2 = float(np.mean(errors_p2[-100:]))
    print(f"Initial shock MSE on tau=30: {initial_shock_mse:.6f}")
    print(f"Final adapted MSE on tau=30: {final_mse_phase2:.6f}")

    all_errors = errors_p1 + errors_p2
    all_predictions = preds_p1 + preds_p2
    all_targets = targets_p1 + targets_p2
    drift_point = len(errors_p1)
    plot_path = os.path.join(plot_dir, "concept_drift_adaptation_ewma.png")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    errors_series = pd.Series(all_errors)
    ewma_mse = errors_series.ewm(span=100, adjust=False).mean().to_numpy()
    plt.plot(ewma_mse, label="EWMA MSE (span=100)")
    plt.axvline(
        x=drift_point,
        color="r",
        linestyle="--",
        label="Concept Drift (tau=17 -> tau=30)",
    )
    plt.title("Model Performance During Concept Drift")
    plt.xlabel("Time Steps (post-washout)")
    plt.ylabel("Mean Squared Error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.subplot(2, 1, 2)
    plot_window_start = max(0, drift_point - 200)
    plot_window_end = min(len(all_targets), drift_point + 200)
    time_axis = range(plot_window_start, plot_window_end)
    plt.plot(
        time_axis,
        all_targets[plot_window_start:plot_window_end],
        label="Target",
        alpha=0.8,
    )
    plt.plot(
        time_axis,
        all_predictions[plot_window_start:plot_window_end],
        label="Prediction",
        alpha=0.8,
        linestyle="--",
    )
    plt.axvline(x=drift_point, color="r", linestyle="--", label="Concept Drift")
    plt.title("Predictions vs. Targets Around Drift Point")
    plt.xlabel("Time Steps (post-washout)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved concept drift plot to {plot_path}")
    plt.close()
    assert final_mse_phase1 < 0.05, "Model failed to learn initial task"
    assert (
        final_mse_phase2 < initial_shock_mse / 1.5
    ), "Model should adapt to reduce error"
    assert final_mse_phase2 < 0.05, "Model failed to adapt to new task"


def test_continuous_learning_and_evaluation(default_config):
    """
    Tests if the model can learn online, evaluated by a running error metric.
    """
    print("\n--- [Continuous Eval] Running continuous evaluation ---")
    n_steps = 3000
    washout_steps = 100
    running_avg_window = 200

    data = mackey_glass_generator(n_points=n_steps)
    data_ts = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)

    model = PredictiveCoding(cfg=default_config)
    assert model is not None, "Model creation failed"

    state_tuple = model.base.new_state(model.cfg.batch_size)
    errors = []
    targets_for_errors = []
    for step in range(1, n_steps):
        u_t = data_ts[step - 1].unsqueeze(0).to(DEVICE, model.dtype)
        target_t = data_ts[step].unsqueeze(0).to(DEVICE, model.dtype)

        pred, next_state_tuple = model.forward(u_t, state_tuple)

        if step > washout_steps:
            loss, state_tuple = model.backward(
                pred, target_t, state_tuple, next_state_tuple
            )
            errors.append(loss.item())
            targets_for_errors.append(float(target_t.squeeze().item()))
        else:
            state_tuple = next_state_tuple

    errors_np = np.array(errors)
    initial_eval_mse = float(np.mean(errors_np[:running_avg_window]))
    final_eval_mse = float(np.mean(errors_np[-running_avg_window:]))

    initial_rmse = math.sqrt(initial_eval_mse)
    final_rmse = math.sqrt(final_eval_mse)
    initial_std = (
        float(np.std(targets_for_errors[:running_avg_window]))
        if len(targets_for_errors) >= running_avg_window
        else float(np.std(targets_for_errors)) if len(targets_for_errors) > 0 else 0.0
    )
    final_std = (
        float(np.std(targets_for_errors[-running_avg_window:]))
        if len(targets_for_errors) >= running_avg_window
        else float(np.std(targets_for_errors)) if len(targets_for_errors) > 0 else 0.0
    )
    initial_eval_nrmse = initial_rmse / (initial_std if initial_std > 0 else 1.0)
    final_eval_nrmse = final_rmse / (final_std if final_std > 0 else 1.0)

    print(
        f"Initial running NRMSE: {initial_eval_nrmse:.6f} (MSE: {initial_eval_mse:.6f})"
    )
    print(f"Final running NRMSE: {final_eval_nrmse:.6f} (MSE: {final_eval_mse:.6f})")

    assert (
        final_eval_mse < initial_eval_mse / 5.0
    ), "Error should decrease significantly"
    assert final_eval_mse < 0.02, "Model did not converge to a low error"


def test_recovery_from_perturbation(default_config):
    """
    Tests if the model can recover performance after significant damage.
    """
    print("\n--- [Perturbation] Phase 1: Training to stability ---")
    n_train_steps = 2000
    n_recovery_steps = 2000
    washout = 100
    plot_dir = "perturbation_plots"
    os.makedirs(plot_dir, exist_ok=True)

    full_data = mackey_glass_generator(
        n_points=n_train_steps + n_recovery_steps, seed=42
    )
    train_data = full_data[:n_train_steps]
    recovery_data = full_data[n_train_steps:]

    train_ts = torch.tensor(train_data, dtype=torch.float32).unsqueeze(-1)
    recovery_ts = torch.tensor(recovery_data, dtype=torch.float32).unsqueeze(-1)

    model = PredictiveCoding(cfg=default_config)
    assert model is not None, "Model creation failed"

    errors_pre, preds_pre, targets_pre, final_state_pre = run_and_collect_loop(
        model, train_ts, n_train_steps, washout, train=True
    )
    pre_mse = float(np.mean(errors_pre[-100:]))
    print(f"Stable pre-perturbation MSE: {pre_mse:.6f}")
    assert pre_mse < 0.01, "Model failed to reach stable state"

    print("\n--- [Perturbation] Applying severe damage ---")
    with torch.no_grad():
        live_slots = model.base.active_blocks.nonzero().squeeze(-1)
        num_to_ablate = int(len(live_slots) * 0.75)
        slots_to_ablate = live_slots[:num_to_ablate]
        model.base.weight_values.data[slots_to_ablate] = 0.0
        model.readout.weight.data *= 0.1
        model.base.activity_bias.data += (
            torch.randn_like(model.base.activity_bias) * 0.5
        )
        print(f"  - Zeroed {len(slots_to_ablate)}/{len(live_slots)} W_rec blocks")
        print("  - Reduced Wout magnitude by 90%")
        print("  - Added random noise to bias (std=0.5)")

    post_perturb_mse_immediate, _ = run_learning_loop(
        model, recovery_ts, 200, 100, train=False
    )
    print(f"Immediate post-perturbation MSE: {post_perturb_mse_immediate:.6f}")
    assert post_perturb_mse_immediate > pre_mse * 3, "Perturbation was not effective"

    print("\n--- [Perturbation] Phase 2: Recovering from damage ---")
    errors_post, preds_post, targets_post, _ = run_and_collect_loop(
        model,
        recovery_ts,
        n_recovery_steps,
        0,
        train=True,
        initial_state_tuple=final_state_pre,
    )
    post_recovery_mse = float(np.mean(errors_post[-100:]))
    print(f"Final post-recovery MSE: {post_recovery_mse:.6f}")

    all_errors = errors_pre + errors_post
    all_predictions = preds_pre + preds_post
    all_targets = targets_pre + targets_post
    perturb_point = len(errors_pre)
    plot_path = os.path.join(plot_dir, "recovery_from_perturbation_ewma.png")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    errors_series = pd.Series(all_errors)
    ewma_mse = errors_series.ewm(span=100, adjust=False).mean().to_numpy()
    plt.plot(ewma_mse, label="EWMA MSE (span=100)")
    plt.axvline(
        x=perturb_point, color="r", linestyle="--", label="Perturbation Applied"
    )
    plt.title("Model Performance During Recovery from Perturbation")
    plt.xlabel("Time Steps (post-washout)")
    plt.ylabel("Mean Squared Error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.subplot(2, 1, 2)
    plot_window_start = max(0, perturb_point - 200)
    plot_window_end = min(len(all_targets), perturb_point + 200)
    time_axis = range(plot_window_start, plot_window_end)
    plt.plot(
        time_axis,
        all_targets[plot_window_start:plot_window_end],
        label="Target",
        alpha=0.8,
    )
    plt.plot(
        time_axis,
        all_predictions[plot_window_start:plot_window_end],
        label="Prediction",
        alpha=0.8,
        linestyle="--",
    )
    plt.axvline(x=perturb_point, color="r", linestyle="--", label="Perturbation")
    plt.title("Predictions vs. Targets Around Perturbation Point")
    plt.xlabel("Time Steps (post-washout)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved perturbation recovery plot to {plot_path}")
    plt.close()
    assert (
        post_recovery_mse < post_perturb_mse_immediate / 2
    ), "Recovery should reduce error"
    assert post_recovery_mse < 0.02, "Model failed to recover to acceptable performance"


def test_model_is_not_trivial(default_config):
    """
    Tests that the model is actually learning dynamics, not just outputting constants.
    """
    print("\n--- [Trivial Check] Training model ---")
    n_steps = 1000
    washout = 100

    data = mackey_glass_generator(n_points=n_steps, seed=42)
    data_ts = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)

    model = PredictiveCoding(cfg=default_config)
    assert model is not None, "Model creation failed"

    state_tuple = model.base.new_state(model.cfg.batch_size)
    for step in range(1, n_steps):
        u_t = data_ts[step - 1].unsqueeze(0).to(DEVICE, model.dtype)
        target_t = data_ts[step].unsqueeze(0).to(DEVICE, model.dtype)

        pred, next_state_tuple = model.forward(u_t, state_tuple)

        if step > washout:
            _, state_tuple = model.backward(
                pred, target_t, state_tuple, next_state_tuple
            )
        else:
            state_tuple = next_state_tuple

    print("\n--- [Trivial Check] Analyzing predictions ---")
    predictions, targets = [], []
    eval_state = model.base.new_state(model.cfg.batch_size)
    with torch.no_grad():
        for step in range(washout, n_steps):
            u_t = data_ts[step - 1].unsqueeze(0).to(DEVICE, model.dtype)
            target_t = data_ts[step].to(DEVICE, model.dtype)

            pred, eval_state = model.forward(u_t, eval_state)
            predictions.append(pred.squeeze().item())
            targets.append(target_t.squeeze().item())

    predictions = np.array(predictions)
    targets = np.array(targets)

    pred_std = np.std(predictions)
    target_std = np.std(targets)
    mean_predictor_mse = np.mean((targets - np.mean(targets)) ** 2)
    model_mse = np.mean((targets - predictions) ** 2)
    correlation = np.corrcoef(predictions, targets)[0, 1]

    model_rmse = math.sqrt(model_mse)
    mean_predictor_rmse = math.sqrt(mean_predictor_mse)
    model_nrmse = model_rmse / (target_std if target_std > 0 else 1.0)
    mean_predictor_nrmse = mean_predictor_rmse / (target_std if target_std > 0 else 1.0)

    print(f"Prediction std: {pred_std:.6f}, Target std: {target_std:.6f}")
    print(
        f"Mean predictor NRMSE: {mean_predictor_nrmse:.6f} (MSE: {mean_predictor_mse:.6f})"
    )
    print(f"Model NRMSE: {model_nrmse:.6f} (MSE: {model_mse:.6f})")
    print(f"Prediction-target correlation: {correlation:.6f}")

    plot_dir = "perturbation_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "prediction_check.png")

    plt.figure(figsize=(12, 6))
    plt.plot(targets[:200], label="Target", alpha=0.7)
    plt.plot(predictions[:200], label="Prediction", alpha=0.7)
    plt.legend()
    plt.title("Model Predictions vs Targets")
    plt.savefig(plot_path)
    print(f"Saved prediction plot to {plot_path}")
    plt.close()

    assert pred_std > target_std * 0.5, "Predictions have too little variance"
    assert model_mse < mean_predictor_mse, "Model not significantly better than mean"
    assert correlation > 0.7, "Predictions not well correlated with targets"
