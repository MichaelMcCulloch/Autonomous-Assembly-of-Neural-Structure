import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import SupervisedConfig
from src.sbb.paradigms.predictive_coding import PredictiveCoding
from tests.common import mackey_glass_generator


def run_supervised_learning_phase(model, data_ts, batch_size, n_steps, washout):
    """Helper to run a supervised learning phase and return the final error."""
    model.train()
    state_tuple = model.base.new_state(batch_size)
    errors = []
    for step in range(n_steps):
        input_t = data_ts[step].unsqueeze(0).to(DEVICE, model.cfg.dtype)
        target_t = data_ts[step + 1].unsqueeze(0).to(DEVICE, model.cfg.dtype)

        prediction, next_state_tuple = model.forward(input_t, state_tuple)

        if step >= washout:
            loss, state_tuple = model.backward(
                prediction, target_t, state_tuple, next_state_tuple
            )
            errors.append(loss.item())
        else:
            state_tuple = next_state_tuple

    return np.mean(errors[-100:]) if errors else 0.0


def test_mackey_glass_learning(tmp_path):
    """Tests generative rollout on Mackey-Glass task with proper train/test separation."""
    target_k_neuron = 40
    initial_weight_scale = 1.0 / math.sqrt(target_k_neuron + 1e-12)

    cfg = SupervisedConfig(
        num_blocks=128,
        neurons_per_block=32,
        dtype=torch.float32,
        seed=42,
        noise=0.005,
        input_features=1,
        output_features=1,
        batch_size=1,
        target_connectivity=2,
        initial_weight_scale=initial_weight_scale,
        structural_plasticity=True,
    )
    model = PredictiveCoding(cfg=cfg)
    n_train_steps = 2000
    n_rollout_steps = 50
    washout = 100
    total_points = n_train_steps + n_rollout_steps + 1
    data = mackey_glass_generator(n_points=total_points, seed=42)
    data_ts = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)

    train_error = run_supervised_learning_phase(
        model, data_ts, 1, n_train_steps, washout
    )
    assert train_error < 0.02, "Model failed to learn the task before rollout"

    eval_model = copy.deepcopy(model).eval()
    generated_sequence = []

    current_input = data_ts[n_train_steps].unsqueeze(0).to(DEVICE)
    state_tuple = model.base.new_state(cfg.batch_size)

    with torch.no_grad():
        for _ in range(n_rollout_steps):
            prediction, state_tuple = eval_model.forward(current_input, state_tuple)
            generated_sequence.append(prediction.squeeze().item())
            current_input = prediction

    generated_np = np.array(generated_sequence)
    ground_truth_np = (
        data_ts[n_train_steps + 1 : n_train_steps + 1 + n_rollout_steps]
        .squeeze()
        .numpy()
    )

    rollout_mse = np.mean((generated_np - ground_truth_np) ** 2)
    np.std(generated_np)
    np.corrcoef(generated_np, ground_truth_np)[0, 1]

    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth_np, label="Ground Truth", marker="o", linestyle="--")
    plt.plot(generated_np, label="Generated Rollout", marker="x", linestyle="-")
    plt.title(f"Mackey-Glass Autonomous Rollout (MSE: {rollout_mse:.4f})")
    plt.legend()
    plt.savefig(tmp_path / "mackey_glass.png")
    plt.close()

    assert rollout_mse < 0.15, "Generative rollout error is too high."


def lorenz_generator(n_points=1000, dt=0.01, seed=None):
    from scipy.integrate import solve_ivp

    if seed is not None:
        np.random.seed(seed)

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
    return (data - min_v) / (max_v - min_v + 1e-9), max_v - min_v, min_v


def test_lorenz_learning(tmp_path):
    """Tests generative rollout on the Lorenz attractor task."""

    cfg = SupervisedConfig(
        num_blocks=32,
        neurons_per_block=32,
        dtype=torch.float32,
        seed=42,
        noise=0.005,
        input_features=3,
        output_features=3,
        batch_size=1,
        target_connectivity=4,
        structural_plasticity=True,
    )
    model = PredictiveCoding(cfg=cfg)

    train_steps, test_steps, washout = 5000, 1000, 200
    total_points = train_steps + test_steps + washout + 1
    data, ranges, mins = lorenz_generator(total_points, seed=42)
    data_ts = torch.from_numpy(data).float()

    state_tuple = model.base.new_state(cfg.batch_size)
    model.train()
    for step in range(washout, train_steps):
        input_t = data_ts[step - 1].unsqueeze(0).to(DEVICE)
        target_t = data_ts[step].unsqueeze(0).to(DEVICE)
        pred, next_state_tuple = model.forward(input_t, state_tuple)
        _, state_tuple = model.backward(pred, target_t, state_tuple, next_state_tuple)

    eval_model = copy.deepcopy(model).eval()
    preds = []
    current_input = data_ts[train_steps - 1].unsqueeze(0).to(DEVICE)
    rollout_state = state_tuple
    with torch.no_grad():
        for _ in range(test_steps):
            pred, rollout_state = eval_model.forward(current_input, rollout_state)
            preds.append(pred.squeeze(0).cpu().numpy())
            current_input = pred

    preds_np = np.array(preds) * ranges + mins
    targets_np = data[train_steps : train_steps + test_steps] * ranges + mins
    rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(
        targets_np[:, 0], targets_np[:, 1], targets_np[:, 2], lw=0.5, label="Target"
    )
    ax.plot(
        preds_np[:, 0],
        preds_np[:, 1],
        preds_np[:, 2],
        lw=0.5,
        label="Prediction",
        linestyle="--",
    )
    ax.set_title(f"Lorenz Attractor Rollout (RMSE: {rmse:.4f})")
    ax.legend()
    plt.savefig(tmp_path / "lorenz.png")
    plt.close()

    assert np.isfinite(rmse), "Lorenz rollout NaN"


def narma10_generator(n_points=1000, seed=None):
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
    return (y - y_min) / (y_max - y_min + 1e-9), u, y_max - y_min


def test_narma10_learning(tmp_path):
    """Tests 1-step-ahead prediction on the NARMA10 task."""
    cfg = SupervisedConfig(
        num_blocks=16,
        neurons_per_block=32,
        input_features=1,
        output_features=1,
        batch_size=1,
        dtype=torch.float32,
        seed=42,
    )
    model = PredictiveCoding(cfg=cfg)
    train_steps, test_steps, washout = 2000, 500, 100
    total_points = train_steps + test_steps + washout + 1
    y_norm, u, y_range = narma10_generator(total_points, seed=42)
    y_ts = torch.from_numpy(y_norm).float().unsqueeze(-1)
    u_ts = torch.from_numpy(u).float().unsqueeze(-1)

    state_tuple = model.base.new_state(cfg.batch_size)
    model.train()
    for step in range(washout, train_steps):
        input_t = u_ts[step - 1].unsqueeze(0).to(DEVICE)
        target_t = y_ts[step].unsqueeze(0).to(DEVICE)
        pred, next_state_tuple = model.forward(input_t, state_tuple)
        _, state_tuple = model.backward(pred, target_t, state_tuple, next_state_tuple)

    eval_model = copy.deepcopy(model).eval()
    preds, targets = [], []
    eval_state = state_tuple
    with torch.no_grad():
        for step in range(test_steps):
            idx = train_steps + step
            input_t = u_ts[idx - 1].unsqueeze(0).to(DEVICE)
            pred, eval_state = eval_model.forward(input_t, eval_state)
            preds.append(pred.squeeze().item())
            targets.append(y_ts[idx].item())

    preds_np, targets_np = np.array(preds), np.array(targets)
    rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))
    nrmse = rmse / (y_range if y_range > 1e-9 else 1.0)

    plt.figure(figsize=(12, 6))
    plt.plot(targets_np, label="Target")
    plt.plot(preds_np, label="Prediction", linestyle="--")
    plt.title(f"NARMA10 1-Step Prediction (NRMSE: {nrmse:.4f})")
    plt.legend()
    plt.savefig(tmp_path / "narma10_prediction.png")
    plt.close()
    assert nrmse < 0.3, "NARMA10 NRMSE is too high"
    print(f"NARMA10 NRMSE is {nrmse}")
