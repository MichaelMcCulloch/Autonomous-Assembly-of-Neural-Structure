import os
import torch
import numpy as np
import matplotlib

from sbb.heads.costate import CoState
from sbb.heads.feedback import Feedback
from tests.common import orthogonal_like, mackey_glass_generator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbb.paradigms.predictive_coding import (
    SupervisedConfig,
    PredictiveCoding,
)
from sbb.const import DEVICE


def test_cs_head_alignment_general(tmp_path):
    """
    Isolated test that trains CoStateHead to match a known tanh-linear mapping:
      target_costate = tanh(x @ W_true^T)
    It logs cosine similarity and MSE over steps and saves a plot.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    dtype = torch.float32

    B = 128
    input_dim = 16
    state_dim = 128
    steps = 300

    W_true = orthogonal_like(state_dim, input_dim, dtype)

    cs_head = CoState(
        input_dim=input_dim,
        state_dim=state_dim,
        dtype=dtype,
        max_norm=10.0,
        delta_max_norm=0.2,
        name="UnitTestCoStateHead",
    )

    fb_head = Feedback(
        input_dim=input_dim,
        state_dim=state_dim,
        dtype=dtype,
        max_norm=10.0,
        delta_max_norm=0.2,
        name="UnitTestFeedbackHead",
    )

    cs_cos_history = []
    cs_mse_history = []
    fb_cos_history = []
    fb_mse_history = []

    for _ in range(steps):
        x = torch.randn(B, input_dim, device=DEVICE, dtype=dtype)
        target = torch.tanh(x @ W_true.T)
        cs_pred = cs_head(x)
        fb_pred = fb_head(x)

        cs_pred_n = cs_pred / (cs_pred.norm(dim=1, keepdim=True) + 1e-12)
        fb_pred_n = fb_pred / (fb_pred.norm(dim=1, keepdim=True) + 1e-12)
        tgt_n = target / (target.norm(dim=1, keepdim=True) + 1e-12)
        cs_cos = (cs_pred_n * tgt_n).sum(dim=1).mean().item()
        fb_cos = (fb_pred_n * tgt_n).sum(dim=1).mean().item()

        cs_mse = torch.mean((cs_pred - target) ** 2).item()
        fb_mse = torch.mean((fb_pred - target) ** 2).item()
        cs_cos_history.append(cs_cos)
        cs_mse_history.append(cs_mse)

        fb_cos_history.append(fb_cos)
        fb_mse_history.append(fb_mse)

        cs_head.backward(local_signal=x, target_costate=target)
        fb_head.backward(local_signal=x, target_costate=target)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(cs_cos_history, label="CoState Cosine(sim(λ̂, λ*))", color="tab:blue")
    ax1.plot(
        fb_cos_history, label="Feedback Cosine(sim(λ̂, λ*))", color="skyblue", alpha=0.6
    )
    ax1.set_ylabel("Cosine Similarity", color="tab:blue")
    ax1.set_xlabel("Step")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(cs_mse_history, label="CoState MSE(λ̂, λ*)", color="tab:red", alpha=0.7)
    ax2.plot(fb_mse_history, label="Feedback MSE(λ̂, λ*)", color="lightcoral", alpha=0.5)
    ax2.set_ylabel("MSE", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plot_path = os.path.join(str(tmp_path), "cs_alignment_general.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Co-state alignment (general) plot saved to: {plot_path}")
    assert cs_cos_history[-1] > 0.95, f"Final cosine too low: {cs_cos_history[-1]:.3f}"


def test_cs_alignment_vs_mse(tmp_path):
    """
    Runs a 500-step Mackey-Glass training sequence.
    Tracks cosine alignment between λ̂ (co-state head output) and λ* (analytic target),
    along with prediction MSE. Produces a plot of both curves versus step.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda")
    dtype = torch.float32

    cfg = SupervisedConfig(
        num_blocks=8,
        neurons_per_block=32,
        input_features=1,
        output_features=1,
        batch_size=32,
        dtype=dtype,
        seed=0,
        noise=0.0,
    )

    model = PredictiveCoding(cfg=cfg).to(device, dtype)
    model.train()

    steps = 50000
    washout = 50
    data = mackey_glass_generator(n_points=steps + 1, seed=123)
    series = torch.tensor(data, device=DEVICE, dtype=dtype).unsqueeze(-1)

    state_tuple = model.base.new_state(cfg.batch_size)

    cosine_history = []
    mse_history = []

    for t in range(steps):
        input_t = series[t].unsqueeze(0).expand(cfg.batch_size, -1)
        target_t = series[t + 1].unsqueeze(0).expand(cfg.batch_size, -1)

        prediction, next_state_tuple = model.forward(input_t, state_tuple)

        error = prediction - target_t
        loss, state_tuple = model.backward(
            prediction, target_t, state_tuple, next_state_tuple
        )

        if t >= washout:
            lambda_hat = model.feedback(error)
            lambda_star = error @ model.readout.weight

            pred_norm = lambda_hat.norm(dim=1, keepdim=True) + 1e-12
            targ_norm = lambda_star.norm(dim=1, keepdim=True) + 1e-12
            cosine = ((lambda_hat / pred_norm) * (lambda_star / targ_norm)).sum(dim=1)
            cosine_history.append(cosine.mean().item())
            mse_history.append(loss.item())

    assert len(cosine_history) > 0
    assert len(cosine_history) == len(mse_history)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(cosine_history, label="Cosine(λ̂, λ*)", color="tab:blue")
    ax1.set_xlabel("Training step (post-washout)")
    ax1.set_ylabel("Cosine alignment", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(mse_history, label="Prediction MSE", color="tab:red", alpha=0.7)
    ax2.set_ylabel("MSE", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plot_path = os.path.join(str(tmp_path), "cs_alignment_vs_mse.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Costate alignment vs MSE plot saved to: {plot_path}")


def test_cs_alignment_predictive_coding(tmp_path):
    """
    Predictive Coding co-state alignment diagnostic:
    Tracks cosine between λ̂ = CoStateHead(error) and λ* = error @ readout.weight
    during online learning on random data.

    Uses:
    - Linear head with NLMS for predictions
    - CoState head with RLS(diagonal=False) for co-state estimation
    """
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda")
    dtype = torch.float32

    cfg = SupervisedConfig(
        num_blocks=32,
        neurons_per_block=32,
        input_features=4,
        output_features=4,
        batch_size=32,
        dtype=dtype,
        seed=0,
        noise=0.0,
    )
    model = PredictiveCoding(cfg=cfg).to(device, dtype)
    model.train()

    steps = 300
    cos_history = []
    mse_history = []

    state_tuple = model.base.new_state(cfg.batch_size)

    for t in range(steps):
        x = torch.randn(cfg.batch_size, cfg.input_features, device=DEVICE, dtype=dtype)
        target = torch.randn(
            cfg.batch_size, cfg.output_features, device=DEVICE, dtype=dtype
        )

        y_pred, next_state = model.forward(x, state_tuple)
        loss, state_tuple = model.backward(y_pred, target, state_tuple, next_state)
        error = y_pred - target
        lambda_hat = model.feedback(error)
        lambda_star = error @ model.readout.weight

        pred_n = lambda_hat / (lambda_hat.norm(dim=1, keepdim=True) + 1e-12)
        tgt_n = lambda_star / (lambda_star.norm(dim=1, keepdim=True) + 1e-12)
        cos = (pred_n * tgt_n).sum(dim=1).mean().item()
        mse = torch.mean((lambda_hat - lambda_star) ** 2).item()
        cos_history.append(cos)
        mse_history.append(mse)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(cos_history, label="Cosine(sim(λ̂, λ*))", color="tab:blue")
    ax1.set_ylabel("Cosine Similarity", color="tab:blue")
    ax1.set_xlabel("Step")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(mse_history, label="MSE(λ̂, λ*)", color="tab:red", alpha=0.7)
    ax2.set_ylabel("MSE", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plot_name = "cs_alignment_pc.png"
    plot_path = os.path.join(str(tmp_path), plot_name)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Co-state alignment (PredictiveCoding) plot saved to: {plot_path}")

    # Verify co-state alignment converges
    assert cos_history[-1] > 0.85, f"Final cosine too low: {cos_history[-1]:.3f}"
