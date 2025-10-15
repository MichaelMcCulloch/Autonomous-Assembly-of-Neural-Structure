import os
import numpy as np
import torch
import matplotlib

from tests.common import ridge_dual, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbb.paradigms.predictive_coding import (
    SupervisedConfig,
    PredictiveCoding,
)
from sbb.const import DEVICE


def test_short_horizon_memory_capacity(tmp_path):
    """
    Reservoir-style short-horizon memory capacity with a proper train/test split
    and ridge regularization to avoid the underdetermined, overfit regime.

    We report R^2 on a held-out set for delays d=1..D, and check that
    early delays retain more information than late ones.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    Nb, BS = 32, 32
    Fi, Fo = 1, 1
    B = 1
    steps = 600  # keep runtime reasonable for CI
    max_delay = 10
    ridge_alpha = 1.0  # strong enough to regularize N >> T

    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=21,
        noise=0.0,
    )
    # Disable all learning to keep a fixed reservoir
    cfg.activity_lr = 0.0
    model = PredictiveCoding(cfg).to(device, dtype)
    model.eval()

    # Generate scalar input sequence
    u = torch.randn(steps + max_delay + 5, B, Fi, device=DEVICE, dtype=dtype) * 0.5

    # Collect reservoir states
    state = model.base.new_state(B)
    states = []
    with torch.no_grad():
        for t in range(steps + max_delay + 5):
            _, state = model.forward(u[t], state)
            states.append(state.activations.squeeze(0))  # [N]
    S_all = torch.stack(states, dim=0).detach().cpu().numpy()  # [T, N]

    # Trim warmup; align S[t] with u[t-d]
    S = S_all[max_delay + 5 : max_delay + 5 + steps]  # [steps, N]
    U = u[max_delay + 5 : max_delay + 5 + steps].squeeze(-1).squeeze(-1).cpu().numpy()

    # Train/test split (e.g., 70/30)
    split = int(0.7 * steps)

    r2_train, r2_test = [], []
    for d in range(1, max_delay + 1):
        # Predict u_{t-d} from s_t
        Sx = S[d:]  # [steps - d, N]
        y = U[:-d]  # [steps - d]

        # Split
        S_train, S_test = Sx[: split - d], Sx[split - d :]
        y_train, y_test = y[: split - d], y[split - d :]

        # Guard against very small splits (happens at large d)
        if S_train.shape[0] < 10 or S_test.shape[0] < 10:
            r2_train.append(float("nan"))
            r2_test.append(float("nan"))
            continue

        # Fit ridge in the dual (stable when N >> T)
        w = ridge_dual(S_train, y_train, ridge_alpha)

        yhat_train = S_train @ w
        yhat_test = S_test @ w

        r2_train.append(r2_score(y_train, yhat_train))
        r2_test.append(r2_score(y_test, yhat_test))

    # Plot
    plt.figure(figsize=(7, 4))
    xs = np.arange(1, max_delay + 1)
    plt.plot(xs, r2_train, marker="o", label="Train R^2 (ridge)")
    plt.plot(xs, r2_test, marker="x", label="Test R^2 (ridge)")
    plt.xlabel("Delay d (steps)")
    plt.ylabel("R^2")
    plt.title("Short-horizon memory capacity (train/test, ridge)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out = os.path.join(str(tmp_path), "memory_capacity_ridge.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[MemoryCapacity] plot saved to: {out}")

    # Basic sanity assertions (robust across seeds)
    r2t = np.array(r2_test, dtype=np.float64)
    assert np.isfinite(r2t).sum() >= max_delay - 2  # most delays finite
    # Early delays should retain more info than late ones
    early = np.nanmean(r2t[:3])
    late = np.nanmean(r2t[-3:])
    assert (
        early > late + 0.02
    ), f"Early R^2 ({early:.3f}) not greater than late ({late:.3f})"
