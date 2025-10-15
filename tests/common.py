from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch

from sbb.base import BaseModel
from sbb.const import DEVICE
from sbb.hyperparameters import BaseConfig
from sbb.paradigms.predictive_coding import PredictiveCoding


def mackey_glass_generator(
    n_points: int, tau: float = 17, seed: int = 42
) -> np.ndarray:
    """Generates the Mackey-Glass time series."""

    initial_value = 1.2
    series = np.zeros(n_points)
    tau_int = int(np.ceil(tau))

    np.random.seed(seed)
    series[:tau_int] = initial_value + np.random.normal(0, 0.1, tau_int) * (
        seed % 10 / 10.0
    )

    for i in range(tau_int, n_points):
        series[i] = series[i - 1] + (
            0.2 * series[i - tau_int] / (1 + series[i - tau_int] ** 10)
            - 0.1 * series[i - 1]
        )
    s_min, s_max = np.min(series), np.max(series)
    return (series - s_min) / (s_max - s_min + 1e-12)


def run_supervised_learning_phase(
    model, data_ts, batch_size, n_steps, washout, train=True
):
    """
    Helper function to run a supervised learning phase and return the final error.
    """
    if train:
        model.train()
    else:
        model.eval()

    errors = []
    running_avg_window = 100
    state_tuple = model.base.new_state(batch_size)

    with torch.no_grad():
        for step in range(1, washout):
            u_t = data_ts[step - 1].unsqueeze(0).to(DEVICE, model.dtype)
            _, state_tuple = model.forward(u_t, state_tuple)

    for step in range(washout, n_steps):
        u_t = data_ts[step - 1].unsqueeze(0).to(DEVICE, model.dtype)
        target_t = data_ts[step].unsqueeze(0).to(DEVICE, model.dtype)

        if train:
            pred, next_state_tuple = model.forward(u_t, state_tuple)
            loss, state_tuple = model.backward(
                pred, target_t, state_tuple, next_state_tuple
            )
            errors.append(loss.item())
        else:
            with torch.no_grad():
                pred, state_tuple = model.forward(u_t, state_tuple)
                loss = torch.mean((pred - target_t) ** 2)
                errors.append(loss.item())

    return np.mean(errors[-running_avg_window:]) if errors else float("inf")


def build_dense_W(model: PredictiveCoding) -> torch.Tensor:
    """
    Reconstruct a dense N x N recurrent matrix from block-sparse buffers.
    This is moderate-size only; keep N small to avoid memory blowups.
    """
    gi = model.base
    N = gi.cfg.total_neurons
    BS = gi.cfg.neurons_per_block
    W = torch.zeros(N, N, device=DEVICE, dtype=gi.dtype)
    active_slots = gi.active_blocks.nonzero(as_tuple=True)[0]
    if active_slots.numel() == 0:
        return W
    rows = gi.weight_rows[active_slots]
    cols = gi.weight_cols[active_slots]
    vals = gi.weight_values[active_slots]
    for r, c, block in zip(rows.tolist(), cols.tolist(), vals):
        r0, r1 = r * BS, (r + 1) * BS
        c0, c1 = c * BS, (c + 1) * BS
        W[r0:r1, c0:c1] = block
    return W


def power_iteration(W: torch.Tensor, iters: int = 50):
    """Approximate spectral radius with power iteration."""
    N = W.shape[0]
    v = torch.randn(N, device=DEVICE, dtype=W.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = W @ v
        n = v.norm() + 1e-12
        v = v / n
    # Rayleigh quotient
    lam = (v @ (W @ v)) / (v @ v + 1e-12)
    return float(abs(lam).item())


@dataclass
class DecayDiagnosticResults:
    """Results from decay dynamics diagnostic."""

    # Weight norm dynamics
    initial_weight_norm: float
    final_weight_norm: float
    max_weight_norm: float
    norm_growth_rate: float  # per step
    norm_stability_score: float  # 1.0 = perfect stability, 0.0 = explosion/collapse

    # Plasticity headroom
    rescue_success_rate: float  # fraction of weak blocks strengthened by Hebbian
    false_prune_rate: float  # fraction of useful blocks that decayed to pruning

    # Pruning alignment
    pruning_threshold_mean: float
    blocks_near_threshold: int  # blocks within 10% of threshold
    blocks_far_above_threshold: int  # blocks > 2x threshold

    # Block size sensitivity
    decay_force_per_param_128x128: float
    decay_force_per_param_64x64: float
    decay_force_per_param_32x32: float
    decay_force_per_param_16x16: float
    size_sensitivity_ratio: float  # should be > 1 (larger blocks get gentler decay)

    # Overall health score
    overall_score: float  # composite metric in [0, 1]


def create_test_network(
    num_blocks: int = 64,
    neurons_per_block: int = 32,
    device: str = "cuda",
) -> BaseModel:
    """Create a network for decay diagnostics."""
    cfg = BaseConfig(
        num_blocks=num_blocks,
        neurons_per_block=neurons_per_block,
        batch_size=8,
        input_features=4,
        output_features=2,
        seed=42,
        evolution_substeps=2,
        noise=0.01,
        tau_fast=0.02,
        target_connectivity=2,
    )
    return BaseModel(cfg)


def simulate_plasticity_dynamics(
    network: BaseModel,
    num_steps: int = 1000,
    hebbian_strength: float = 0.1,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Simulate weight dynamics under Hebbian + Oja + Decay.

    Returns:
        weight_norms: norm at each step
        pruning_thresholds: dynamic threshold at each step
        blocks_near_threshold: count at each step
    """
    weight_norms = []
    pruning_thresholds = []
    blocks_near_threshold = []

    batch_size = network.cfg.batch_size
    state = network.new_state(batch_size)

    for step in range(num_steps):
        # Generate synthetic input with some structure (not just noise)
        input_seq = (
            torch.randn(
                1,
                batch_size,
                network.cfg.input_features,
                device=DEVICE,
                dtype=network.dtype,
            )
            * 0.1
        )

        # Forward pass
        state, trajectory = network.forward(input_seq, state)

        # Synthetic variational signal (error-like, correlates with state)
        variational_signal = (
            trajectory * torch.randn_like(trajectory) * hebbian_strength
        )

        # Compute inverse norms
        state_norms_sq = torch.sum(trajectory**2, dim=-1, keepdim=True)
        inv_norms = 1.0 / (state_norms_sq + 1e-8)

        # Apply plasticity
        network.backward(
            system_states=trajectory,
            eligibility_traces=state.eligibility_trace.unsqueeze(0),
            activity_traces=state.homeostatic_trace.unsqueeze(0),
            variational_signal=variational_signal,
            inverse_state_norms=inv_norms,
        )

        # Measure weight norm
        with torch.no_grad():
            active_slots = network.active_blocks.nonzero().squeeze(-1)
            if active_slots.numel() > 0:
                active_weights = network.weight_values[active_slots]
                weight_norm = torch.linalg.norm(active_weights).item()
                weight_norms.append(weight_norm)

                # Get pruning threshold
                threshold = network.plasticity.structural._cached_pruning_threshold
                pruning_thresholds.append(threshold)

                # Count blocks near threshold (within 10%)
                block_norms = torch.linalg.norm(
                    active_weights.flatten(start_dim=-2), dim=-1
                )
                near_threshold = (
                    ((block_norms < threshold * 1.1) & (block_norms > threshold * 0.9))
                    .sum()
                    .item()
                )
                blocks_near_threshold.append(near_threshold)
            else:
                weight_norms.append(0.0)
                pruning_thresholds.append(0.0)
                blocks_near_threshold.append(0)

    return weight_norms, pruning_thresholds, blocks_near_threshold


def measure_rescue_capability(
    network: BaseModel,
    num_trials: int = 50,
) -> float:
    """
    Measure whether decay allows Hebbian forces to rescue weak connections.

    Tests: Can a weak block near threshold be strengthened by strong Hebbian input?
    """
    batch_size = network.cfg.batch_size
    rescued_count = 0

    for trial in range(num_trials):
        # Create weak block near threshold
        with torch.no_grad():
            active_slots = network.active_blocks.nonzero().squeeze(-1)
            if active_slots.numel() == 0:
                continue

            # Pick random active block
            slot_idx = active_slots[torch.randint(len(active_slots), (1,))].item()

            # Get current threshold
            threshold = network.plasticity.structural._cached_pruning_threshold

            # Set block to be weak (just above threshold)
            initial_norm = threshold * 1.05
            block_weights = network.weight_values[slot_idx]
            current_norm = torch.linalg.norm(block_weights)
            if current_norm > 0:
                network.weight_values[slot_idx] = block_weights * (
                    initial_norm / current_norm
                )

            # Apply strong Hebbian input for 20 steps (increased from 10 for longer simulations)
            state = network.new_state(batch_size)
            for _ in range(20):
                input_seq = (
                    torch.randn(
                        1,
                        batch_size,
                        network.cfg.input_features,
                        device=DEVICE,
                        dtype=network.dtype,
                    )
                    * 1.0
                )  # Strong input

                state, trajectory = network.forward(input_seq, state)

                # Strong correlated variational signal (Hebbian strengthening)
                variational_signal = trajectory * 1.0

                state_norms_sq = torch.sum(trajectory**2, dim=-1, keepdim=True)
                inv_norms = 1.0 / (state_norms_sq + 1e-8)

                network.backward(
                    system_states=trajectory,
                    eligibility_traces=state.eligibility_trace.unsqueeze(0),
                    activity_traces=state.homeostatic_trace.unsqueeze(0),
                    variational_signal=variational_signal,
                    inverse_state_norms=inv_norms,
                )

            # Check if block was rescued (strengthened above 1.5x threshold)
            final_norm = torch.linalg.norm(network.weight_values[slot_idx]).item()
            if final_norm > threshold * 1.5:
                rescued_count += 1

    return rescued_count / num_trials if num_trials > 0 else 0.0


def block_average_map(M: torch.Tensor, num_blocks: int, bs: int) -> torch.Tensor:
    N = num_blocks * bs
    assert M.shape == (N, N)
    out = torch.empty(num_blocks, num_blocks, device=DEVICE, dtype=M.dtype)
    for i in range(num_blocks):
        for j in range(num_blocks):
            r0, r1 = i * bs, (i + 1) * bs
            c0, c1 = j * bs, (j + 1) * bs
            out[i, j] = M[r0:r1, c0:c1].mean()
    return out


def corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a0 = a - a.mean()
    b0 = b - b.mean()
    pear = float(np.dot(a0, b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12))
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    spear = float(np.dot(ar, br) / (np.linalg.norm(ar) * np.linalg.norm(br) + 1e-12))
    return pear, spear


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    # labels: 0/1
    s = scores.astype(np.float64)
    y = labels.astype(np.int32)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return np.nan
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / (pos + 1e-12)
    fpr = fps / (neg + 1e-12)
    # trapezoid integral
    auc = np.trapz(tpr, fpr)
    return float(auc)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    s = scores.astype(np.float64)
    y = labels.astype(np.int32)
    pos = (y == 1).sum()
    if pos == 0:
        return np.nan
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    prec = tp / (np.arange(len(y_sorted)) + 1)
    rec = tp / (pos + 1e-12)
    # Average precision: sum over recall steps of precision * Δrecall
    # Using rectangle method on the discrete steps
    ap = 0.0
    prev_rec = 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_rec)
        prev_rec = r
    return float(ap)


def precision_at_k(scores: np.ndarray, target: np.ndarray, frac: float = 0.1) -> float:
    n = len(scores)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(-scores)[:k]
    return float(target[idx].mean())


def orthogonal_like(rows: int, cols: int, dtype):
    m = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    if rows >= cols:
        q, _ = torch.linalg.qr(m, mode="reduced")
    else:
        qt, _ = torch.linalg.qr(m.T, mode="reduced")
        q = qt.T
    return q


def ridge_dual(S_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    """
    Closed-form ridge in the dual:
      w = S^T (S S^T + alpha I)^(-1) y
    S_train: [T, N], y_train: [T]
    Returns w: [N]
    """
    S = torch.from_numpy(S_train).float()
    y = torch.from_numpy(y_train).float()
    T = S.shape[0]
    K = S @ S.T + alpha * torch.eye(T, dtype=S.dtype)
    v = torch.linalg.solve(K, y)
    w = S.T @ v
    return w.numpy()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    denom = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def block_map_from_weights_delta(
    W_before: torch.Tensor, W_after: torch.Tensor, Nb: int, BS: int
) -> torch.Tensor:
    # Frobenius norm of per-block delta
    delta = W_after - W_before
    out = torch.zeros(Nb, Nb, device=DEVICE, dtype=W_before.dtype)
    for i in range(Nb):
        r0, r1 = i * BS, (i + 1) * BS
        for j in range(Nb):
            c0, c1 = j * BS, (j + 1) * BS
            out[i, j] = torch.linalg.norm(delta[r0:r1, c0:c1])
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.ravel().astype(np.float64)
    b0 = b.ravel().astype(np.float64)
    return float(np.dot(a0, b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12))


def top_k_overlap(
    pred_scores: np.ndarray, target_scores: np.ndarray, frac=0.1
) -> float:
    """Compute overlap between top-k elements of two score arrays."""
    n = pred_scores.size
    k = max(1, int(round(frac * n)))
    idx_pred = np.argsort(-np.abs(pred_scores.ravel()))[:k]
    idx_true = np.argsort(-np.abs(target_scores.ravel()))[:k]
    true_mask = np.zeros(n, dtype=bool)
    true_mask[idx_true] = True
    return float(true_mask[idx_pred].mean())


def active_block_mask(model: PredictiveCoding, Nb: int) -> torch.Tensor:
    """
    Returns a [Nb,Nb] bool mask of active (row,col) blocks from the sparse buffers.
    """
    gi = model.base
    mask = torch.zeros(Nb, Nb, dtype=torch.bool, device=DEVICE)
    idx = gi.active_blocks.nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        rows = gi.weight_rows[idx]
        cols = gi.weight_cols[idx]
        mask[rows, cols] = True
    return mask


def participation_ratio(C: torch.Tensor) -> float:
    # C: covariance matrix (N x N)
    evals = torch.linalg.eigvalsh(C.to(torch.float64))
    evals = torch.maximum(evals, torch.tensor(0.0, device=evals.device))
    s1 = evals.sum()
    s2 = (evals**2).sum() + 1e-12
    return float((s1 * s1) / s2) if s1 > 0 else 0.0


def power_radius(A: torch.Tensor, iters=30) -> float:
    N = A.shape[0]
    v = torch.randn(N, device=DEVICE, dtype=A.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = A @ v
        n = v.norm() + 1e-12
        v = v / n
    lam = (v @ (A @ v)) / (v @ v + 1e-12)
    return float(lam.abs().item())
