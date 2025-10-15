```./test_decay_dynamics.py
# """
# Diagnostic tests for evaluating weight decay dynamics.

# This module provides systematic tests to evaluate different decay rules by measuring:
# 1. Weight norm stability (does it prevent explosion?)
# 2. Plasticity headroom (can Hebbian/Oja forces rescue useful connections?)
# 3. Pruning alignment (does decay pressure align with structural pruning?)
# 4. Block size sensitivity (do larger blocks get appropriate treatment?)

# These diagnostics are faster than full RL training and provide quantitative metrics
# for comparing decay strategies.
# """

# import numpy as np
# import torch
# from sbb.const import DEVICE
# from tests.common import (
#     DecayDiagnosticResults,
#     create_test_network,
#     measure_rescue_capability,
#     simulate_plasticity_dynamics,
# )


# def test_current_decay_rule():
#     """Test the currently implemented decay rule.
#     Run full diagnostic suite and compute all metrics.
#     """
#     network = create_test_network()

#     # 1. Simulate dynamics
#     weight_norms, thresholds, near_threshold_counts = simulate_plasticity_dynamics(
#         network, num_steps=1000, hebbian_strength=0.1
#     )

#     # 2. Weight norm dynamics
#     initial_norm = weight_norms[0]
#     final_norm = weight_norms[-1]
#     max_norm = max(weight_norms)

#     # Growth rate (log scale to handle both growth and decay)
#     norm_ratio = final_norm / (initial_norm + 1e-8)
#     norm_growth_rate = (norm_ratio - 1.0) / len(weight_norms)

#     # Stability score: penalize both explosion and collapse
#     # Ideal is modest growth (1.0 to 1.5 range)
#     if 1.0 <= norm_ratio <= 1.5:
#         norm_stability_score = 1.0
#     elif norm_ratio < 1.0:
#         # Collapse
#         norm_stability_score = max(0.0, norm_ratio)
#     else:
#         # Explosion
#         norm_stability_score = max(0.0, 1.0 - (norm_ratio - 1.5) / 10.0)

#     # 3. Rescue capability (test on fresh network to avoid equilibrium artifacts)
#     fresh_network = create_test_network()
#     rescue_rate = measure_rescue_capability(fresh_network, num_trials=500)

#     # False prune rate (assume 10% if rescue rate is low)
#     false_prune_rate = max(0.0, 0.1 - rescue_rate * 0.1)

#     # 4. Pruning alignment
#     threshold_mean = np.mean(thresholds)
#     near_threshold_mean = np.mean(near_threshold_counts)

#     # Count blocks far above threshold
#     with torch.no_grad():
#         active_slots = network.active_blocks.nonzero().squeeze(-1)
#         if active_slots.numel() > 0:
#             active_weights = network.weight_values[active_slots]
#             block_norms = torch.linalg.norm(
#                 active_weights.flatten(start_dim=-2), dim=-1
#             )
#             far_above = (block_norms > threshold_mean * 2.0).sum().item()
#         else:
#             far_above = 0

#     # 5. Block size sensitivity (requires creating networks with different sizes)
#     network_128 = create_test_network(num_blocks=16, neurons_per_block=128)
#     network_64 = create_test_network(num_blocks=16, neurons_per_block=64)
#     network_32 = create_test_network(num_blocks=16, neurons_per_block=32)
#     network_16 = create_test_network(num_blocks=16, neurons_per_block=16)

#     # Measure decay force per parameter by running one step
#     def measure_decay_force(net):
#         state = net.new_state(net.cfg.batch_size)
#         input_seq = (
#             torch.randn(
#                 1,
#                 net.cfg.batch_size,
#                 net.cfg.input_features,
#                 device=DEVICE,
#                 dtype=net.dtype,
#             )
#             * 0.1
#         )

#         # Store initial weights
#         with torch.no_grad():
#             active_slots = net.active_blocks.nonzero().squeeze(-1)
#             if active_slots.numel() == 0:
#                 return 0.0
#             initial_weights = net.weight_values[active_slots].clone()

#         # One plasticity step with zero Hebbian (pure decay)
#         state, trajectory = net.forward(input_seq, state)
#         state_norms_sq = torch.sum(trajectory**2, dim=-1, keepdim=True)
#         inv_norms = 1.0 / (state_norms_sq + 1e-8)

#         net.apply_plasticity(
#             system_states=trajectory,
#             eligibility_traces=state.eligibility_trace.unsqueeze(0),
#             activity_traces=state.homeostatic_trace.unsqueeze(0),
#             projected_fields=state.input_projection.unsqueeze(0),
#             variational_signal=torch.zeros_like(trajectory),  # No Hebbian
#             inverse_state_norms=inv_norms,
#         )

#         # Measure change per parameter
#         with torch.no_grad():
#             final_weights = net.weight_values[active_slots]
#             delta = (final_weights - initial_weights).abs().mean().item()
#             num_params = final_weights.numel()
#             return delta / num_params if num_params > 0 else 0.0

#     decay_force_128 = measure_decay_force(network_128)
#     decay_force_64 = measure_decay_force(network_64)
#     decay_force_32 = measure_decay_force(network_32)
#     decay_force_16 = measure_decay_force(network_16)

#     # Larger blocks should have smaller decay force per param
#     size_sensitivity_ratio = decay_force_16 / (decay_force_32 + 1e-10)

#     # 6. Overall health score (weighted composite)
#     overall_score = (
#         norm_stability_score * 0.4  # Weight norm stability is critical
#         + rescue_rate * 0.3  # Must allow plasticity to work
#         + (1.0 - false_prune_rate) * 0.2  # Don't kill useful connections
#         + min(1.0, size_sensitivity_ratio / 2.0) * 0.1  # Bonus for size-awareness
#     )

#     results = DecayDiagnosticResults(
#         initial_weight_norm=initial_norm,
#         final_weight_norm=final_norm,
#         max_weight_norm=max_norm,
#         norm_growth_rate=norm_growth_rate,
#         norm_stability_score=norm_stability_score,
#         rescue_success_rate=rescue_rate,
#         false_prune_rate=false_prune_rate,
#         pruning_threshold_mean=float(threshold_mean),
#         blocks_near_threshold=int(near_threshold_mean),
#         blocks_far_above_threshold=far_above,
#         decay_force_per_param_128x128=decay_force_128,
#         decay_force_per_param_64x64=decay_force_64,
#         decay_force_per_param_32x32=decay_force_32,
#         decay_force_per_param_16x16=decay_force_16,
#         size_sensitivity_ratio=size_sensitivity_ratio,
#         overall_score=overall_score,
#     )

#     """Print a human-readable diagnostic report."""
#     print("\n" + "=" * 70)
#     print("DECAY DYNAMICS DIAGNOSTIC REPORT")
#     print("=" * 70)

#     print("\n[1] Weight Norm Dynamics")
#     print(f"    Initial norm:      {results.initial_weight_norm:.3f}")
#     print(f"    Final norm:        {results.final_weight_norm:.3f}")
#     print(f"    Max norm:          {results.max_weight_norm:.3f}")
#     print(f"    Growth rate:       {results.norm_growth_rate:+.6f} per step")
#     print(f"    Stability score:   {results.norm_stability_score:.3f} / 1.0")

#     status = "✓ STABLE" if results.norm_stability_score > 0.8 else "⚠ UNSTABLE"
#     print(f"    Status:            {status}")

#     print("\n[2] Plasticity Headroom")
#     print(f"    Rescue success:    {results.rescue_success_rate:.1%}")
#     print(f"    False prune rate:  {results.false_prune_rate:.1%}")

#     status = "✓ GOOD" if results.rescue_success_rate > 0.5 else "⚠ LIMITED"
#     print(f"    Status:            {status}")

#     print("\n[3] Pruning Alignment")
#     print(f"    Avg threshold:     {results.pruning_threshold_mean:.4f}")
#     print(f"    Blocks near threshold: {results.blocks_near_threshold}")
#     print(f"    Blocks far above:      {results.blocks_far_above_threshold}")

#     print("\n[4] Block Size Sensitivity")
#     print(f"    Decay/param (128x128): {results.decay_force_per_param_128x128:.8f}")
#     print(f"    Decay/param (64x64):   {results.decay_force_per_param_64x64:.8f}")
#     print(f"    Decay/param (32x32):   {results.decay_force_per_param_32x32:.8f}")
#     print(f"    Decay/param (16x16):   {results.decay_force_per_param_16x16:.8f}")
#     print(f"    Sensitivity ratio:     {results.size_sensitivity_ratio:.2f}x")

#     status = "✓ SIZE-AWARE" if results.size_sensitivity_ratio > 1.5 else "⚠ SIZE-BLIND"
#     print(f"    Status:              {status}")

#     print("\n[5] Overall Health")
#     print(f"    Composite score:   {results.overall_score:.3f} / 1.0")

#     if results.overall_score > 0.8:
#         grade = "A (Excellent)"
#     elif results.overall_score > 0.6:
#         grade = "B (Good)"
#     elif results.overall_score > 0.4:
#         grade = "C (Fair)"
#     else:
#         grade = "D (Poor)"

#     print(f"    Grade:             {grade}")
#     print("\n" + "=" * 70 + "\n")

#     # Assert minimum requirements
#     assert results.norm_stability_score > 0.5, "Weight norms must be reasonably stable"
#     assert results.overall_score > 0.4, "Overall health must be at least fair"
```
```./test_eligibility.py
import os
import math
import torch
import numpy as np
import matplotlib

from tests.common import (
    auroc,
    average_precision,
    build_dense_W,
    precision_at_k,
    block_average_map,
    corr,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbb.paradigms.predictive_coding import (
    SupervisedConfig,
    PredictiveCoding,
)
from sbb.const import DEVICE


def test_eligibility_forward_mode_exact_with_ranking(tmp_path):
    """
    Exact forward-mode e-prop gradient vs. two surrogates, now with ranking metrics:
      - AUROC / AUPRC
      - Precision@K
    Saves heatmaps, scatters, and PR curves to tmp_path.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    Nb = 32
    BS = 32
    N = Nb * BS
    Fi, Fo = 3, 3
    B = 2
    T = 24
    seed = 321

    torch.manual_seed(seed)
    np.random.seed(seed)

    # TODO: Re-enable homeostasis (activity_lr > 0) to validate biological plausibility
    #       claims. Currently disabled for diagnostic clarity but undermines paper's
    #       homeostatic regulation narrative.
    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=seed,
        noise=0.0,
        activity_lr=0.0,
        structural_plasticity=True,
    )
    model = PredictiveCoding(cfg=cfg).to(device, dtype)
    model.eval()

    dt = cfg.time_step_delta
    tau_fast = cfg.tau_fast
    M_micro = max(1, int(round(tau_fast / dt)))
    alpha = math.exp(-dt / (tau_fast + 1e-12))

    W_dense = build_dense_W(model)
    W_in = model.base.weight_in
    bias = model.base.activity_bias.squeeze(0)
    W_out = model.readout.weight
    assert W_dense.shape == (N, N)

    X = torch.randn(T, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(T, B, Fo, device=DEVICE, dtype=dtype)

    s = torch.zeros(N, device=DEVICE, dtype=dtype)
    E_exact = torch.zeros(N, N, device=DEVICE, dtype=dtype)
    grad_exact = torch.zeros(N, N, device=DEVICE, dtype=dtype)
    grad_diag = torch.zeros_like(grad_exact)
    grad_emaS = torch.zeros_like(grad_exact)

    with torch.no_grad():
        for t in range(T):
            ext = torch.tanh(X[t, 0] @ W_in.T)
            S_bar = torch.zeros(N, device=DEVICE, dtype=dtype)
            G_bar = torch.zeros(N, device=DEVICE, dtype=dtype)

            for m in range(M_micro):
                s_prev = s
                rec = W_dense @ s_prev
                base = rec + ext + bias
                pot = torch.tanh(base)
                g = 1.0 - pot * pot
                one_minus_alpha = 1.0 - alpha

                WE = W_dense @ E_exact
                inj = s_prev.unsqueeze(0).expand(N, N)
                E_exact = alpha * E_exact + one_minus_alpha * (
                    g.unsqueeze(1) * (WE + inj)
                )
                s = alpha * s_prev + one_minus_alpha * pot

                S_bar = alpha * S_bar + one_minus_alpha * s_prev
                G_bar = alpha * G_bar + one_minus_alpha * g

            y = s @ W_out.T
            err = y - Y[t, 0]
            dL_ds = err @ W_out

            grad_exact += dL_ds.unsqueeze(1) * E_exact

            grad_diag += (dL_ds * G_bar).unsqueeze(1) @ S_bar.unsqueeze(0)
            grad_emaS += dL_ds.unsqueeze(1) @ S_bar.unsqueeze(0)

    exact_blocks = block_average_map(grad_exact, Nb, BS).detach().cpu().numpy()
    diag_blocks = block_average_map(grad_diag, Nb, BS).detach().cpu().numpy()
    emaS_blocks = block_average_map(grad_emaS, Nb, BS).detach().cpu().numpy()

    # Ranking metrics vs exact
    def make_labels_from_exact(exact_mat: np.ndarray, top_frac=0.1):
        flat = exact_mat.ravel()
        k = max(1, int(round(top_frac * flat.size)))
        thresh = np.partition(np.abs(flat), -k)[-k]
        return (np.abs(flat) >= thresh).astype(np.int32)

    labels = make_labels_from_exact(exact_blocks, top_frac=0.1)
    for name, approx in [("diag_factorized", diag_blocks), ("ema_s_only", emaS_blocks)]:
        s = np.abs(approx.ravel()).astype(np.float64)
        auc = auroc(s, labels)
        ap = average_precision(s, labels)
        p_at_10 = precision_at_k(s, labels, 0.1)
        print(
            f"[Eligibility ranking] {name}: AUROC={auc:.3f} AP={ap:.3f} P@10%={p_at_10:.3f}"
        )

        # PR curve
        order = np.argsort(-s)
        y_sorted = labels[order]
        tp = np.cumsum(y_sorted)
        prec = tp / (np.arange(len(y_sorted)) + 1)
        rec = tp / (y_sorted.sum() + 1e-12)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, label=f"{name} (AP {ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR curve vs exact ({name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(str(tmp_path), f"eligibility_pr_{name}.png"), dpi=150)
        plt.close()

    # Existing visuals
    def save_heatmap(mat, title, fname):
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Pre block j")
        ax.set_ylabel("Post block i")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(str(tmp_path), fname), dpi=150)
        plt.close(fig)

    save_heatmap(exact_blocks, "Exact dL/dW (blockwise)", "grad_exact_blocks.png")
    save_heatmap(
        diag_blocks, "Diag factorized approx (blockwise)", "grad_diag_blocks.png"
    )
    save_heatmap(emaS_blocks, "EMA(s) only approx (blockwise)", "grad_emaS_blocks.png")

    # Scatter comparisons
    def save_scatter(a, b, title, fname):
        a = a.ravel()
        b = b.ravel()
        pear, spear = corr(a, b)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(a, b, s=14, alpha=0.65)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(f"{title}\n(Pearson {pear:.3f}, Spearman {spear:.3f})")
        ax.set_xlabel("Exact blocks")
        ax.set_ylabel("Approx blocks")
        if np.std(a) > 1e-12:
            m = np.dot(a - a.mean(), b - b.mean()) / (np.var(a) * len(a) + 1e-12)
            b0 = b.mean() - m * a.mean()
            xs = np.linspace(a.min(), a.max(), 100)
            ax.plot(xs, m * xs + b0, color="tab:red", lw=2, alpha=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(str(tmp_path), fname), dpi=150)
        plt.close(fig)

    save_scatter(
        exact_blocks,
        diag_blocks,
        "Exact vs Diag factorized",
        "scatter_exact_vs_diag.png",
    )
    save_scatter(
        exact_blocks, emaS_blocks, "Exact vs EMA(s) only", "scatter_exact_vs_emaS.png"
    )

    # Assertions: basic sanity (finite values)
    assert np.isfinite(exact_blocks).all()
    assert np.isfinite(diag_blocks).all()
    assert np.isfinite(emaS_blocks).all()


def test_eligibility_pre_baseline_with_lag(tmp_path):
    """
    Diagnostic:
      - Compare H_pre[i,j] = ⟨ E_block[i,t] * J_true[j,t] ⟩_t
                G_pre[i,j] = ⟨ Sprev_block[i,t] * J_true[j,t] ⟩_t
      - Estimate temporal lag τ* maximizing corr(E_block[:,i], mean_j J_true[:,j])
    Saves heatmaps, scatter, and lag histogram.
    """
    device = torch.device("cuda")
    dtype = torch.float32

    Nb, BS = 32, 32
    Fi, Fo = 3, 3
    B, T = 16, 64
    max_lag = 8  # steps

    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=123,
        noise=0.0,
    )

    # Freeze synaptic/core learning to measure signals only
    cfg.activity_lr = 0.0

    model = PredictiveCoding(cfg=cfg).to(device, dtype)
    model.eval()

    def block_means(vec_N: torch.Tensor):
        return vec_N.view(B, Nb, BS).mean(dim=2)

    E_blk_ts = []
    Spre_blk_ts = []
    Jtrue_blk_ts = []

    X = torch.randn(T, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(T, B, Fo, device=DEVICE, dtype=dtype)

    state = model.base.new_state(B)

    with torch.no_grad():
        for t in range(T):
            s_prev = state.activations
            x_t = X[t]
            y_t = Y[t]
            y_pred, next_state = model.forward(x_t, state)

            error = y_pred - y_t
            dL_dS = error @ model.readout.weight
            post_gain = 1.0 - next_state.activations.pow(2)
            Jtrue = dL_dS * post_gain

            E_blk = block_means(next_state.eligibility_trace).mean(0)  # [Nb]
            Spre_blk = block_means(s_prev).mean(0)  # [Nb]
            Jtrue_blk = block_means(Jtrue).mean(0)  # [Nb]

            E_blk_ts.append(E_blk)
            Spre_blk_ts.append(Spre_blk)
            Jtrue_blk_ts.append(Jtrue_blk)

            state = next_state

    # [T, Nb]
    E_series = torch.stack(E_blk_ts, dim=0).cpu().numpy()
    torch.stack(Spre_blk_ts, dim=0).cpu().numpy()
    Jtrue_series = torch.stack(Jtrue_blk_ts, dim=0).cpu().numpy()
    Jtrue_mean = Jtrue_series.mean(axis=1)  # [T]

    # Build H_pre and G_pre as before for visuals
    E_blk_ts_full = torch.stack(
        [e.unsqueeze(0).repeat(B, 1) for e in E_blk_ts], dim=0
    )  # just to reuse code
    Spre_blk_ts_full = torch.stack(
        [s.unsqueeze(0).repeat(B, 1) for s in Spre_blk_ts], dim=0
    )
    Jtrue_blk_ts_full = torch.stack(
        [j.unsqueeze(0).repeat(B, 1) for j in Jtrue_blk_ts], dim=0
    )

    E_blk_ts_mean = E_blk_ts_full.mean(dim=1)
    Spre_blk_ts_mean = Spre_blk_ts_full.mean(dim=1)
    Jtrue_blk_ts_mean = Jtrue_blk_ts_full.mean(dim=1)

    H_pre = (E_blk_ts_mean.T @ Jtrue_blk_ts_mean) / float(T)
    G_pre = (Spre_blk_ts_mean.T @ Jtrue_blk_ts_mean) / float(T)

    Hf = H_pre.detach().cpu().numpy().ravel()
    Gf = G_pre.detach().cpu().numpy().ravel()

    def _pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / den)

    def _spearman(a, b):
        ar = np.argsort(np.argsort(a)).astype(np.float64)
        br = np.argsort(np.argsort(b)).astype(np.float64)
        ar -= ar.mean()
        br -= br.mean()
        den = np.linalg.norm(ar) * np.linalg.norm(br) + 1e-12
        return float(np.dot(ar, br) / den)

    pear = _pearson(Hf, Gf)
    spear = _spearman(Hf, Gf)
    print(f"[Eligibility Pre] Pearson: {pear:.4f}, Spearman: {spear:.4f}")

    os.makedirs(str(tmp_path), exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axs[0].imshow(H_pre.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    axs[0].set_title("H_pre = ⟨E_block · J_true⟩_t")
    axs[0].set_xlabel("Post block j")
    axs[0].set_ylabel("Pre block i")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(G_pre.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    axs[1].set_title("G_pre = ⟨S_prev_block · J_true⟩_t")
    axs[1].set_xlabel("Post block j")
    axs[1].set_ylabel("Pre block i")
    fig.colorbar(im1, ax=axs[1])
    heat_path = os.path.join(str(tmp_path), "eligibility_pre_heatmaps.png")
    plt.savefig(heat_path, dpi=150)
    plt.close(fig)
    print(f"[Eligibility Pre] Saved heatmaps to: {heat_path}")

    # Scatter
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(Hf, Gf, s=10, alpha=0.6)
    ax2.set_xlabel("H_pre (E x J_true)")
    ax2.set_ylabel("G_pre (S_prev x J_true)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_title(f"H_pre vs G_pre (Pearson {pear:.3f}, Spearman {spear:.3f})")
    scatter_path = os.path.join(str(tmp_path), "eligibility_pre_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig2)

    # Temporal lag estimate τ* per pre-block i: corr(E_i(t), mean_j J_true(t+τ))
    def best_lag(x, y, max_lag):
        # x,y shape [T]
        T = len(x)
        lags = range(-max_lag, max_lag + 1)
        best_tau, best_r = 0, -1.0
        for tau in lags:
            if tau < 0:
                xs = x[-tau:]
                ys = y[: T + tau]
            elif tau > 0:
                xs = x[: T - tau]
                ys = y[tau:]
            else:
                xs = x
                ys = y
            if len(xs) < 3:  # too short
                continue
            xr = xs - xs.mean()
            yr = ys - ys.mean()
            r = float(
                np.dot(xr, yr) / (np.linalg.norm(xr) * np.linalg.norm(yr) + 1e-12)
            )
            if abs(r) > abs(best_r):
                best_r = r
                best_tau = tau
        return best_tau, best_r

    lags_list: list[int] = []
    cors_list: list[float] = []
    for i in range(Nb):
        tau, r = best_lag(E_series[:, i], Jtrue_mean, max_lag)
        lags_list.append(tau)
        cors_list.append(r)
    lags = np.array(lags_list)
    cors = np.array(cors_list)
    print(
        f"[Eligibility Lag] mean τ*={lags.mean():.2f}, median τ*={np.median(lags):.2f}, mean |corr|={np.mean(np.abs(cors)):.3f}"
    )

    plt.figure(figsize=(6, 4))
    bins_array = np.arange(-max_lag - 0.5, max_lag + 1.5, 1.0)
    plt.hist(lags, bins=bins_array.tolist(), edgecolor="k", alpha=0.8)
    plt.title("Distribution of best lag τ* (E vs mean J_true)")
    plt.xlabel("τ* (steps)")
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(str(tmp_path), "eligibility_best_lag_hist.png"), dpi=150)
    plt.close()

    assert np.isfinite(Hf).all() and np.isfinite(Gf).all()
```
```./test_feedback.py
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

    cs_head = Feedback(
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

        cs_head.align(local_signal=x, target_costate=target)
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

    steps = 500
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
```
```./test_input_saturation.py
import torch
from sbb.const import DEVICE
from sbb.paradigms.predictive_coding import SupervisedConfig
from sbb.base import BaseModel


def test_input_projection_not_saturated():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    # Small but representative core
    cfg = SupervisedConfig(
        num_blocks=32,
        neurons_per_block=32,
        input_features=1,
        output_features=1,
        batch_size=8,
        dtype=dtype,
        seed=123,
        noise=0.0,
    )
    integ = BaseModel(cfg).to(device, dtype)
    # Synthetic inputs with variance similar to tasks (MG/NARMA in [-0.9,0.9])
    T = 256
    x = (
        torch.rand(T, cfg.batch_size, cfg.input_features, device=DEVICE, dtype=dtype)
        - 0.5
    ) * 1.8
    # Pre-tanh activations (no nonlinearity)
    z = x @ integ.machine.weight_in.T
    frac_pre = (z.abs() > 2.0).float().mean().item()  # tanh(|z|=2)≈0.964
    # Projected input actually used
    y = torch.tanh(z)
    frac_post = (y.abs() > 0.98).float().mean().item()
    # Assert: large swaths of network must remain in linear/weakly nonlinear regime
    assert frac_pre <= 0.15, f"Pre-tanh saturation too high: {frac_pre:.3f}"
    assert frac_post <= 0.50, f"Tanh output saturation too high: {frac_post:.3f}"


def test_state_gain_not_collapsed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    cfg = SupervisedConfig(
        num_blocks=32,
        neurons_per_block=32,
        input_features=3,
        output_features=3,
        batch_size=8,
        dtype=dtype,
        seed=321,
        noise=0.0,
    )
    integ = BaseModel(cfg).to(device, dtype)
    T = 128
    x = (
        torch.randn(T, cfg.batch_size, cfg.input_features, device=DEVICE, dtype=dtype)
        * 0.5
    )
    st0 = integ.new_state(cfg.batch_size)
    stf, traj = integ.forward(x, st0)
    # Post-gain proxy: 1 - s^2; if tanh is saturating everywhere this collapses near 0
    gain = 1.0 - traj.pow(2)
    mean_gain = gain.mean().item()
    frac_low_gain = (gain < 0.1).float().mean().item()
    assert mean_gain >= 0.2, f"Mean post-gain too small: {mean_gain:.3f}"
    assert frac_low_gain <= 0.5, f"Too many units with tiny gain: {frac_low_gain:.3f}"
```
```./test_memory_capacity_small.py
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
```
```./test_neurotrophic.py
import os
import numpy as np
import torch
import matplotlib
from sbb.const import DEVICE
from tests.common import auroc, average_precision

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from sbb.paradigms.predictive_coding import (
    PredictiveCoding,
    SupervisedConfig,
)


def test_trophic_map_predicts_growth(tmp_path):
    """
    Tracks AUROC/AP of trophic_support_map predicting newly grown edges.
    Saves AUROC timeline and a heatmap snapshot.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    Nb, BS = 32, 32
    Fi, Fo = 3, 3
    B = 32
    steps = 10000

    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=7,
        noise=0.0,
        trophic_map_ema_alpha=1e-10,
        structural_plasticity=True,
    )
    model = PredictiveCoding(cfg).to(device, dtype)
    model.train()

    X = torch.randn(steps + 1, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(steps + 1, B, Fo, device=DEVICE, dtype=dtype)

    state = model.base.new_state(B)

    aurocs = []
    aps = []
    growth_counts = []
    nb = Nb

    for t in range(steps):
        prev_active = model.base.active_blocks.clone()
        rows_prev = model.base.weight_rows.clone()
        cols_prev = model.base.weight_cols.clone()

        troph = model.base.trophic_support_map.clone()  # [Nb,Nb]
        troph_np = troph.detach().cpu().numpy()

        x = X[t]
        y = Y[t]
        pred, st_next = model.forward(x, state)
        _, state = model.backward(pred, y, state, st_next)

        curr_active = model.base.active_blocks.clone()
        rows_curr = model.base.weight_rows.clone()
        cols_curr = model.base.weight_cols.clone()

        # Build (i,j) active sets before and after
        def set_from(active_mask, rows, cols):
            idx = active_mask.nonzero(as_tuple=True)[0]
            r = rows[idx].detach().cpu().numpy()
            c = cols[idx].detach().cpu().numpy()
            return set((int(ri), int(ci)) for ri, ci in zip(r, c))

        S_prev = set_from(prev_active, rows_prev, cols_prev)
        S_curr = set_from(curr_active, rows_curr, cols_curr)

        grown = S_curr.difference(S_prev)
        # Candidates: not diagonal, not already active before
        candidates = [
            (i, j)
            for i in range(nb)
            for j in range(nb)
            if i != j and (i, j) not in S_prev
        ]

        if len(candidates) == 0:
            aurocs.append(np.nan)
            aps.append(np.nan)
            growth_counts.append(0)
            continue

        scores = np.array([troph_np[i, j] for (i, j) in candidates], dtype=np.float64)
        labels = np.array(
            [1 if (i, j) in grown else 0 for (i, j) in candidates], dtype=np.int32
        )
        growth_counts.append(int(labels.sum()))
        auc = auroc(scores, labels)
        ap = average_precision(scores, labels)
        aurocs.append(auc)
        aps.append(ap)

    # Logging and plots
    print(
        f"[GrowthPredict] Steps with growth: {sum(1 for g in growth_counts if g>0)} / {steps}"
    )
    print(
        f"[GrowthPredict] AUROC median: {np.nanmedian(aurocs):.3f} | AP median: {np.nanmedian(aps):.3f}"
    )

    plt.figure(figsize=(10, 4))
    plt.plot(aurocs, label="AUROC")
    plt.plot(aps, label="AP")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Trophic Map Predictiveness over Time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            str(tmp_path),
            "trophic_predictiveness_timeline.png",
        ),
        dpi=150,
    )
    plt.close()

    # Snapshot heatmap at last step
    troph_final = model.base.trophic_support_map.detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(troph_final, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Trophic Support Map (final snapshot)")
    plt.xlabel("Post block j")
    plt.ylabel("Pre block i")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            str(tmp_path),
            "trophic_map_final.png",
        ),
        dpi=150,
    )
    plt.close()

    # At least we produced finite results; don't fail if no growth happened (skip)
    if all(g == 0 for g in growth_counts):
        pytest.skip(
            "No growth events observed in this short run; diagnostic plots still saved."
        )
    assert np.isfinite(np.array(aurocs, dtype=np.float64)).any()


def test_structural_edge_survival(tmp_path):
    """
    Runs a modest training with active growth and pruning, tracks edge lifetimes.
    Stratifies survival by trophic score-at-birth quartiles.
    Saves survival curves and a scatter of (score, lifetime).
    """
    device = torch.device("cuda")
    dtype = torch.float32
    Nb, BS = 32, 32
    Fi, Fo = 3, 3
    B = 32
    steps = 10000

    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=11,
        noise=0.0,
        trophic_map_ema_alpha=1e-10,
        structural_plasticity=True,
    )
    model = PredictiveCoding(cfg).to(device, dtype)
    model.train()

    X = torch.randn(steps + 1, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(steps + 1, B, Fo, device=DEVICE, dtype=dtype)

    state = model.base.new_state(B)

    # Track (i,j) -> birth step and score at birth; and death steps
    births = {}  # (i,j) -> (t_birth, score_at_birth)
    deaths = {}  # (i,j) -> t_death

    def active_set():
        gi = model.base
        idx = gi.active_blocks.nonzero(as_tuple=True)[0]
        rows = gi.weight_rows[idx].detach().cpu().numpy()
        cols = gi.weight_cols[idx].detach().cpu().numpy()
        return set((int(r), int(c)) for r, c in zip(rows, cols))

    prev = active_set()

    for t in range(steps):
        troph = model.base.trophic_support_map.detach().cpu().numpy()
        x = X[t]
        y = Y[t]
        pred, st_next = model.forward(x, state)
        _, state = model.backward(pred, y, state, st_next)

        curr = active_set()
        grown = curr.difference(prev)
        pruned = prev.difference(curr)

        for i, j in grown:
            if i == j:  # ignore diagonal
                continue
            # record birth if new
            if (i, j) not in births:
                births[(i, j)] = (t, float(troph[i, j]))
        for i, j in pruned:
            if (i, j) in births and (i, j) not in deaths:
                deaths[(i, j)] = t
        prev = curr

    lifetimes: list[int] = []
    scores: list[float] = []
    for k, (tb, s) in births.items():
        td = deaths.get(k, steps)  # right-censored at steps
        lifetimes.append(td - tb)
        scores.append(s)

    if len(lifetimes) < 5:
        pytest.skip(
            "Not enough growth/prune events for survival analysis in this short run."
        )

    lifetimes = np.array(lifetimes, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)

    # Stratify by score-at-birth quartiles
    qs = np.quantile(scores, [0.25, 0.5, 0.75])
    strata = {
        "Q1 (low)": (scores <= qs[0]),
        "Q2-Q3 (mid)": ((scores > qs[0]) & (scores <= qs[2])),
        "Q4 (high)": (scores > qs[2]),
    }

    # Empirical survival S(k) = P(lifetime >= k)
    kmax = max(1, int(lifetimes.max()))
    ks = np.arange(1, kmax + 1)
    plt.figure(figsize=(8, 5))
    for name, mask in strata.items():
        if mask.sum() < 3:
            continue
        lt = lifetimes[mask]
        S = [(lt >= k).mean() for k in ks]
        plt.plot(ks, S, label=f"{name} (n={mask.sum()})")
    plt.xlabel("k (steps since birth)")
    plt.ylabel("Survival S(k)")
    plt.title("Edge Survival by Trophic Score-at-Birth")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            str(tmp_path),
            "survival_curves.png",
        ),
        dpi=150,
    )
    plt.close()

    # Scatter: score vs lifetime
    plt.figure(figsize=(6, 5))
    plt.scatter(scores, lifetimes, s=14, alpha=0.6)
    plt.xlabel("Score at birth (trophic)")
    plt.ylabel("Lifetime (steps)")
    plt.title("Edge lifetime vs trophic score at birth")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            str(tmp_path),
            "score_vs_lifetime.png",
        ),
        dpi=150,
    )
    plt.close()

    corr = np.corrcoef(scores, lifetimes)[0, 1]
    print(
        f"[Survival] Pearson(score, lifetime) = {corr:.3f} over {len(lifetimes)} edges"
    )
    assert np.isfinite(corr)


def test_trophic_field_baseline(tmp_path):
    """
    Diagnostic: how effective is the current TPE/trophic post-side heuristic?

    We compare two block-to-block maps over a short, no-learning run:
      - H_post[i,j] = ⟨ E_block[i,t] * J_heur[j,t] ⟩_t
      - G_post[i,j] = ⟨ E_block[i,t] * J_true[j,t] ⟩_t
    where:
      E_block[i,t] = block-mean eligibility trace (post-step)
      J_heur[j,t]  = block-mean(∂ℓ/∂s_{t+1})             (current heuristic)
      J_true[j,t]  = block-mean((1 - s_{t+1}^2) ⊙ ∂ℓ/∂s_{t+1})  (principled)

    We then plot heatmaps of H_post and G_post, and a scatter with
    Pearson/Spearman correlations.
    """
    device = torch.device("cuda")
    dtype = torch.float32

    Nb, BS = 32, 32
    Fi, Fo = 3, 3
    B, T = 16, 100

    # TODO: Re-enable homeostasis (remove activity_lr override) to validate claims.
    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=321,
        noise=0.0,
        trophic_map_ema_alpha=1e-10,
        structural_plasticity=True,
    )
    cfg.activity_lr = 0.0  # TODO: Remove this override

    model = PredictiveCoding(cfg=cfg).to(device, dtype)
    model.eval()

    def block_means(vec_N: torch.Tensor):
        return vec_N.view(B, Nb, BS).mean(dim=2)

    E_blk_ts: list[torch.Tensor] = []
    Jheur_blk_ts: list[torch.Tensor] = []
    Jtrue_blk_ts: list[torch.Tensor] = []

    X = torch.randn(T, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(T, B, Fo, device=DEVICE, dtype=dtype)

    state = model.base.new_state(B)

    with torch.no_grad():
        for t in range(T):
            x_t = X[t]
            y_t = Y[t]
            y_pred, next_state = model.forward(x_t, state)

            error = y_pred - y_t
            dL_dS = error @ model.readout.weight
            post_gain = 1.0 - next_state.activations.pow(2)
            dL_dS_true = dL_dS * post_gain

            E_blk = block_means(next_state.eligibility_trace)
            Jheur_blk = block_means(dL_dS)
            Jtrue_blk = block_means(dL_dS_true)

            E_blk_ts.append(E_blk)
            Jheur_blk_ts.append(Jheur_blk)
            Jtrue_blk_ts.append(Jtrue_blk)

            state = next_state

    E_blk_tensor = torch.stack(E_blk_ts, dim=0).mean(dim=1)
    Jheur_blk_tensor = torch.stack(Jheur_blk_ts, dim=0).mean(dim=1)
    Jtrue_blk_tensor = torch.stack(Jtrue_blk_ts, dim=0).mean(dim=1)

    H_post = (E_blk_tensor.T @ Jheur_blk_tensor) / float(T)
    G_post = (E_blk_tensor.T @ Jtrue_blk_tensor) / float(T)

    Hf = H_post.detach().cpu().numpy().ravel()
    Gf = G_post.detach().cpu().numpy().ravel()

    def _pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / den)

    def _spearman(a, b):
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        ar = ar.astype(np.float64)
        br = br.astype(np.float64)
        ar -= ar.mean()
        br -= br.mean()
        den = np.linalg.norm(ar) * np.linalg.norm(br) + 1e-12
        return float(np.dot(ar, br) / den)

    pear = _pearson(Hf, Gf)
    spear = _spearman(Hf, Gf)
    print(f"[Trophic Post] Pearson: {pear:.4f}, Spearman: {spear:.4f}")

    os.makedirs(str(tmp_path), exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axs[0].imshow(H_post.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    axs[0].set_title("H_post = ⟨E_block · (∂ℓ/∂s)⟩_t")
    axs[0].set_xlabel("Post block j")
    axs[0].set_ylabel("Pre block i")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(G_post.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    axs[1].set_title("G_post = ⟨E_block · ((1−s^2)⊙∂ℓ/∂s)⟩_t")
    axs[1].set_xlabel("Post block j")
    axs[1].set_ylabel("Pre block i")
    fig.colorbar(im1, ax=axs[1])

    heat_path = os.path.join(
        str(tmp_path),
        "trophic_post_heatmaps.png",
    )
    plt.savefig(heat_path, dpi=150)
    plt.close(fig)
    print(f"[Trophic Post] Saved heatmaps to: {heat_path}")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(Hf, Gf, s=10, alpha=0.6)
    ax2.set_xlabel("H_post (E x ∂ℓ/∂s)")
    ax2.set_ylabel("G_post (E x (1−s^2)⊙∂ℓ/∂s)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_title(f"H_post vs G_post (Pearson {pear:.3f}, Spearman {spear:.3f})")

    if np.std(Hf) > 1e-12:
        m = np.dot(Hf - Hf.mean(), Gf - Gf.mean()) / (np.var(Hf) * len(Hf) + 1e-12)
        b = Gf.mean() - m * Hf.mean()
        xs = np.linspace(Hf.min(), Hf.max(), 100)
        ax2.plot(xs, m * xs + b, color="tab:red", alpha=0.7, lw=2)

    scatter_path = os.path.join(
        str(tmp_path),
        "trophic_post_scatter.png",
    )
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig2)
    print(f"[Trophic Post] Saved scatter to: {scatter_path}")

    assert np.isfinite(Hf).all() and np.isfinite(Gf).all()
```
```./test_recurrent.py
import os
import numpy as np
import torch
import matplotlib

from tests.common import (
    build_dense_W,
    power_radius,
    active_block_mask,
    block_map_from_weights_delta,
    cosine,
    participation_ratio,
    corr,
    top_k_overlap,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbb.paradigms.predictive_coding import (
    PredictiveCoding,
    SupervisedConfig,
)
from sbb.const import DEVICE, EPS


def test_weight_update_alignment_surrogate(tmp_path):
    """
    Improved diagnostic surrogate for ΔW:
      - Uses the same ingredients as the kernel:
          Hebbian (eligibility) term
          Oja term
          Adaptive decay term
      - Compares per-block Frobenius |ΔW| to the surrogate magnitude on ACTIVE edges.
    Saves heatmaps and scatter; prints correlation, cosine, and P@10%.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    Nb, BS = 32, 32
    Fi, Fo = 3, 3
    B = 32
    T_warmup = 100

    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=42,
        noise=0.0,
        structural_plasticity=False,  # freeze topology
    )
    model = PredictiveCoding(cfg).to(device, dtype)
    model.train()

    X = torch.randn(T_warmup + 2, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(T_warmup + 2, B, Fo, device=DEVICE, dtype=dtype)

    state = model.base.new_state(B)

    # Warmup to align heads and costate
    for t in range(T_warmup):
        x = X[t]
        y = Y[t]
        y_pred, st_next = model.forward(x, state)
        _, state = model.backward(y_pred, y, state, st_next)

    # Snapshot W before
    with torch.no_grad():
        W_before = build_dense_W(model).clone()

    # Measurement step: get ingredients for surrogate
    x = X[T_warmup]
    y = Y[T_warmup]
    s_prev = state.activations.clone()  # [B,N]
    y_pred, st_next = model.forward(x, state)
    error = y_pred - y  # [B, Fo]
    W_out = model.readout.weight  # [Fo, N]
    dL_dS = error @ W_out  # [B, N]  (λ* proxy)
    post_gain = 1.0 - st_next.activations.pow(2)  # [B,N]
    lam_eff = dL_dS * post_gain  # [B,N]  (≈ λ̂ ⊙ gate)
    inv_norms = 1.0 / (s_prev.pow(2).sum(dim=1, keepdim=True) + EPS)  # [B,1]

    # Aggregate to per-neuron vectors like the kernel does
    mod_vec = (lam_eff * inv_norms).sum(dim=0)  # [N]
    hebb_vec = (st_next.eligibility_trace * inv_norms).sum(dim=0)  # [N]
    oja_vec = (s_prev * inv_norms).sum(dim=0)  # [N]

    # Perform one learning step so we can measure ΔW
    _, state = model.backward(y_pred, y, state, st_next)

    with torch.no_grad():
        W_after = build_dense_W(model).clone()

    # Build surrogate per block using kernel-inspired forms
    sur_map = torch.zeros(Nb, Nb, device=DEVICE, dtype=dtype)
    # Use a typical pruning threshold value (dynamically computed in practice)
    pruning_threshold = 0.1
    for i in range(Nb):
        r0, r1 = i * BS, (i + 1) * BS
        hebb_i = hebb_vec[r0:r1]  # [BS]
        oja_i = oja_vec[r0:r1]  # [BS]
        for j in range(Nb):
            c0, c1 = j * BS, (j + 1) * BS
            hebb_j = hebb_vec[c0:c1]  # [BS]
            oja_j = oja_vec[c0:c1]  # [BS]
            Wb = W_before[r0:r1, c0:c1]  # [BS, BS]

            # Modulatory gate (post-synaptic, broadcast to all pre)
            mod_j = mod_vec[c0:c1]  # [BS]
            modulatory_gate = torch.tanh(mod_j).unsqueeze(0)  # [1, BS]

            # Hebbian term
            hebb_delta = torch.outer(hebb_i, hebb_j)  # [BS, BS]
            hebbian_term = hebb_delta / (BS * BS)

            # Oja term
            oja_delta = oja_i.unsqueeze(1) * (
                oja_j.unsqueeze(0) - oja_i.unsqueeze(1) * Wb
            )
            oja_term = oja_delta / BS

            # Gated plasticity
            gated_plasticity = modulatory_gate * (hebbian_term + oja_term)

            # Decay term (matches kernel lines 110-115, 120-123)
            wnorm_sq = (Wb * Wb).sum()
            block_magnitude = torch.sqrt(wnorm_sq + EPS)
            fixed_decay = -Wb
            mag_strength = block_magnitude / (pruning_threshold + EPS)
            mag_decay = -Wb * mag_strength * 0.1
            decay_delta = fixed_decay * 0.9 + mag_decay * 0.1
            block_capacity_sq = (BS * BS) * (BS * BS)
            decay_term = decay_delta / block_capacity_sq

            # Combine (matches kernel line 128)
            tot = gated_plasticity + decay_term
            sur_map[i, j] = torch.linalg.norm(tot)

    # Produce target ΔW map
    dW_map = block_map_from_weights_delta(W_before, W_after, Nb, BS)

    # Mask to ACTIVE edges only to avoid swamping with zeros
    act_mask_t = active_block_mask(model, Nb)
    act_mask = act_mask_t.detach().cpu().numpy().astype(bool)

    dW_active = dW_map.detach().cpu().numpy()[act_mask]
    sur_active = sur_map.detach().cpu().numpy()[act_mask]

    pear, spear = corr(dW_active, sur_active)
    cos = cosine(dW_active, sur_active)
    p10 = top_k_overlap(sur_active, dW_active, frac=0.1)
    print(
        f"[ΔW surrogate (active)] Pearson={pear:.3f} Spearman={spear:.3f} Cosine={cos:.3f} P@10%={p10:.3f}"
    )

    # Heatmaps (full, for visual context)
    def heat(mat, title, fname):
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Pre block j")
        plt.ylabel("Post block i")
        plt.tight_layout()
        plt.savefig(os.path.join(str(tmp_path), fname), dpi=150)
        plt.close()

    heat(
        dW_map.detach().cpu().numpy(),
        "ΔW block Frobenius (one step)",
        "deltaW_blocks.png",
    )
    heat(
        sur_map.detach().cpu().numpy(),
        "Kernel-inspired surrogate",
        "surrogate_blocks.png",
    )

    # Scatter (active only)
    plt.figure(figsize=(6, 6))
    plt.scatter(dW_active, sur_active, s=14, alpha=0.65)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("|ΔW| block (active only)")
    plt.ylabel("Surrogate score (active only)")
    plt.title("ΔW vs Kernel-inspired surrogate (active edges)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(str(tmp_path), "scatter_deltaW_vs_surrogate_ACTIVE.png"), dpi=150
    )
    plt.close()

    # Robust sanity check
    assert np.isfinite(dW_map.detach().cpu().numpy()).all()
    assert np.isfinite(sur_map.detach().cpu().numpy()).all()


def test_state_participation_ratio_over_time(tmp_path):
    """
    Tracks intrinsic dimensionality (participation ratio) of the latent state over time.
    Saves a timeline plot.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    Nb, BS = 32, 32
    Fi, Fo = 3, 3
    B = 32
    steps = 200

    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=13,
        noise=0.0,
        structural_plasticity=False,
    )
    model = PredictiveCoding(cfg).to(device, dtype)
    model.train()

    X = torch.randn(steps, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(steps, B, Fo, device=DEVICE, dtype=dtype)
    state = model.base.new_state(B)

    pr_values = []
    window_states = []

    for t in range(steps):
        pred, st_next = model.forward(X[t], state)
        _, state = model.backward(pred, Y[t], state, st_next)
        # Collect mean-centered states (over batch)
        s = st_next.activations.detach()  # [B, N]
        window_states.append(s)
        if len(window_states) > 20:  # rolling window to keep cost low
            window_states.pop(0)
        S = torch.cat(window_states, dim=0)  # [(<=20*B), N]
        S = S - S.mean(dim=0, keepdim=True)
        C = (S.T @ S) / max(1, (S.shape[0] - 1))
        pr = participation_ratio(C)
        pr_values.append(pr)

    plt.figure(figsize=(8, 4))
    plt.plot(pr_values)
    plt.xlabel("Step")
    plt.ylabel("Participation ratio (d_eff)")
    plt.title("State intrinsic dimensionality over time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(str(tmp_path), "participation_ratio_timeline.png"), dpi=150
    )
    plt.close()

    pr_tensor = torch.tensor(pr_values)
    print(
        f"[StateGeometry] mean d_eff={pr_tensor.mean():.2f}, min={pr_tensor.min():.2f}, max={pr_tensor.max():.2f}"
    )
    assert torch.isfinite(pr_tensor).all()


def test_online_jacobian_spectral_radius(tmp_path):
    """
    Approximates Jacobian J_t ≈ α I + (1−α) diag(1−s_t^2) W over a short run.
    Plots spectral radius over time.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    Nb, BS = 32, 32
    Fi, Fo = 3, 3
    B = 16
    steps = 50

    cfg = SupervisedConfig(
        num_blocks=Nb,
        neurons_per_block=BS,
        input_features=Fi,
        output_features=Fo,
        batch_size=B,
        dtype=dtype,
        seed=5,
        noise=0.0,
        structural_plasticity=False,  # keep weights more stable for this diagnostic
    )
    model = PredictiveCoding(cfg).to(device, dtype)
    model.train()

    X = torch.randn(steps, B, Fi, device=DEVICE, dtype=dtype)
    Y = torch.randn(steps, B, Fo, device=DEVICE, dtype=dtype)
    state = model.base.new_state(B)

    alphas = []
    radii = []
    for t in range(steps):
        pred, st_next = model.forward(X[t], state)
        # one learn step to keep activity realistic but mild
        _, state = model.backward(pred, Y[t], state, st_next)

        s_t = st_next.activations.mean(dim=0)  # average over batch
        g_t = 1.0 - s_t.pow(2)  # diag elements
        W = build_dense_W(model)
        alpha = torch.exp(
            torch.tensor(
                -cfg.time_step_delta / (cfg.tau_fast + 1e-12),
                device=DEVICE,
                dtype=dtype,
            )
        )
        A = alpha * torch.eye(W.shape[0], device=DEVICE, dtype=dtype) + (
            1.0 - alpha
        ) * (g_t * W)
        rad = power_radius(A)
        alphas.append(float(alpha.item()))
        radii.append(rad)

    plt.figure(figsize=(8, 4))
    plt.plot(radii, label="ρ(J_t)")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Step")
    plt.ylabel("Spectral radius")
    plt.title("Approx. Jacobian ρ over time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(str(tmp_path), "jacobian_radius_timeline.png"), dpi=150)
    plt.close()

    print(
        f"[JacobianMonitor] ρ mean={np.mean(radii):.3f} min={np.min(radii):.3f} max={np.max(radii):.3f}"
    )
    assert np.isfinite(np.array(radii)).all()
```
