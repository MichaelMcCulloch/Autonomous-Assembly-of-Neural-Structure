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
