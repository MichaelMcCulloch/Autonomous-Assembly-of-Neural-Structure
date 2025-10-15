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
