import gc
import pytest
import torch
import time

from sbb.const import DEVICE
from sbb.paradigms.predictive_coding import SupervisedConfig
from sbb.paradigms.predictive_coding import PredictiveCoding


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_accumulation = True
torch.backends.cudnn.allow_tf32 = True


def format_hertz(sps):
    """Formats a number into Hz, kHz, MHz, or GHz."""
    if sps >= 1_000_000_000:
        return f"{sps / 1_000_000_000:.2f} G"
    if sps >= 1_000_000:
        return f"{sps / 1_000_000:.2f} M"
    if sps >= 1_000:
        return f"{sps / 1_000:.2f} k"
    return f"{sps:.2f} "


def run_full_loop(model, total_steps, warmup_steps):
    """
    Runs a full forward/learn loop for performance measurement.
    This function is used for the high-level SPS benchmark.
    """

    dtype = model.cfg.dtype
    batch_size = model.cfg.batch_size
    input_features = model.cfg.input_features
    output_features = model.cfg.output_features

    inputs = [
        torch.randn(batch_size, input_features, device=DEVICE, dtype=dtype)
        for _ in range(total_steps)
    ]
    targets = [
        torch.randn(batch_size, output_features, device=DEVICE, dtype=dtype)
        for _ in range(total_steps)
    ]

    state_tuple = model.base.new_state(batch_size)
    density_sum = torch.tensor(0.0, device=DEVICE, dtype=dtype)
    measured_steps = 0

    for t in range(total_steps):
        u = inputs[t]
        y_target = targets[t]

        pred, next_state_tuple = model.forward(u, state_tuple)

        if t >= warmup_steps:
            _, state_tuple = model.backward(
                pred, y_target, state_tuple, next_state_tuple
            )

            num_active_blocks = model.base.active_blocks.sum()
            density = num_active_blocks / (model.cfg.num_blocks**2)
            density_sum = density_sum + density
            measured_steps += 1
        else:
            state_tuple = next_state_tuple

    if measured_steps > 0:
        avg_density = float((density_sum / measured_steps).item())
    else:
        avg_density = 0.0
    return avg_density


@pytest.mark.parametrize("nerons_per_block", [32])
@pytest.mark.parametrize("num_blocks", [10240])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("evolution_substeps", [16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_internal_sps_benchmark(
    nerons_per_block, num_blocks, batch_size, evolution_substeps
):
    """
    Measures canonical throughput and latency.
    Throughput: Number of individual items processed per second (items/sec).
    Latency: Time required to process a single batch (ms/batch).
    """
    cfg = SupervisedConfig(
        num_blocks=num_blocks,
        neurons_per_block=nerons_per_block,
        batch_size=batch_size,
        input_features=2,
        output_features=1,
        seed=42,
        evolution_substeps=evolution_substeps,
    )
    model = PredictiveCoding(cfg=cfg)
    model.train()

    total_steps = 100
    warmup_steps = 10

    run_full_loop(model, warmup_steps, 0)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    avg_density = run_full_loop(model, total_steps, warmup_steps)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start_time

    measured_steps = total_steps - warmup_steps

    throughput_items_per_sec = (measured_steps * cfg.batch_size) / duration

    latency_ms_per_batch = (duration / measured_steps) * 1000

    formatted_throughput = format_hertz(throughput_items_per_sec)

    print(
        f"N: {nerons_per_block*num_blocks} | N/B: {nerons_per_block} | Blks: {num_blocks} | Batch: {batch_size} | Res: {evolution_substeps}  | "
        f"Throughput: {formatted_throughput}items/s | Latency: {latency_ms_per_batch:.4f} ms/batch | Avg Density: {avg_density:.4f}"
    )

    min_throughput = 100.0
    max_throughput = 100000.0
    min_latency = 1.0
    max_latency = 100.0
    gc.collect()
    assert (
        throughput_items_per_sec > min_throughput
    ), f"Throughput ({formatted_throughput}items/s) is below baseline ({format_hertz(min_throughput)}items/s)"

    assert (
        max_throughput > throughput_items_per_sec
    ), f"Throughput ({formatted_throughput}items/s) is above maximum ({format_hertz(max_throughput)}items/s)"
    assert (
        latency_ms_per_batch > min_latency
    ), f"Latency ({latency_ms_per_batch}ms/batch) is below baseline ({min_latency} ms/batch)"
    assert (
        max_latency > latency_ms_per_batch
    ), f"Latency ({latency_ms_per_batch}ms/batch) is above baseline ({max_latency} ms/batch)"


def main():
    """
    Runs performance benchmarks across various configurations and generates visualizations.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    # Configuration space to benchmark
    # Each (neurons_per_block, num_blocks) combination is a unique "layer" in the 3D visualization
    neurons_per_block_values = [16, 32, 64]
    num_blocks_values = [1024, 2048, 5120]
    batch_size_values = [1, 8, 16, 32, 64]
    evolution_substeps_values = [1, 8, 16, 32]

    # Generate all combinations
    configs = [
        {
            "neurons_per_block": npb,
            "num_blocks": nb,
            "batch_size": bs,
            "evolution_substeps": es,
        }
        for npb in neurons_per_block_values
        for nb in num_blocks_values
        for bs in batch_size_values
        for es in evolution_substeps_values
    ]

    results = []

    print("=" * 80)
    print("Running Performance Benchmarks")
    print("=" * 80)

    for i, cfg_dict in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing configuration: {cfg_dict}")

        cfg = SupervisedConfig(
            num_blocks=cfg_dict["num_blocks"],
            neurons_per_block=cfg_dict["neurons_per_block"],
            batch_size=cfg_dict["batch_size"],
            input_features=2,
            output_features=1,
            seed=42,
            evolution_substeps=cfg_dict["evolution_substeps"],
        )
        model = PredictiveCoding(cfg=cfg)
        model.train()

        total_steps = 100
        warmup_steps = 10

        # Warmup
        run_full_loop(model, warmup_steps, 0)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        avg_density = run_full_loop(model, total_steps, warmup_steps)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start_time

        measured_steps = total_steps - warmup_steps
        throughput = (measured_steps * cfg.batch_size) / duration
        latency = (duration / measured_steps) * 1000
        total_neurons = cfg_dict["neurons_per_block"] * cfg_dict["num_blocks"]

        results.append(
            {
                **cfg_dict,
                "total_neurons": total_neurons,
                "throughput": throughput,
                "latency": latency,
                "density": avg_density,
            }
        )

        print(
            f"  → Throughput: {format_hertz(throughput)}items/s | Latency: {latency:.2f}ms | Density: {avg_density:.4f}"
        )

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Create visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Performance Benchmarks: Predictive Coding Network",
        fontsize=16,
        fontweight="bold",
    )

    # Extract data for easier manipulation
    neuron_counts = np.array([r["total_neurons"] for r in results])
    throughputs = np.array([r["throughput"] for r in results])
    latencies = np.array([r["latency"] for r in results])
    np.array([r["density"] for r in results])
    batch_sizes = np.array([r["batch_size"] for r in results])
    substeps = np.array([r["evolution_substeps"] for r in results])

    # Plot 1: Throughput vs Network Size (sized by substeps, colored by batch size)
    ax1 = axes[0, 0]
    for bs in [1, 32]:
        for substep in [1, 16]:
            mask = (batch_sizes == bs) & (substeps == substep)
            neurons = neuron_counts[mask]
            tput = throughputs[mask]
            marker_size = 50 if substep == 1 else 150
            marker = "o" if substep == 1 else "^"
            ax1.scatter(
                neurons,
                tput,
                s=marker_size,
                alpha=0.6,
                label=f"B={bs}, S={substep}",
                marker=marker,
            )
    ax1.set_xlabel("Total Neurons", fontsize=12)
    ax1.set_ylabel("Throughput (items/s)", fontsize=12)
    ax1.set_title("Throughput vs Network Size", fontsize=13, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)

    # Plot 2: Latency vs Network Size (sized by batch size, colored by evolution substeps)
    ax2 = axes[0, 1]
    for substep in [1, 16]:
        for bs in [1, 32]:
            mask = (substeps == substep) & (batch_sizes == bs)
            neurons = neuron_counts[mask]
            lat = latencies[mask]
            marker_size = 50 if bs == 1 else 150
            marker = "s" if bs == 1 else "D"
            ax2.scatter(
                neurons,
                lat,
                s=marker_size,
                alpha=0.6,
                label=f"S={substep}, B={bs}",
                marker=marker,
            )
    ax2.set_xlabel("Total Neurons", fontsize=12)
    ax2.set_ylabel("Latency (ms/batch)", fontsize=12)
    ax2.set_title("Latency vs Network Size", fontsize=13, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)

    # Plot 3: Heatmap - Effect of batch_size and substeps on throughput (fixed architecture)
    ax3 = axes[1, 0]

    # Get unique network architectures
    architectures = sorted(
        set((r["neurons_per_block"], r["num_blocks"]) for r in results)
    )
    batch_vals = sorted(set(batch_sizes))
    substep_vals = sorted(set(substeps))

    # Create a throughput matrix for each architecture
    # We'll show the largest architecture as the main heatmap
    npb, nb = architectures[-1]  # Largest network
    throughput_matrix = np.zeros((len(batch_vals), len(substep_vals)))

    for i, bs in enumerate(batch_vals):
        for j, ss in enumerate(substep_vals):
            mask = (batch_sizes == bs) & (substeps == ss) & (neuron_counts == npb * nb)
            if mask.any():
                throughput_matrix[i, j] = throughputs[mask][0]

    im = ax3.imshow(throughput_matrix, cmap="viridis", aspect="auto")
    ax3.set_xticks(range(len(substep_vals)))
    ax3.set_yticks(range(len(batch_vals)))
    ax3.set_xticklabels(substep_vals)
    ax3.set_yticklabels(batch_vals)
    ax3.set_xlabel("Evolution Substeps", fontsize=12)
    ax3.set_ylabel("Batch Size", fontsize=12)
    ax3.set_title(
        f"Throughput (items/s)\nNetwork: {npb*nb:,} neurons ({npb}×{nb})",
        fontsize=13,
        fontweight="bold",
    )

    # Add text annotations
    for i in range(len(batch_vals)):
        for j in range(len(substep_vals)):
            ax3.text(
                j,
                i,
                f"{throughput_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax3, label="Throughput (items/s)")

    # Plot 4: Heatmap - Effect of batch_size and substeps on latency (fixed architecture)
    ax4 = axes[1, 1]

    # Create a latency matrix for the same architecture
    latency_matrix = np.zeros((len(batch_vals), len(substep_vals)))

    for i, bs in enumerate(batch_vals):
        for j, ss in enumerate(substep_vals):
            mask = (batch_sizes == bs) & (substeps == ss) & (neuron_counts == npb * nb)
            if mask.any():
                latency_matrix[i, j] = latencies[mask][0]

    im2 = ax4.imshow(latency_matrix, cmap="plasma", aspect="auto")
    ax4.set_xticks(range(len(substep_vals)))
    ax4.set_yticks(range(len(batch_vals)))
    ax4.set_xticklabels(substep_vals)
    ax4.set_yticklabels(batch_vals)
    ax4.set_xlabel("Evolution Substeps", fontsize=12)
    ax4.set_ylabel("Batch Size", fontsize=12)
    ax4.set_title(
        f"Latency (ms/batch)\nNetwork: {npb*nb:,} neurons ({npb}×{nb})",
        fontsize=13,
        fontweight="bold",
    )

    # Add text annotations
    for i in range(len(batch_vals)):
        for j in range(len(substep_vals)):
            ax4.text(
                j,
                i,
                f"{latency_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

    plt.colorbar(im2, ax=ax4, label="Latency (ms/batch)")

    # Print summary insights to console
    print("\n" + "=" * 80)
    print("Key Insights (for largest network)")
    print("=" * 80)

    # Calculate effects
    batch_effect = throughput_matrix[1, :] / throughput_matrix[0, :]  # batch32 / batch1
    substep_effect = (
        throughput_matrix[:, 1] / throughput_matrix[:, 0]
    )  # substep16 / substep1

    print("\nBatch Size Effect (B=32 vs B=1):")
    print(f"  Throughput improvement: {batch_effect.mean():.2f}x average")
    print(
        f"  Latency change: {(latency_matrix[1, :] / latency_matrix[0, :]).mean():.2f}x average"
    )

    print("\nEvolution Substeps Effect (S=16 vs S=1):")
    print(f"  Throughput change: {substep_effect.mean():.2f}x average")
    print(
        f"  Latency change: {(latency_matrix[:, 1] / latency_matrix[:, 0]).mean():.2f}x average"
    )

    print("\nBest configuration (highest throughput):")
    best_i, best_j = np.unravel_index(
        throughput_matrix.argmax(), throughput_matrix.shape
    )
    print(f"  Batch={batch_vals[best_i]}, Substeps={substep_vals[best_j]}")
    print(f"  Throughput: {throughput_matrix[best_i, best_j]:.1f} items/s")
    print(f"  Latency: {latency_matrix[best_i, best_j]:.1f} ms/batch")

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_benchmark_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to: {filename}")

    # Create "layer cake" 3D visualization

    fig2 = plt.figure(figsize=(18, 10))
    fig2.suptitle(
        "Layer Cake: Throughput & Latency vs (Batch Size, Substeps) for Fixed Network Sizes",
        fontsize=16,
        fontweight="bold",
    )

    # Create meshgrid for batch_size and substeps
    BS, SS = np.meshgrid(batch_vals, substep_vals)

    # Get unique architectures sorted by size
    architectures = sorted(
        set((r["neurons_per_block"], r["num_blocks"]) for r in results)
    )
    n_arch = len(architectures)

    # Two subplots: one for throughput, one for latency
    ax3d_tput = fig2.add_subplot(121, projection="3d")
    ax3d_lat = fig2.add_subplot(122, projection="3d")

    colors = plt.get_cmap("viridis")(np.linspace(0, 1, n_arch))

    for layer_idx, (npb, nb) in enumerate(architectures):
        total_n = npb * nb

        # Build throughput and latency surfaces for this architecture
        tput_surface = np.zeros((len(substep_vals), len(batch_vals)))
        lat_surface = np.zeros((len(substep_vals), len(batch_vals)))

        for i, ss in enumerate(substep_vals):
            for j, bs in enumerate(batch_vals):
                mask = (
                    (batch_sizes == bs) & (substeps == ss) & (neuron_counts == total_n)
                )
                if mask.any():
                    tput_surface[i, j] = throughputs[mask][0]
                    lat_surface[i, j] = latencies[mask][0]

        # Plot throughput surface (layer cake style - offset in Z)
        z_offset_tput = layer_idx * 200  # Vertical spacing between layers
        ax3d_tput.plot_surface(
            BS,
            SS,
            tput_surface + z_offset_tput,
            alpha=0.7,
            cmap="viridis",
            edgecolor="none",
            label=f"{total_n:,}N",
        )

        # Add wireframe for better visibility
        ax3d_tput.plot_wireframe(
            BS,
            SS,
            tput_surface + z_offset_tput,
            color=colors[layer_idx],
            alpha=0.3,
            linewidth=0.5,
        )

        # Plot latency surface
        z_offset_lat = layer_idx * 50  # Vertical spacing between layers
        ax3d_lat.plot_surface(
            BS,
            SS,
            lat_surface + z_offset_lat,
            alpha=0.7,
            cmap="plasma",
            edgecolor="none",
            label=f"{total_n:,}N",
        )

        # Add wireframe
        ax3d_lat.plot_wireframe(
            BS,
            SS,
            lat_surface + z_offset_lat,
            color=colors[layer_idx],
            alpha=0.3,
            linewidth=0.5,
        )

        # Add text label for each layer
        label_x = batch_vals[-1] * 0.8
        label_y = substep_vals[-1] * 0.8
        ax3d_tput.text(
            label_x,
            label_y,
            z_offset_tput,
            f"{npb}×{nb}\n({total_n:,}N)",
            fontsize=9,
            fontweight="bold",
        )
        ax3d_lat.text(
            label_x,
            label_y,
            z_offset_lat,
            f"{npb}×{nb}\n({total_n:,}N)",
            fontsize=9,
            fontweight="bold",
        )

    # Configure throughput plot
    ax3d_tput.set_xlabel("Batch Size", fontsize=11, labelpad=10)
    ax3d_tput.set_ylabel("Evolution Substeps", fontsize=11, labelpad=10)
    ax3d_tput.set_zlabel(
        "Throughput (items/s)\n+ Layer Offset", fontsize=11, labelpad=10
    )
    ax3d_tput.set_title("Throughput Layer Cake", fontsize=13, fontweight="bold", pad=20)
    ax3d_tput.view_init(elev=20, azim=45)

    # Configure latency plot
    ax3d_lat.set_xlabel("Batch Size", fontsize=11, labelpad=10)
    ax3d_lat.set_ylabel("Evolution Substeps", fontsize=11, labelpad=10)
    ax3d_lat.set_zlabel("Latency (ms/batch)\n+ Layer Offset", fontsize=11, labelpad=10)
    ax3d_lat.set_title("Latency Layer Cake", fontsize=13, fontweight="bold", pad=20)
    ax3d_lat.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Save 3D figure
    filename_3d = f"performance_benchmark_3d_{timestamp}.png"
    plt.savefig(filename_3d, dpi=150, bbox_inches="tight")
    print(f"✓ Saved 3D layer cake plot to: {filename_3d}")

    # Save CSV
    csv_filename = f"performance_benchmark_{timestamp}.csv"
    with open(csv_filename, "w") as f:
        f.write(
            "neurons_per_block,num_blocks,batch_size,evolution_substeps,total_neurons,throughput_items_per_s,latency_ms_per_batch,density\n"
        )
        for r in results:
            f.write(
                f"{r['neurons_per_block']},{r['num_blocks']},{r['batch_size']},{r['evolution_substeps']},"
                f"{r['total_neurons']},{r['throughput']:.4f},{r['latency']:.4f},{r['density']:.6f}\n"
            )
    print(f"✓ Saved data to: {csv_filename}")

    plt.show()

    return results


if __name__ == "__main__":
    main()
