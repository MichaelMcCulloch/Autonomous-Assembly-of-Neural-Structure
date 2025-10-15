import dataclasses
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sbb import BaseConfig

from sbb.paradigms.policy_gradient import (
    ReinforcementLearningConfig,
)
from sbb.paradigms.predictive_coding import (
    SupervisedConfig,
)
from sbb.paradigms.active_inference import (
    ActiveInferenceHyperparameters,
)


def analyze_and_plot_all_parameters():
    """
    Generates and plots all configuration parameters against total network size (N).
    """

    NEURONS_PER_BLOCK = 32
    MIN_NUM_BLOCKS = 32
    MAX_NUM_BLOCKS = 15360
    NUM_SAMPLES = 100
    NON_PLOTTABLE_FIELDS = [
        "device",
        "dtype",
        "initial_connectivity_map_rows",
        "initial_connectivity_map_cols",
    ]

    block_counts = np.unique(
        np.logspace(
            np.log10(MIN_NUM_BLOCKS), np.log10(MAX_NUM_BLOCKS), NUM_SAMPLES, dtype=int
        )
    )

    data: Dict[str, List[Any]] = {}

    for b in block_counts:
        try:
            configs = {
                "base": BaseConfig(num_blocks=b, neurons_per_block=NEURONS_PER_BLOCK),
                "sl": SupervisedConfig(
                    num_blocks=b, neurons_per_block=NEURONS_PER_BLOCK
                ),
                "rl": ReinforcementLearningConfig(
                    num_blocks=b, neurons_per_block=NEURONS_PER_BLOCK
                ),
                "aif": ActiveInferenceHyperparameters(
                    num_blocks=b, neurons_per_block=NEURONS_PER_BLOCK
                ),
            }

            all_params = {}
            for cfg_name, cfg_instance in configs.items():
                all_params.update(dataclasses.asdict(cfg_instance))

            all_params["num_initial_connections"] = len(
                configs["base"].initial_connectivity_map_rows
            )

            for key, value in all_params.items():
                if key in NON_PLOTTABLE_FIELDS:
                    continue
                data.setdefault(key, []).append(value)

        except (ValueError, TypeError):

            continue

    min_len = min(len(v) for v in data.values() if isinstance(v, list))
    for key, val in data.items():
        if isinstance(val, list):
            data[key] = val[:min_len]

    plot_groups = [
        {
            "title": "Learning Rates vs. N",
            "keys": [k for k in data.keys() if k.endswith("_lr")],
            "xscale": "log",
            "yscale": "log",
        },
        {
            "title": "Network Structure vs. N",
            "keys": ["max_synaptic_connections", "num_initial_connections"],
            "twin_keys": [],
        },
        {
            "title": "Weight & Growth Scales vs. N",
            "keys": [
                "initial_weight_scale",
                "initial_synaptic_efficacy",
                "initial_synaptic_polarity",
                "decay_lr",
            ],
            "yscale": "log",
        },
        {
            "title": "Timescales & Decays vs. N",
            "keys": [
                "eligibility_decay",
                "time_step_delta",
            ],
        },
        {
            "title": "Growth, Rewiring, & Activity vs. N",
            "keys": [
                "structural_plasticity",
                "trophic_map_ema_alpha",
                "homeostatic_setpoint",
            ],
            "yscale": "log",
        },
        {
            "title": "Static & Hyperparameters vs. N",
            "keys": [
                "target_connectivity",
                "gamma",
                "gae_lambda",
                "prediction_horizon",
                "max_norm",
                "noise",
            ],
        },
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 22), constrained_layout=True)
    axes = axes.flatten()
    plt.get_cmap("viridis")(np.linspace(0, 1, max(len(g["keys"]) for g in plot_groups)))

    fig.suptitle(
        f"Derived Configuration Parameters vs. Total Neurons (N)\n"
        f"(neurons_per_block = {NEURONS_PER_BLOCK})",
        fontsize=20,
        weight="bold",
    )

    x_values = data.get("total_neurons", [])

    for i, group in enumerate(plot_groups):
        ax = axes[i]
        ax.set_title(group["title"], fontsize=14, weight="bold")
        ax.set_xlabel("N (Total Neurons)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if group.get("xscale"):
            ax.set_xscale(group["xscale"])
        if group.get("yscale"):
            ax.set_yscale(group["yscale"])

        ax_twin = None

        param_keys = group["keys"]
        plot_colors = plt.get_cmap("nipy_spectral")(
            np.linspace(0.1, 0.9, len(param_keys))
        )

        for j, key in enumerate(param_keys):
            if key not in data or not data[key]:
                continue
            y_values = data[key]
            current_ax = ax
            label = key.replace("_", " ").replace(" lr", " LR")

            if key in group.get("twin_keys", []):
                if ax_twin is None:
                    ax_twin = ax.twinx()
                current_ax = ax_twin
                ax_twin.set_ylabel("Parameter Value (RHS)", color=plot_colors[j])
                ax_twin.tick_params(axis="y", labelcolor=plot_colors[j])
            else:
                ax.set_ylabel("Parameter Value")

            current_ax.plot(x_values, y_values, label=label, color=plot_colors[j])

        ax.legend(loc="best", fontsize=9)
        if ax_twin:

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax_twin.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)

    for i in range(len(plot_groups), len(axes)):
        axes[i].axis("off")

    output_filename = "config_parameter_analysis_full.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    analyze_and_plot_all_parameters()
