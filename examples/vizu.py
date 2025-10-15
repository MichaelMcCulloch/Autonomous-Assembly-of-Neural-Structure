import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
from typing import Optional, Tuple

from sbb.const import EPS


class AANSVisualizer:
    """
    A non-invasive observer for visualizing the internal state of an AANS model.
    This module queries the BaseModel to render its dynamic topology,
    pruning candidates, growth potentials, and the distributions of the underlying
    plasticity metrics in lockstep with the simulation.
    """

    def __init__(self, simulation_integrator):
        """
        Initializes the visualizer for an AANS model.

        Args:
            simulation_integrator: The BaseModel instance to visualize.
        """
        if not hasattr(simulation_integrator, "plasticity"):
            raise AttributeError("The provided object must be a BaseModel instance.")

        self.simulation_integrator = simulation_integrator
        self.plasticity = self.simulation_integrator.plasticity
        self.num_blocks = self.simulation_integrator.cfg.num_blocks
        self.gradient_steps = 256

        plt.ion()
        self.fig = plt.figure(figsize=(13, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[20, 1, 1], wspace=0.3)

        self.ax_main = self.fig.add_subplot(gs[0, 0])
        self.ax_prune_bar = self.fig.add_subplot(gs[0, 1])
        self.ax_growth_bar = self.fig.add_subplot(gs[0, 2])

        ma_window = 100
        self.saliency_min_hist: deque[float] = deque(maxlen=ma_window)
        self.saliency_max_hist: deque[float] = deque(maxlen=ma_window)
        self.growth_min_hist: deque[float] = deque(maxlen=ma_window)
        self.growth_max_hist: deque[float] = deque(maxlen=ma_window)

        initial_rgb_matrix = np.zeros((self.num_blocks, self.num_blocks, 3))
        self.image = self.ax_main.imshow(initial_rgb_matrix, interpolation="nearest")
        legend_patches = [
            mpatches.Patch(
                color="#FF0000",
                label="Active Connection\nPruning Viability: mag×(1+trophic)"
            ),
            mpatches.Patch(
                color="#0000FF",
                label="Growth Candidate\nGrowth Viability: threshold×norm_trophic"
            ),
            mpatches.Patch(
                color="#FFFFFF",
                label="Self-connections (diagonal)"
            ),
        ]
        self.ax_main.legend(
            handles=legend_patches, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9
        )
        self.ax_main.set_xlabel("Destination Block Index", fontsize=12)
        self.ax_main.set_ylabel("Source Block Index", fontsize=12)

        saliency_cmap = LinearSegmentedColormap.from_list(
            "viability_map", ["#8B0000", "#FF0000", "#FFFFFF"]
        )
        growth_cmap = LinearSegmentedColormap.from_list(
            "growth_map", ["#000000", "#0000FF"]
        )

        static_gradient = np.linspace(0, 1, self.gradient_steps).reshape(-1, 1)

        self.im_prune = self.ax_prune_bar.imshow(
            static_gradient,
            cmap=saliency_cmap,
            aspect="auto",
            origin="lower",
        )
        self.im_prune.set_clim(0, 1)
        self.ax_prune_bar.set_title("Pruning\nViability", fontsize=10)
        self.ax_prune_bar.set_xticks([])
        self.ax_prune_bar.set_yticks([0, self.gradient_steps - 1])
        self.ax_prune_bar.yaxis.tick_right()

        self.im_growth = self.ax_growth_bar.imshow(
            static_gradient,
            cmap=growth_cmap,
            aspect="auto",
            origin="lower",
        )
        self.im_growth.set_clim(0, 1)
        self.ax_growth_bar.set_title("Growth\nViability", fontsize=10)
        self.ax_growth_bar.set_xticks([])
        self.ax_growth_bar.set_yticks([0, self.gradient_steps - 1])
        self.ax_growth_bar.yaxis.tick_right()

        self.fig.tight_layout(rect=(0.0, 0.03, 0.85, 0.95))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _get_visualization_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
        """
        Constructs matrices for visualization using unified viability metrics.

        Returns
        -------
        adj_matrix : np.ndarray
            Binary adjacency matrix (1 = active connection, 0 = inactive)
        pruning_viabilities : np.ndarray
            Raw pruning viability values for active connections
        growth_viabilities_matrix : np.ndarray
            Growth viability values for all locations (matrix form)
        survival_threshold : float
            Global survival threshold for pruning
        growth_threshold : float
            Growth threshold (30% of survival threshold)
        raw_trophic_map : np.ndarray
            Raw trophic support map for reference
        """
        with torch.no_grad():
            active_mask = self.simulation_integrator.active_blocks.cpu()
            rows = self.simulation_integrator.weight_rows.cpu()
            cols = self.simulation_integrator.weight_cols.cpu()
            weight_values = self.simulation_integrator.weight_values.cpu()

            structural_modifier = self.plasticity.structural
            raw_trophic_map = structural_modifier.trophic_support_map.cpu().numpy()

            # Get unified thresholds from the structural plasticity module
            survival_threshold = structural_modifier._cached_survival_threshold
            growth_threshold = survival_threshold * 0.3  # Growth uses 30% of survival threshold

        adj_matrix = np.zeros((self.num_blocks, self.num_blocks), dtype=np.float32)
        active_rows = rows[active_mask]
        active_cols = cols[active_mask]
        if active_rows.numel() > 0:
            adj_matrix[active_rows, active_cols] = 1.0

        # Compute pruning viability for active connections
        # Unified formula: viability = magnitude × (1 + trophic_support)
        pruning_viabilities = np.array([])
        pruning_viability_matrix = np.zeros_like(adj_matrix)
        active_indices = torch.where(active_mask)[0]

        if active_indices.numel() > 0:
            active_weights = weight_values[active_indices]
            magnitudes = torch.linalg.norm(
                active_weights.flatten(start_dim=1), dim=1
            ).numpy()

            trophic_at_connections = raw_trophic_map[active_rows, active_cols]
            # Unified viability: magnitude × (1 + trophic_support)
            viabilities = magnitudes * (1.0 + trophic_at_connections)
            pruning_viabilities = viabilities

            pruning_viability_matrix[active_rows, active_cols] = viabilities

        # Compute growth viability for inactive locations
        # Growth formula: survival_threshold × normalized_trophic
        growth_viabilities_matrix = np.zeros_like(adj_matrix)
        inactive_mask = (adj_matrix == 0)

        if inactive_mask.any():
            max_trophic = raw_trophic_map[inactive_mask].max() if raw_trophic_map[inactive_mask].max() > 0 else 1.0
            normalized_trophic = np.clip(raw_trophic_map / (max_trophic + EPS), 0, 1)
            growth_viabilities_matrix = survival_threshold * normalized_trophic
            # Zero out active connections (only show growth potential for inactive)
            growth_viabilities_matrix[~inactive_mask] = 0.0

        return (
            adj_matrix,
            pruning_viabilities,
            growth_viabilities_matrix,
            survival_threshold,
            growth_threshold,
            raw_trophic_map,
        )

    def update_adjacency_matrix(self, step_info: Optional[str] = None):
        """
        Fetches the latest data and updates all visualization components
        using unified viability thresholds.
        """
        (
            adj_matrix,
            pruning_viabilities,
            growth_viabilities_matrix,
            survival_threshold,
            growth_threshold,
            raw_trophic_map,
        ) = self._get_visualization_data()

        # Use actual thresholds for normalization instead of moving averages
        # This makes the visualization match exactly what pruning/growth see
        pruning_min, pruning_max = 0.0, survival_threshold * 2.0
        if pruning_viabilities.size > 0 and survival_threshold > 0:
            # Track moving average of actual range for smoother visualization
            self.saliency_min_hist.append(0.0)
            self.saliency_max_hist.append(pruning_viabilities.max())
            pruning_ma_max = np.mean(self.saliency_max_hist)  # type: ignore[assignment]
            pruning_max = max(survival_threshold * 2.0, pruning_ma_max)

        # Color bar labels show threshold
        self.ax_prune_bar.set_yticklabels(
            [f"0.0", f"{survival_threshold:.2e}"]
        )

        growth_min, growth_max = 0.0, survival_threshold
        if growth_threshold > 0:
            # Track moving average for smoother visualization
            growth_values_inactive = growth_viabilities_matrix[adj_matrix == 0]
            if growth_values_inactive.size > 0:
                self.growth_min_hist.append(0.0)
                self.growth_max_hist.append(growth_values_inactive.max())
                growth_ma_max = np.mean(self.growth_max_hist)  # type: ignore[assignment]
                growth_max = max(survival_threshold, growth_ma_max)

        self.ax_growth_bar.set_yticklabels(
            [f"0.0", f"{growth_threshold:.2e}"]
        )

        # Build RGB image with threshold-based normalization
        rgb_image = np.zeros((self.num_blocks, self.num_blocks, 3), dtype=np.float32)

        # Blue channel: Growth viability for inactive connections
        # Normalize by growth_max to show relative strength
        normalized_growth = np.clip(growth_viabilities_matrix / (growth_max + EPS), 0, 1)
        inactive_mask = 1.0 - adj_matrix
        rgb_image[:, :, 2] = normalized_growth * inactive_mask

        # Red channel: Pruning viability for active connections
        # Normalize by pruning_max, with threshold marking the survival line
        pruning_viability_matrix = np.zeros_like(adj_matrix)
        if pruning_viabilities.size > 0:
            active_indices = np.where(adj_matrix)
            active_rows = active_indices[0]
            active_cols = active_indices[1]
            # Map viabilities back to matrix
            viab_idx = 0
            for r, c in zip(active_rows, active_cols):
                if r != c:  # Skip diagonal
                    if viab_idx < len(pruning_viabilities):
                        pruning_viability_matrix[r, c] = pruning_viabilities[viab_idx]
                        viab_idx += 1

        normalized_pruning = np.clip(pruning_viability_matrix / (pruning_max + EPS), 0, 1)

        active_mask_bool = adj_matrix.astype(bool)
        # Active connections: Red with intensity based on viability
        rgb_image[active_mask_bool, 0] = 1.0  # Full red
        rgb_image[active_mask_bool, 1] = normalized_pruning[active_mask_bool]  # Green reduces viability
        rgb_image[active_mask_bool, 2] = normalized_pruning[active_mask_bool]  # Blue reduces viability
        # This creates: low viability = dark red, high viability = white/pink

        # Diagonal: white
        indices = np.arange(self.num_blocks)
        rgb_image[indices, indices, :] = 1.0

        self.image.set_data(np.clip(rgb_image, 0, 1))

        # Title with biological metrics
        num_connections = np.sum(adj_matrix) - np.sum(np.diag(adj_matrix))
        density = (
            num_connections / (self.num_blocks**2 - self.num_blocks) * 100
            if self.num_blocks > 1
            else 0
        )

        # Calculate resource metrics
        num_active = int(num_connections)
        max_slots = self.num_blocks * self.num_blocks - self.num_blocks
        num_free = max_slots - num_active
        resource_scarcity = 1.0 - (num_free / max_slots) if max_slots > 0 else 0.0

        title = "AANS Unified Viability Model\n"
        if step_info is not None:
            title += f"{step_info} | "
        title += f"Connections: {num_active} | Density: {density:.1f}% | "
        title += f"Survival Threshold: {survival_threshold:.2f} | Scarcity: {resource_scarcity:.2%}"
        self.ax_main.set_title(title, fontsize=13)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Closes the plot window gracefully."""
        plt.ioff()
        plt.close(self.fig)


class TraceVisualizer:
    """
    A non-invasive observer for visualizing the internal traces of an AANS model.
    This module uses matplotlib to render scrolling spectrograms of eligibility traces
    and a secondary signal (e.g., a modulatory or feedback signal).
    """

    def __init__(
        self,
        num_neurons: int,
        scroll_window_size: int = 1000,
        title: str = "AANS Live Trace Visualization",
        trace1_name: str = "Eligibility Trace",
        trace2_name: str = "Modulatory Signal",
    ):
        self.num_neurons = num_neurons
        self.scroll_window_size = scroll_window_size

        self.trace1_history: list[np.ndarray] = []
        self.trace2_history: list[np.ndarray] = []

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True
        )
        self.fig.suptitle(title, fontsize=16)

        self.ax1.set_xlim(0, self.scroll_window_size - 1)
        self.ax1.set_title(trace1_name)
        self.ax1.set_ylabel("Neuron Index")
        self.im1 = self.ax1.imshow(
            np.zeros((self.num_neurons, 1)),
            aspect="auto",
            cmap="viridis",
            extent=[0, 0, self.num_neurons, 0],
        )
        self.cbar1 = self.fig.colorbar(self.im1, ax=self.ax1, label="Trace Value")

        self.ax2.set_xlim(0, self.scroll_window_size - 1)
        self.ax2.set_title(trace2_name)
        self.ax2.set_xlabel("Global Time Step")
        self.ax2.set_ylabel("Neuron Index")
        self.im2 = self.ax2.imshow(
            np.zeros((self.num_neurons, 1)),
            aspect="auto",
            cmap="coolwarm",
            extent=[0, 0, self.num_neurons, 0],
        )
        self.cbar2 = self.fig.colorbar(self.im2, ax=self.ax2, label="Signal Value")

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, trace1_data: np.ndarray, trace2_data: np.ndarray):
        """
        Appends new data and updates the scrolling plots.
        Args:
            trace1_data: A 1D numpy array of shape (num_neurons,).
            trace2_data: A 1D numpy array of shape (num_neurons,).
        """
        self.trace1_history.append(trace1_data)
        self.trace2_history.append(trace2_data)

        num_steps = len(self.trace1_history)
        if num_steps <= self.scroll_window_size:
            trace1_display = np.array(self.trace1_history).T
            trace2_display = np.array(self.trace2_history).T
            start_idx, end_idx = 0, num_steps
        else:
            trace1_display = np.array(self.trace1_history[-self.scroll_window_size :]).T
            trace2_display = np.array(self.trace2_history[-self.scroll_window_size :]).T
            start_idx, end_idx = num_steps - self.scroll_window_size, num_steps
            self.ax1.set_xlim(start_idx, end_idx - 1)
            self.ax2.set_xlim(start_idx, end_idx - 1)

        self.im1.set_data(trace1_display)
        self.im1.set_extent([start_idx, end_idx - 1, self.num_neurons, 0])
        t1_min, t1_max = np.min(trace1_display), np.max(trace1_display)
        self.im1.set_clim(vmin=t1_min, vmax=t1_max if t1_max > t1_min else t1_min + EPS)

        self.im2.set_data(trace2_display)
        self.im2.set_extent([start_idx, end_idx - 1, self.num_neurons, 0])
        t2_min, t2_max = np.min(trace2_display), np.max(trace2_display)
        t2_abs_max = max(abs(t2_min), abs(t2_max))
        self.im2.set_clim(vmin=-t2_abs_max, vmax=t2_abs_max if t2_abs_max > 0 else EPS)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save_figure(self, file_path: str):
        """Saves the current state of the figure to a file."""
        try:
            self.fig.savefig(file_path, dpi=150, bbox_inches="tight")
        except Exception:
            pass

    def close(self):
        """Closes the plot window gracefully."""
        plt.ioff()
        plt.close(self.fig)
