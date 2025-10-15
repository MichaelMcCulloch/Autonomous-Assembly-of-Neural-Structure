import torch
from torch.nn import Parameter, Module
from torch import Tensor
from dataclasses import dataclass

from sbb.kernels.compute_magnitudes import _compute_magnitudes_kernel

from .const import DEVICE, INDEX_DTYPE, EPS, MAX_QUANTILE_SIZE

GROWPRUNE_PER_STEP: int = 1


@dataclass
class StructuralChangeReport:
    grown: int
    pruned: int


class StructuralPlasticity(Module):
    """
    Discrete structural adaptation via unified biological resource competition.

    This module implements the discrete (topological) portion of adaptation,
    maintaining the network's sparse connectivity pattern through a biologically-inspired
    model where connections compete for survival based on a unified viability metric:

        viability = magnitude × (1 + trophic_support)

    where:
    - magnitude: synaptic strength (L2 norm for existing, estimated for potential connections)
    - trophic_support: local confusion/error signal (from SynapticPlasticity)

    Both pruning and growth use the same ecological dynamics:
    - A global survival threshold adapts to resource scarcity and trophic demand
    - Connections below threshold are pruned (weighted by inverse viability)
    - Potential connections above threshold can grow (weighted by viability)

    This creates biological resource competition without magic numbers:
    - High confusion (trophic demand) → raise survival bar → faster turnover
    - Resource scarcity (few free slots) → raise survival bar → selective growth/pruning
    - Trophic support protects weak connections in important regions
    - Low trophic support dooms even strong connections in unimportant regions

    Both operations respect degree constraints to prevent network fragmentation.
    Free connection slots are tracked in a reusable pool for efficient memory management.

    Parameters
    ----------
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Weight precision.
    weight_values : Parameter
        Block-sparse weight tensor (shared, modified during growth).
    weight_rows, weight_cols : Tensor
        Sparse connectivity coordinates (shared, modified during growth/pruning).
    active_blocks : Tensor
        Connection active mask (shared, modified during growth/pruning).
    in_degree, out_degree : Tensor
        Per-block connection counts (shared, modified during growth/pruning).
    trophic_support_map : Tensor
        Growth potential field [num_blocks, num_blocks] (shared, read here).
    initial_synaptic_efficacy : float
        Weight scale for new connections.
    initial_synaptic_polarity : float
        Bias term added to new connection initialization.
    target_connectivity : int
        Target connections per block (steady state).
    structural_plasticity : bool
        Allow changing connectivity.
    neurons_per_block : int
        Block dimension.
    num_blocks : int
        Number of blocks in network.

    Attributes
    ----------
    connections_to_change_per_step : int
        Number of connections to attempt growing/pruning each update.
    free_slots : Tensor
        Pool of available connection slots.
    num_free_slots : Tensor
        Current count of free slots.
    invalid_mask_buffer : Tensor
        Temporary buffer for growth candidate filtering.
    last_structural_change_report : StructuralChangeReport
        Most recent growth/pruning statistics.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        weight_values: Parameter,
        weight_rows: Tensor,
        weight_cols: Tensor,
        active_blocks: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        trophic_support_map: Tensor,
        initial_synaptic_efficacy: float,
        initial_synaptic_polarity: float,
        target_connectivity: int,
        structural_plasticity: bool,
        neurons_per_block: int,
        num_blocks: int,
    ):
        super().__init__()

        self.active_blocks = active_blocks
        self.initial_synaptic_efficacy = initial_synaptic_efficacy
        self.weight_cols = weight_cols
        self.initial_synaptic_polarity = initial_synaptic_polarity
        self.structural_plasticity = structural_plasticity
        self.target_connectivity = target_connectivity

        self.dtype = dtype
        self.trophic_support_map = trophic_support_map
        self.in_degree = in_degree
        self.neurons_per_block = neurons_per_block
        self.num_blocks = num_blocks
        self.out_degree = out_degree
        self.weight_rows = weight_rows
        self.weight_values = weight_values
        self.last_structural_change_report = StructuralChangeReport(grown=0, pruned=0)
        self.training = True
        self.max_slots = self.active_blocks.numel()
        inactive_slots = (~self.active_blocks).nonzero(as_tuple=True)[0]
        num_free = inactive_slots.numel()
        self.register_buffer(
            "free_slots",
            torch.zeros(self.max_slots, dtype=INDEX_DTYPE, device=DEVICE),
            persistent=False,
        )
        self.free_slots[:num_free] = inactive_slots  # type: ignore[operator]
        self.register_buffer(
            "num_free_slots",
            torch.tensor(num_free, dtype=torch.long, device=DEVICE),
            persistent=False,
        )

        self.register_buffer(
            "invalid_mask_buffer",
            torch.zeros(
                (self.num_blocks, self.num_blocks), dtype=torch.bool, device=DEVICE
            ),
            persistent=False,
        )

        # Cache unified viability threshold (recomputed each topology change)
        self._cached_survival_threshold = 0.0

    def backward(self) -> StructuralChangeReport:
        """
        Execute one step of structural plasticity: prune weak, grow strong.

        Called by PlasticityController (typically per-timestep). First attempts
        to prune weak connections, then attempts to grow new connections in
        high-trophic-support locations. The order matters: pruning frees slots
        for immediate reuse by growth.

        Uses a unified biological model where connections compete for trophic
        resources. Connection viability = (magnitude × trophic_support) - survival_threshold.
        Both pruning and growth use the same viability metric, creating a unified
        resource competition dynamic.

        Returns
        -------
        StructuralChangeReport
            Counts of connections grown and pruned this step.

        Notes
        -----
        In eval mode, returns zero changes without modifying topology.
        Growth is stochastic: even with eligible candidates, probability
        of success decreases with network density to maintain homeostasis.
        """
        if not self.training:
            self.last_structural_change_report = StructuralChangeReport(
                grown=0, pruned=0
            )
            return self.last_structural_change_report

        # Compute unified survival threshold once before pruning/growth
        self._cached_survival_threshold = self._compute_survival_threshold()

        pruned_count = self._prune()
        grown_count = self._grow()
        self.last_structural_change_report = StructuralChangeReport(
            grown=grown_count, pruned=pruned_count
        )
        return self.last_structural_change_report

    def _compute_survival_threshold(self) -> float:
        """
        Compute unified survival threshold based on ecological resource dynamics.

        The survival threshold represents the minimum viability (magnitude × trophic_support)
        a connection needs to survive or be created. This threshold rises with:
        1. Resource scarcity (few free slots) → need to be more selective
        2. Trophic demand (high confusion/error) → indicates need for restructuring

        Biological analog: When neurotrophic factors are scarce (resource pressure) or
        when activity-dependent demand is high (confusion), the baseline for connection
        survival increases.

        Returns
        -------
        float
            Global survival threshold for connection viability.
        """
        if not self.structural_plasticity:
            return 0.0

        # Measure 1: Resource scarcity (0 = abundant, 1 = scarce)
        resource_scarcity = 1.0 - (self.num_free_slots.item() / self.max_slots)  # type: ignore[operator]

        # Measure 2: Trophic demand (normalized confusion/error levels)
        trophic_values = self.trophic_support_map[self.trophic_support_map > 0]
        if trophic_values.numel() > 0:
            # Normalize by max to get relative demand
            avg_trophic = trophic_values.mean()
            max_trophic = trophic_values.max()
            normalized_trophic_demand = (avg_trophic / (max_trophic + EPS)).clamp(0, 1)
        else:
            normalized_trophic_demand = 0.0

        # Sample active connections to estimate viability distribution
        active_slots = self.active_blocks.nonzero(as_tuple=True)[0]
        num_active = active_slots.shape[0]
        if num_active == 0:
            return 0.0

        sample_size = min(num_active, GROWPRUNE_PER_STEP * 5)
        sample_indices = torch.randint(0, num_active, (sample_size,), device=DEVICE)
        slots_to_check = active_slots[sample_indices]

        # Compute connection magnitudes
        magnitudes = torch.empty(sample_size, dtype=self.dtype, device=DEVICE)
        grid = (sample_size,)

        _compute_magnitudes_kernel[grid](
            self.weight_values,
            slots_to_check,
            magnitudes,
            sample_size,
            self.neurons_per_block,
            self.weight_values.stride(0),
            self.weight_values.stride(1),
            self.weight_values.stride(2),
        )

        # Get trophic support for these connections
        rows_sampled = self.weight_rows[slots_to_check]
        cols_sampled = self.weight_cols[slots_to_check]
        trophic_support = self.trophic_support_map[rows_sampled, cols_sampled]

        # Compute viability = magnitude × (1 + trophic_support)
        # The (1 + trophic) ensures connections with zero trophic support still have some viability
        viability = magnitudes * (1.0 + trophic_support)

        # Threshold percentile adapts to ecological pressure
        # High scarcity OR high demand → lower percentile → higher survival bar
        # (lower percentile of viability distribution = higher absolute threshold)
        # Note: We use a lower base percentile (10%) to allow more growth initially
        survival_percentile = 10 + resource_scarcity * 25 + normalized_trophic_demand * 15  # 10% to 50%

        positive_viability = viability[viability > 0]
        if positive_viability.numel() > 0:
            threshold = torch.quantile(
                positive_viability.float(), survival_percentile / 100
            ).item()
            return threshold
        else:
            return 0.0

    def _prune(self) -> int:
        """
        Remove connections with viability below survival threshold.

        Uses unified viability metric: viability = magnitude × (1 + trophic_support).
        Connections compete for survival based on both their synaptic strength and
        their local trophic support (confusion/error signal). This creates a biological
        dynamic where weak connections in important (high-confusion) regions can survive,
        while strong connections in low-importance regions may be pruned.

        Constraints:
        - Never prune diagonal blocks (self-connections)
        - Never prune if it would reduce in_degree or out_degree to 0 (prevents fragmentation)
        - Pruned slots are added back to the free pool for reuse

        Trophic support for pruned connections is zeroed to prevent immediate regrowth.

        Returns
        -------
        int
            Number of connections actually pruned.

        Notes
        -----
        Probabilistic sampling trades exhaustive search for speed. Weaker connections
        (lower viability) are more likely to be selected for pruning.
        """
        if (not self.structural_plasticity) or self._cached_survival_threshold == 0.0:
            return 0

        active_slots = self.active_blocks.nonzero(as_tuple=True)[0]
        num_active = active_slots.shape[0]
        if num_active == 0:
            return 0

        # Sample subset to check
        sample_size = min(num_active, GROWPRUNE_PER_STEP * 5)
        sample_indices = torch.randint(0, num_active, (sample_size,), device=DEVICE)
        slots_to_check = active_slots[sample_indices]

        # Compute connection magnitudes
        magnitudes = torch.empty(sample_size, dtype=self.dtype, device=DEVICE)
        grid = (sample_size,)

        _compute_magnitudes_kernel[grid](
            self.weight_values,
            slots_to_check,
            magnitudes,
            sample_size,
            self.neurons_per_block,
            self.weight_values.stride(0),
            self.weight_values.stride(1),
            self.weight_values.stride(2),
        )

        rows_to_check = self.weight_rows[slots_to_check]
        cols_to_check = self.weight_cols[slots_to_check]

        # Get trophic support for these connections
        trophic_support = self.trophic_support_map[rows_to_check, cols_to_check]

        # Compute unified viability = magnitude × (1 + trophic_support)
        viability = magnitudes * (1.0 + trophic_support)

        # Apply pruning constraints
        is_diagonal = rows_to_check == cols_to_check
        in_degree_ok = self.in_degree[cols_to_check] > 1
        out_degree_ok = self.out_degree[rows_to_check] > 1
        degree_ok = in_degree_ok & out_degree_ok
        viability_ok = viability < self._cached_survival_threshold

        candidate_mask = ~is_diagonal & degree_ok & viability_ok
        candidate_indices_in_subset = candidate_mask.nonzero(as_tuple=True)[0]
        num_found = candidate_indices_in_subset.shape[0]
        if num_found == 0:
            return 0

        # Weighted sampling: lower viability → higher pruning probability
        candidate_slots = slots_to_check[candidate_indices_in_subset]
        candidate_viability = viability[candidate_indices_in_subset]
        candidate_weights = 1.0 / (candidate_viability + EPS)

        # Hybrid: randomly sample sqrt(num_found) candidates, then take topK from that
        num_to_sample = min(int(num_found**0.5), MAX_QUANTILE_SIZE)
        num_to_prune = min(num_to_sample, GROWPRUNE_PER_STEP)

        # First, randomly sample candidates
        sample_indices = torch.randint(0, num_found, (num_to_sample,), device=DEVICE)
        sampled_slots = candidate_slots[sample_indices]
        sampled_weights = candidate_weights[sample_indices]

        # Then, take topK weakest (lowest viability) from the random sample
        _, final_indices = torch.topk(sampled_weights, num_to_prune, largest=True)
        slots_to_prune = sampled_slots[final_indices]
        num_pruned = slots_to_prune.numel()
        if num_pruned == 0:
            return 0

        # Execute pruning: update mask, degrees, trophic map, free pool
        self.active_blocks[slots_to_prune] = False
        rows_pruned = self.weight_rows[slots_to_prune]
        cols_pruned = self.weight_cols[slots_to_prune]

        row_counts = torch.bincount(rows_pruned, minlength=self.num_blocks)
        self.out_degree.sub_(row_counts.to(self.out_degree.dtype)).clamp_min_(0)
        col_counts = torch.bincount(cols_pruned, minlength=self.num_blocks)
        self.in_degree.sub_(col_counts.to(self.in_degree.dtype)).clamp_min_(0)

        self.trophic_support_map[rows_pruned, cols_pruned] = 0.0

        current_num_free = self.num_free_slots.item()  # type: ignore[operator]
        self.free_slots[current_num_free : current_num_free + num_pruned] = (  # type: ignore[operator,misc]
            slots_to_prune
        )
        self.num_free_slots.add_(num_pruned)  # type: ignore[operator]

        return num_pruned

    def _grow(self) -> int:
        """
        Form new connections at locations with viability above survival threshold.

        Uses the same unified viability metric as pruning: viability = magnitude × (1 + trophic_support).
        For potential connections, "magnitude" is estimated from the trophic support scaled by initial
        synaptic efficacy. This creates biological competition: new connections must reach the same
        viability threshold that existing connections need to survive.

        Growth is weighted by viability: higher viability locations are more likely to be selected.
        This mirrors pruning's inverse weighting, creating a unified selection pressure.

        Returns
        -------
        int
            Number of connections actually grown.

        Notes
        -----
        Growth probability decreases with network density to maintain homeostasis.
        New connection weights are scaled by relative trophic strength to reflect
        their predicted importance.
        """
        num_available = int(self.num_free_slots.item())  # type: ignore[operator]
        if not self.structural_plasticity or num_available == 0:
            return 0

        # Build mask of invalid growth locations (existing connections + diagonal)
        invalid_mask = self.invalid_mask_buffer
        invalid_mask.fill_(False)  # type: ignore[operator]
        active_indices = self.active_blocks.nonzero(as_tuple=True)[0]
        if active_indices.numel() > 0:
            rows, cols = (
                self.weight_rows[active_indices],
                self.weight_cols[active_indices],
            )
            invalid_mask[rows, cols] = True  # type: ignore[operator]
        invalid_mask.fill_diagonal_(True)  # type: ignore[operator]

        inactive_mask = ~invalid_mask  # type: ignore[operator]
        trophic_values_inactive = self.trophic_support_map[inactive_mask]

        # No growth if no trophic support
        if trophic_values_inactive.numel() == 0 or trophic_values_inactive.max() <= 0:
            return 0

        # Estimate potential viability for candidate locations
        # Biological insight: We estimate a connection's "potential viability" based on its trophic support.
        # The trophic signal predicts how strong the connection *would become* if grown, not just its initial strength.
        # This is analogous to growth cones being attracted to neurotrophic gradients that predict future utility.
        candidate_trophic = self.trophic_support_map[inactive_mask]

        # Unified viability metric: Use trophic support directly as viability predictor
        # The trophic map already encodes magnitude × (eligibility × error), which predicts gradient flow
        # We scale by a factor that represents the expected mature magnitude of a useful connection
        max_trophic = candidate_trophic.max()

        if max_trophic > 0:
            # Normalize trophic to [0, 1] and use as relative viability
            normalized_trophic = (candidate_trophic / (max_trophic + EPS)).clamp(0, 1)
            # Estimate that viable connections would grow to match the survival threshold
            # This creates fair competition: high-trophic candidates compete with existing connections
            candidate_viability = self._cached_survival_threshold * normalized_trophic
        else:
            return 0

        # Filter by survival threshold (scaled by percentile to allow controlled growth)
        # Use a lower percentile threshold for growth to enable exploration
        growth_percentile = 0.3  # Accept top 70% of trophic signal locations
        growth_threshold = self._cached_survival_threshold * growth_percentile

        viability_above_threshold = candidate_viability > growth_threshold

        if viability_above_threshold.sum() == 0:
            return 0

        # Get candidate coordinates
        candidate_rows, candidate_cols = inactive_mask.nonzero(as_tuple=True)
        viable_mask = viability_above_threshold

        candidate_rows = candidate_rows[viable_mask]
        candidate_cols = candidate_cols[viable_mask]
        viable_viability = candidate_viability[viable_mask]
        viable_trophic = candidate_trophic[viable_mask]

        num_found = candidate_rows.shape[0]
        if num_found == 0:
            return 0

        # Homeostatic density control: growth probability decreases with density
        num_active = self.max_slots - num_available
        density = num_active / self.max_slots
        prob_success = 1.0 - density

        # Weighted sampling: higher viability → higher growth probability
        # (mirrors inverse weighting in pruning)
        MAX_MULTINOMIAL = 16777216
        if num_found > MAX_MULTINOMIAL:
            # Subsample if too large
            subsample_indices = torch.randint(
                0, num_found, (MAX_MULTINOMIAL,), device=DEVICE
            )
            candidate_rows = candidate_rows[subsample_indices]
            candidate_cols = candidate_cols[subsample_indices]
            viable_viability = viable_viability[subsample_indices]
            viable_trophic = viable_trophic[subsample_indices]
            num_found = MAX_MULTINOMIAL

        # Weighted sampling proportional to viability
        sampling_weights = viable_viability / (viable_viability.sum() + EPS)

        # Hybrid: sample sqrt(num_found) weighted by viability
        num_to_sample = min(int(num_found**0.5), MAX_QUANTILE_SIZE)

        if num_to_sample < num_found:
            # Multinomial sampling (weighted)
            sampled_indices = torch.multinomial(
                sampling_weights, num_to_sample, replacement=False
            )
        else:
            sampled_indices = torch.arange(num_found, device=DEVICE)

        # Apply density-based stochastic gating
        growth_attempts = torch.rand(sampled_indices.shape[0], device=DEVICE)
        success_mask = growth_attempts < prob_success
        num_successful = success_mask.sum().item()

        if num_successful == 0:
            return 0

        # Limit by available slots
        actual_num_to_grow = min(num_successful, num_available)
        success_indices = sampled_indices[success_mask][:actual_num_to_grow]

        # Allocate free slots
        slots_to_fill = self.free_slots[:actual_num_to_grow]  # type: ignore[index]

        # Compact free slots pool
        self.free_slots[: num_available - actual_num_to_grow] = self.free_slots[actual_num_to_grow:num_available].clone()  # type: ignore[index,operator]
        self.num_free_slots.sub_(actual_num_to_grow)  # type: ignore[operator]

        # Extract selected candidates
        rows_to_grow = candidate_rows[success_indices]
        cols_to_grow = candidate_cols[success_indices]
        viability_to_grow = viable_viability[success_indices]
        trophic_to_grow = viable_trophic[success_indices]

        # Scale new connections based on relative viability
        # Higher viability → stronger initial connection
        mean_viability = viable_viability.mean()
        magnitude_modifier = (
            (viability_to_grow / (mean_viability + EPS))
            .clamp_(min=0.5, max=2.0)
            .view(-1, 1, 1)
        )
        effective_scale = self.initial_synaptic_efficacy * magnitude_modifier

        # Initialize new weights
        new_weights = torch.randn(
            actual_num_to_grow,
            self.neurons_per_block,
            self.neurons_per_block,
            dtype=self.dtype,
            device=DEVICE,
        )
        new_weights.mul_(effective_scale)

        # Add polarity bias based on trophic sign
        sign_values = torch.sign(trophic_to_grow).view(-1, 1, 1)
        new_weights.add_(
            sign_values * self.initial_synaptic_polarity, alpha=effective_scale.mean()  # type: ignore[arg-type]
        )

        # Execute growth: update weights, connectivity, degrees
        self.weight_values.data[slots_to_fill] = new_weights
        self.weight_rows[slots_to_fill] = rows_to_grow
        self.weight_cols[slots_to_fill] = cols_to_grow
        self.active_blocks[slots_to_fill] = True

        row_counts = torch.bincount(rows_to_grow, minlength=self.num_blocks)
        self.out_degree.add_(row_counts.to(self.out_degree.dtype))
        col_counts = torch.bincount(cols_to_grow, minlength=self.num_blocks)
        self.in_degree.add_(col_counts.to(self.in_degree.dtype))

        # Zero out trophic support to prevent immediate regrowth
        self.trophic_support_map[rows_to_grow, cols_to_grow] = 0.0

        return actual_num_to_grow

    def forward(self):
        pass

    def train(self, mode: bool = True):
        return self
