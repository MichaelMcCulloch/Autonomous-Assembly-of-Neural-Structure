import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from sbb.const import DEVICE, EPS, INDEX_DTYPE


@dataclass
class BaseConfig:
    """
    Centralized configuration with automatic hyperparameter derivation.

    This dataclass manages all system parameters, both user-specified and derived.
    The __post_init__ method computes dozens of dependent parameters from a small
    set of base values using principled scaling laws. This design encodes critical
    knowledge about parameter relationships that would otherwise be lost.

    The derivation follows three core principles:
    1. Variance preservation: Weights scale as 1/sqrt(fan_in) to maintain signal magnitude.
    2. Biological scaling: Connectivity follows sublinear growth inspired by C. elegans data.
    3. Time hierarchy: Learning rates scale with 1/sqrt(N) to balance exploration/exploitation.

    Users should set:
    - num_blocks, neurons_per_block (network size)
    - tau_fast (fast timescale)
    - seed, device, dtype (execution)
    - input/output_features, batch_size (task interface)

    Everything else is derived automatically, but can be overridden by passing explicit values.

    Attributes
    ----------
    num_blocks : int
        Number of neural blocks (graph nodes). Default 16.
    neurons_per_block : int
        Neurons per block (should be multiple of 16 for performance). Default 16.
    tau_fast : float
        Fast state time constant in seconds. Governs relaxation speed. Default 0.02.
    seed : int
        Random seed for reproducibility.
    dtype : torch.dtype
        Compute precision (auto-detects bfloat16 support).
    input_features : int
        Dimensionality of external inputs.
    output_features : int
        Dimensionality of task outputs.
    batch_size : int
        Number of parallel sequences.
    evolution_substeps : int
        Internal integration steps per external timestep. Default 10.
    noise : float
        Gaussian noise std for exploration during training. Default 0.005.
    homeostatic_setpoint : float
        Target average activation rate. Default 0.05.
    tau_activity : float
        Slow homeostatic time constant in seconds. Default 10.0.
    tau_eligibility : float, optional
        Eligibility trace time constant. Derived as 10 * tau_fast if None.
    target_connectivity : int, optional
        Target connections per block. Derived from biological scaling if None.
    activity_lr : float, optional
        Homeostatic bias learning rate. Derived from tau_activity if None.
    initial_weight_scale : float, optional
        Weight initialization scale. Derived as 1/sqrt(fan_in).
    trophic_map_ema_alpha : float, optional
        EMA coefficient for growth potential tracking. Default 0.1.
    initial_synaptic_polarity : float, optional
        Bias term for new connection initialization.
    initial_synaptic_efficacy : float, optional
        Scale for new connections. Accounts for target connectivity.
    structural_plasticity : bool, optional
        Use structural plasticity.
    eligibility_decay : float, optional
        Per-step decay for eligibility factors. Derived from tau_eligibility.
    max_norm : float, optional
        Maximum L2 norm for parameters. Default 10.0.
    delta_max_norm : float, optional
        Maximum L2 norm for parameter updates. Default max_norm / 100.

    Derived Attributes (computed in __post_init__)
    -----------------------------------------------
    total_neurons : int
        num_blocks * neurons_per_block.
    time_step_delta : float
        Internal timestep: tau_fast / evolution_substeps.
    max_synaptic_connections : int
        Pre-allocated connection slots based on target connectivity and headroom.
    initial_connectivity_map_rows, initial_connectivity_map_cols : Tensor
        Initial sparse topology from Barabasi-Albert graph + diagonal connections.

    Notes
    -----
    The __post_init__ magic numbers (K_SATURATION, N_SATURATION_ONSET, etc.) encode
    empirical scaling laws. K_C_ELEGANS=23, N_C_ELEGANS=302 reference the C. elegans
    nervous system. These drive the sublinear connectivity growth: k ~ N^α where
    α ≈ 0.3-0.4, preventing quadratic cost while maintaining expressiveness.
    """

    num_blocks: int = 16
    neurons_per_block: int = 16
    tau_fast: float = 0.02

    seed: int = 42
    dtype: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    input_features: int = 1
    output_features: int = 1
    batch_size: int = 1
    evolution_substeps: int = 3
    noise: float = 0.000
    homeostatic_setpoint: float = 0.05
    tau_activity: Optional[float] = None
    tau_eligibility: Optional[float] = None
    target_connectivity: Optional[int] = None
    activity_lr: Optional[float] = None
    initial_weight_scale: Optional[float] = None
    trophic_map_ema_alpha: Optional[float] = None
    initial_synaptic_polarity: Optional[float] = None
    initial_synaptic_efficacy: Optional[float] = None
    structural_plasticity: Optional[bool] = None
    eligibility_decay: Optional[float] = None
    max_norm: Optional[float] = None
    delta_max_norm: Optional[float] = None
    total_neurons: int = field(init=False)
    time_step_delta: float = field(init=False)
    max_synaptic_connections: int = field(init=False)
    initial_connectivity_map_rows: torch.Tensor = field(init=False, repr=False)
    initial_connectivity_map_cols: torch.Tensor = field(init=False, repr=False)

    def _validate(self):
        if self.num_blocks < 2:
            raise ValueError("num_blocks must be at least 2.")
        if self.neurons_per_block < 1:
            raise ValueError("neurons_per_block must be at least 1.")
        if self.neurons_per_block > 0 and self.neurons_per_block % 16 != 0:
            print(
                f"Warning: neurons_per_block ({self.neurons_per_block}) is not a multiple of 16. This may lead to suboptimal performance with Triton kernels."
            )
        if self.activity_lr is not None and self.activity_lr == 0.0:
            print("Warning: activity_lr is 0. Activity-based homeostasis is disabled.")

    def __post_init__(self):
        K_SATURATION = 7500.0
        N_SATURATION_ONSET = 100000.0
        K_C_ELEGANS = 23.0
        N_C_ELEGANS = 302.0
        DENSITY_HEADROOM = 0.0

        self.total_neurons = self.num_blocks * self.neurons_per_block
        self.time_step_delta = self.tau_fast / self.evolution_substeps
        if self.tau_eligibility is None:
            self.tau_eligibility = self.tau_fast * 10

        if self.tau_activity is None:
            self.tau_activity = self.tau_eligibility * 5000

        if self.target_connectivity is None:
            if self.total_neurons <= 302.0:
                k_target = K_C_ELEGANS
            else:
                growth_exponent = math.log(K_SATURATION / K_C_ELEGANS) / math.log(
                    N_SATURATION_ONSET / N_C_ELEGANS
                )
                growth_factor = K_C_ELEGANS / (N_C_ELEGANS**growth_exponent)
                k_growth = growth_factor * (self.total_neurons**growth_exponent)
                k_target = min(K_SATURATION, k_growth)

            derived_degree = k_target / (self.neurons_per_block)
            k_blocks = int(math.ceil(derived_degree))
            self.target_connectivity = max(1, min(self.num_blocks - 1, k_blocks))

        if (
            self.neurons_per_block > 0
            and self.num_blocks > 1
            and self.target_connectivity > 0
        ):
            # Diagonal connections only
            rows_d = torch.arange(self.num_blocks, dtype=INDEX_DTYPE, device=DEVICE)
            cols_d = torch.arange(self.num_blocks, dtype=INDEX_DTYPE, device=DEVICE)

            num_initial_connections = self.num_blocks

            # Compute density from initial connections
            initial_density = (
                num_initial_connections / (self.num_blocks**2)
                if self.num_blocks > 0
                else 0
            )

            # Compute target density
            total_target_connections = (
                self.target_connectivity * self.num_blocks + self.num_blocks
            )
            target_density = total_target_connections / (self.num_blocks**2)

            # Max density with headroom
            required_density = max(target_density, initial_density)
            max_density = min(1.0, required_density * (1.0 + DENSITY_HEADROOM))
            self.max_synaptic_connections = int(max_density * (self.num_blocks**2))

            self.initial_connectivity_map_rows = rows_d
            self.initial_connectivity_map_cols = cols_d

        else:
            raise (ValueError("zero neurons = zero learning"))

        self.initial_connectivity_map_rows = self.initial_connectivity_map_rows.to(
            DEVICE
        )
        self.initial_connectivity_map_cols = self.initial_connectivity_map_cols.to(
            DEVICE
        )

        if self.initial_weight_scale is None:
            self.initial_weight_scale = 0.88 / math.sqrt(self.neurons_per_block - 1)

        if self.initial_synaptic_efficacy is None:
            self.initial_synaptic_efficacy = 0.88 / math.sqrt(
                self.neurons_per_block - 1
            )

        if self.structural_plasticity is None:
            self.structural_plasticity = True

        if self.initial_synaptic_polarity is None:
            self.initial_synaptic_polarity = self.initial_weight_scale

        if self.activity_lr is None:
            self.activity_lr = self.time_step_delta / (self.tau_activity)

        if self.eligibility_decay is None:
            self.eligibility_decay = 1.0 - math.exp(
                -self.time_step_delta / (self.tau_eligibility)
            )
        if self.trophic_map_ema_alpha is None:
            # Trophic growth should be GLACIAL to pick up environmental regularities, not noise
            self.trophic_map_ema_alpha = EPS
        if self.max_norm is None:
            self.max_norm = 10.0
        if self.delta_max_norm is None:
            self.delta_max_norm = self.max_norm / 100

        self._validate()
