import math
import torch
from torch.nn import Parameter, Module
from torch import Tensor
from typing import Tuple, Dict

from .machine import Machine
from .plasticity import PlasticityController
from .const import DEVICE, EPS, INDEX_DTYPE
from .util import _orthogonal, _zero_blocks
from .hyperparameters import BaseConfig
from .types import SystemStateTuple


class BaseModel(Module):
    """
    Primary integration module orchestrating system dynamics and adaptation.

    This class coordinates the recurrent state evolution (via Machine),
    continuous parameter adaptation (via PlasticityController), and structural
    topology changes. It manages the sparse weight matrices, buffers for state
    variables, and cache invalidation for the block-sparse computation backend.

    The design maintains separation of concerns: paradigms provide error signals,
    this module applies them through gradient-based weight updates and topology
    modifications. All state evolution is stateless -- the forward pass is a
    pure function of inputs and initial state.

    Attributes
    ----------
    cfg : BaseConfig
        Configuration object containing all derived hyperparameters.
    weight_in : Parameter
        Input projection matrix [N, input_features]. Orthogonally initialized.
    weight_values : Parameter
        Block-sparse recurrent weights [max_connections, block_size, block_size].
    active_blocks : Tensor (bool)
        Binary mask indicating which connection slots are active.
    weight_rows, weight_cols : Tensor (int)
        Row and column block indices for each connection slot.
    in_degree, out_degree : Tensor (int)
        Per-block connection counts, used to prevent network fragmentation.
    activity_bias : Parameter
        Homeostatic bias term [1, N] adapted to maintain target activation rate.
    machine : Machine
        Executes the recurrent state update via Triton kernels.
    plasticity : PlasticityController
        Orchestrates synaptic weight updates and structural topology changes.

    Notes
    -----
    Paradigms (supervised, RL, etc.) should call:
    - forward(input_sequence, initial_state) for state evolution
    - apply_plasticity(...) with error/reward signals for learning
    - snapshot_topology() for network structure visualization
    - new_state(batch_size) to initialize states

    Cache invalidation is automatic after plasticity updates. The module
    ensures Triton kernel CSR structures remain synchronized with topology.
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)

        self.cfg = cfg
        self.dtype = cfg.dtype
        self.N = cfg.total_neurons

        self._initialize_topology_buffers(
            cfg.initial_connectivity_map_rows, cfg.initial_connectivity_map_cols
        )
        self._initialize_parameter_tensors()

        self.machine = Machine(
            cfg=cfg,
            weight_in=self.weight_in,  # type: ignore[arg-type]
            active_blocks=self.active_blocks,  # type: ignore[arg-type]
            weight_rows=self.weight_rows,  # type: ignore[arg-type]
            weight_cols=self.weight_cols,  # type: ignore[arg-type]
            weight_values=self.weight_values,  # type: ignore[arg-type]
        )
        self.plasticity = PlasticityController(
            cfg=cfg,
            weight_values=self.weight_values,  # type: ignore[arg-type]
            active_blocks=self.active_blocks,  # type: ignore[arg-type]
            weight_rows=self.weight_rows,  # type: ignore[arg-type]
            weight_cols=self.weight_cols,  # type: ignore[arg-type]
            in_degree=self.in_degree,  # type: ignore[arg-type]
            out_degree=self.out_degree,  # type: ignore[arg-type]
            trophic_support_map=self.trophic_support_map,  # type: ignore[arg-type]
            activity_bias=self.activity_bias,  # type: ignore[arg-type]
        )

    def _initialize_topology_buffers(self, initial_rows: Tensor, initial_cols: Tensor):
        """
        Initialize sparse connectivity structure and degree tracking.

        Creates the core data structures for block-sparse connectivity:
        active_blocks (boolean mask), weight_rows/cols (coordinate lists),
        and in_degree/out_degree (per-block connection counts).

        Self-connections (diagonal blocks) are excluded from degree counts
        to allow proper detection of network fragmentation during pruning.

        Parameters
        ----------
        initial_rows : Tensor
            Block row indices for initial connections.
        initial_cols : Tensor
            Block column indices for initial connections.
        """
        active_blocks = torch.zeros(
            self.cfg.max_synaptic_connections, dtype=torch.bool, device=DEVICE
        )
        weight_rows = torch.full(
            (self.cfg.max_synaptic_connections,), -1, dtype=INDEX_DTYPE, device=DEVICE
        )
        weight_cols = torch.full(
            (self.cfg.max_synaptic_connections,), -1, dtype=INDEX_DTYPE, device=DEVICE
        )
        initial_connections = initial_rows.numel()
        active_blocks[:initial_connections] = True
        weight_rows[:initial_connections] = initial_rows
        weight_cols[:initial_connections] = initial_cols

        in_degree = torch.zeros(self.cfg.num_blocks, dtype=INDEX_DTYPE, device=DEVICE)
        out_degree = torch.zeros(self.cfg.num_blocks, dtype=INDEX_DTYPE, device=DEVICE)

        off_diagonal_mask = initial_rows != initial_cols
        rows_off_diag = initial_rows[off_diagonal_mask]
        cols_off_diag = initial_cols[off_diagonal_mask]

        out_degree.index_add_(
            0, rows_off_diag, torch.ones_like(rows_off_diag, dtype=INDEX_DTYPE)
        )

        in_degree.index_add_(
            0, cols_off_diag, torch.ones_like(cols_off_diag, dtype=INDEX_DTYPE)
        )

        self.register_parameter(
            "weight_values",
            Parameter(
                torch.zeros(
                    self.cfg.max_synaptic_connections,
                    self.cfg.neurons_per_block,
                    self.cfg.neurons_per_block,
                    dtype=self.dtype,
                    device=DEVICE,
                ),
                requires_grad=False,
            ),
        )
        self.register_buffer("active_blocks", active_blocks)
        self.register_buffer("in_degree", in_degree)
        self.register_buffer("out_degree", out_degree)
        self.register_buffer("weight_rows", weight_rows)
        self.register_buffer("weight_cols", weight_cols)

    def _initialize_parameter_tensors(self):
        """
        Initialize weight matrices, biases, and auxiliary learning buffers.

        Performs memory-efficient streaming initialization of recurrent weight
        blocks to avoid large temporary allocations. Diagonal blocks (self-
        connections) are zeroed out. Also initializes orthogonal input weights,
        homeostatic bias, eligibility trace factors (U, V),
        and the trophic support map for structural plasticity.

        Weight initialization follows Xavier/Glorot scaling based on effective
        fan-in (target_connectivity * neurons_per_block).
        """
        active_slots = self.active_blocks.nonzero().squeeze(-1)

        # Streamed initialization to avoid large temporary tensor
        num_active = active_slots.numel()
        BS = self.cfg.neurons_per_block
        dtype_size = (
            torch.finfo(self.dtype).bits // 8 if self.dtype.is_floating_point else 2
        )
        chunk_size = max(1, (128 * 1024 * 1024) // (BS * BS * dtype_size + 1))

        for start in range(0, num_active, chunk_size):
            end = min(start + chunk_size, num_active)
            chunk_indices = active_slots[start:end]

            block_values = (
                torch.randn(len(chunk_indices), BS, BS, dtype=self.dtype, device=DEVICE)
                * self.cfg.initial_weight_scale
            )

            chunk_rows = self.weight_rows[chunk_indices]
            chunk_cols = self.weight_cols[chunk_indices]
            diagonal_mask_in_chunk = (chunk_rows == chunk_cols).nonzero(as_tuple=True)[
                0
            ]

            if diagonal_mask_in_chunk.numel() > 0:
                block_values = _zero_blocks(
                    dtype=self.dtype,
                    block_values=block_values,
                    target_blocks=diagonal_mask_in_chunk,
                    neurons_per_block=BS,
                )

            self.weight_values.data[chunk_indices] = block_values

        self.register_parameter(
            "weight_in",
            Parameter(
                _orthogonal(
                    dtype=self.dtype,
                    rows=self.N,
                    columns=self.cfg.input_features,
                )
                * math.sqrt(self.N + EPS),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "activity_bias",
            Parameter(
                torch.zeros(1, self.N, dtype=self.dtype, device=DEVICE),
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "trophic_support_map",
            torch.zeros(
                self.cfg.num_blocks,
                self.cfg.num_blocks,
                dtype=self.dtype,
                device=DEVICE,
            ),
        )

    def new_state(self, batch_size: int) -> SystemStateTuple:
        """
        Create zero-initialized state for the beginning of a sequence.

        Returns a SystemStateTuple with activations=0, eligibility_trace=0,
        homeostatic_trace initialized to the homeostatic setpoint,
        bias cloned from the current learned bias, and noise=0.

        Parameters
        ----------
        batch_size : int
            Number of parallel sequences in the batch.

        Returns
        -------
        SystemStateTuple
            Initial state ready for forward pass.
        """
        state = torch.zeros(batch_size, self.N, dtype=self.dtype, device=DEVICE)
        eligibility_trace = torch.zeros_like(state)
        bias = self.activity_bias.expand(batch_size, -1).clone()  # type: ignore[operator]
        activity_trace = torch.full_like(state, self.cfg.homeostatic_setpoint)
        input_projection = torch.zeros_like(state)
        noise = torch.zeros(batch_size, dtype=INDEX_DTYPE, device=DEVICE)

        return SystemStateTuple(
            activations=state,
            eligibility_trace=eligibility_trace,
            homeostatic_trace=activity_trace,
            bias=bias,
            input_projection=input_projection,
            noise=noise,
        )

    def forward(
        self,
        input_sequence: Tensor,
        initial_state: SystemStateTuple,
    ) -> Tuple[SystemStateTuple, Tensor]:
        """
        Evolve system state across an input sequence.

        Projects inputs through weight_in, then advances the block-sparse
        recurrent dynamics for each timestep. The computation is delegated
        to Machine.evolve(), which calls the Triton kernel.

        Parameters
        ----------
        input_sequence : Tensor
            External inputs [T, B, input_features].
        initial_state : SystemStateTuple
            Starting state from previous sequence or new_state.

        Returns
        -------
        final_state : SystemStateTuple
            State after processing the full sequence.
        state_trajectory : Tensor
            Hidden states for each timestep [T, B, N].
        """
        return self.machine.forward(input_sequence, initial_state)

    @torch.no_grad()
    def backward(
        self,
        *,
        system_states: Tensor,  # [T, B, N]
        eligibility_traces: Tensor,  # [T, B, N]
        activity_traces: Tensor,  # [T, B, N]
        variational_signal: Tensor,  # [T, B, N]
        inverse_state_norms: Tensor,  # [T, B, 1]
    ):
        """
        Apply continuous and discrete plasticity based on error/reward signals.

        This is the main learning interface called by paradigms. Executes:
        1. Synaptic weight updates (modulator, Hebbian, Oja, decay)
        2. Eligibility trace adaptation (low-rank factorization U, V)
        3. Homeostatic bias adjustment (activity regulation)
        4. Structural topology changes (growth and pruning)
        5. Cache invalidation for Triton kernel CSR structures

        Parameters
        ----------
        system_states : Tensor [T, B, N]
            Hidden state trajectory from forward pass.
        eligibility_traces : Tensor [T, B, N]
            Eligibility trace trajectory (slow-decaying activity trace).
        activity_traces : Tensor [T, B, N]
            Long-term activity averages for homeostasis.
        variational_signal : Tensor [T, B, N]
            Error/reward gradient (e.g., prediction error, policy gradient).
        inverse_state_norms : Tensor [T, B, 1], optional
            Precomputed 1/(||state||^2 + eps). Computed if not provided.

        Notes
        -----
        All updates are applied per-timestep for chunking invariance. The
        variational_signal is gated by the post-synaptic Jacobian (1 - s^2)
        before being used in weight updates, implementing a principled
        approximation to backpropagation through time.

        The method automatically marks topology and weight caches dirty if
        any structural changes occur (growth or pruning).
        """

        report = self.plasticity.backward(
            system_states_trajectory=system_states,
            eligibility_traces_trajectory=eligibility_traces,
            activity_traces_trajectory=activity_traces,
            inverse_state_norms_trajectory=inverse_state_norms,
            variational_gradient_trajectory=variational_signal,
        )

        def _as_int(x):
            try:
                return int(x)
            except Exception:
                return 0

        grown = _as_int(getattr(report, "grown", 0)) if report is not None else 0
        pruned = _as_int(getattr(report, "pruned", 0)) if report is not None else 0

        if (grown + pruned) > 0:
            self.machine.bsr.topology_stale()
        self.machine.bsr.values_stale()

        return

    @torch.no_grad()
    def snapshot_topology(self) -> Dict[str, Tensor]:
        """
        Capture current network topology for visualization or analysis.

        Returns a dictionary with:
        - num_blocks: Total block capacity
        - max_slots: Maximum connection slots
        - num_active: Current active connections
        - rows_active: Active connection source blocks [K]
        - cols_active: Active connection target blocks [K]
        - in_degree: Incoming connections per block [B]
        - out_degree: Outgoing connections per block [B]
        - trophic_support_map: Growth potential matrix [B, B]

        Returns
        -------
        dict
            Snapshot dictionary with cloned tensors (safe to modify).
        """
        active_slots = self.active_blocks.nonzero(as_tuple=True)[0]  # type: ignore[operator]
        rows_active = self.weight_rows[active_slots].clone()  # type: ignore[index]
        cols_active = self.weight_cols[active_slots].clone()  # type: ignore[index]
        return {
            "num_blocks": torch.tensor(self.cfg.num_blocks, device=DEVICE),
            "max_slots": torch.tensor(self.active_blocks.numel(), device=DEVICE),  # type: ignore[operator]
            "num_active": torch.tensor(active_slots.numel(), device=DEVICE),
            "rows_active": rows_active,
            "cols_active": cols_active,
            "in_degree": self.in_degree.clone(),  # type: ignore[operator]
            "out_degree": self.out_degree.clone(),  # type: ignore[operator]
            "trophic_support_map": self.trophic_support_map.clone(),  # type: ignore[operator]
        }

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self
