import torch
from torch.nn import Parameter, Module
from torch import Tensor
from typing import Tuple

from sbb.kernels.advance_sequence import _advance_sequence_kernel
from sbb.kernels.gather_weights import _gather_weights_kernel

from .const import DEVICE, INDEX_DTYPE, EPS
from .types import SystemStateTuple


class BlockSparseRecurrentCore(Module):
    """
    Triton-accelerated block-sparse recurrent computation kernel.

    This module maintains CSR (Compressed Sparse Row) representations of the
    block-sparse connectivity and executes the recurrent state update via a
    fused Triton kernel. The kernel performs:

    1. Block-sparse matrix-vector multiplication (W_sparse @ state)
    2. Exponential Euler integration for state, eligibility, and activity traces
    3. Gaussian noise injection with deterministic RNG state
    4. Double-buffered state updates for sequence processing

    The CSR structure is lazily rebuilt when topology changes. Weight values
    are gathered into a sorted buffer for efficient kernel access. Scratch
    buffers are allocated on first use and reused across forward passes.

    Parameters
    ----------
    active_blocks : Tensor
        Boolean mask [max_connections] indicating which slots are active.
    dtype : torch.dtype
        Compute precision (typically bfloat16 or float32).
    neurons_per_block : int
        Block dimension (should be multiple of 16 for optimal performance).
    num_blocks : int
        Number of neural blocks in the network.
    weight_rows, weight_cols : Tensor
        Sparse connectivity coordinate lists.
    weight_values : Parameter
        Block weight tensor [max_connections, block_size, block_size].
    seed : int
        RNG seed for reproducible noise generation.

    Attributes
    ----------
    row_ptr : Tensor
        CSR row pointer array [num_blocks + 1].
    col_indices : Tensor
        CSR column index array [num_active_connections].
    sorted_values : Tensor
        Weight blocks in CSR order [num_active, block_size, block_size].
    scratch_a_*, scratch_b_* : Tensor
        Double-buffered state variables for ping-pong updates.
    """

    def __init__(
        self,
        active_blocks: Tensor,
        dtype: torch.dtype,
        neurons_per_block: int,
        num_blocks: int,
        weight_cols: Tensor,
        weight_rows: Tensor,
        weight_values: Parameter,
        seed: int,
    ):
        super().__init__()

        self.active_blocks, self.weight_cols, self.seed = (
            active_blocks,
            weight_cols,
            seed,
        )
        self.dtype, self.neurons_per_block, self.num_blocks = (
            dtype,
            neurons_per_block,
            num_blocks,
        )
        self.weight_rows, self.weight_values = weight_rows, weight_values
        self.training = True
        self._topology_changed = True
        self._values_stale = True
        # Initialize CSR arrays (will be properly set in _rebuild_topology)
        self.row_ptr = torch.zeros(num_blocks + 1, dtype=INDEX_DTYPE, device=DEVICE)
        self.col_indices = torch.zeros(0, dtype=INDEX_DTYPE, device=DEVICE)
        self.active_slots_sorted = torch.zeros(0, dtype=INDEX_DTYPE, device=DEVICE)

        self.register_buffer(
            "sorted_values",
            torch.empty(0, dtype=dtype, device=DEVICE),
            persistent=False,
        )
        self.register_buffer(
            "scratch_a_state",
            torch.empty(0, dtype=dtype, device=DEVICE),
            persistent=False,
        )
        self.register_buffer(
            "scratch_a_elig_trace",
            torch.empty(0, dtype=dtype, device=DEVICE),
            persistent=False,
        )
        self.register_buffer(
            "scratch_a_act_trace",
            torch.empty(0, dtype=dtype, device=DEVICE),
            persistent=False,
        )
        self.register_buffer(
            "scratch_b_state",
            torch.empty(0, dtype=dtype, device=DEVICE),
            persistent=False,
        )
        self.register_buffer(
            "scratch_b_elig_trace",
            torch.empty(0, dtype=dtype, device=DEVICE),
            persistent=False,
        )
        self.register_buffer(
            "scratch_b_act_trace",
            torch.empty(0, dtype=dtype, device=DEVICE),
            persistent=False,
        )

    def topology_stale(self):
        self._topology_changed = True
        self._values_stale = True

    def values_stale(self):
        self._values_stale = True

    def _update_topology(self):
        """
        Reconstruct CSR structure from active_blocks and weight_rows/cols.

        Converts the COO (coordinate) representation to CSR format by:
        1. Extracting active connection indices
        2. Sorting by row-major order (row * num_blocks + col)
        3. Building row_ptr array via cumulative sum of row counts
        4. Storing sorted column indices

        After topology rebuild, automatically updates sorted weight values.
        This method is called lazily when _topology_changed flag is set.
        """
        with torch.no_grad():
            active_slots = self.active_blocks.nonzero().squeeze(-1)
            active_rows, active_cols = (
                self.weight_rows[active_slots],
                self.weight_cols[active_slots],
            )
            sort_indices = torch.argsort(active_rows * self.num_blocks + active_cols)
            self.active_slots_sorted = active_slots[sort_indices]
            self.col_indices = active_cols[sort_indices].to(INDEX_DTYPE)
            sorted_rows = active_rows[sort_indices]
            row_ptr = torch.zeros(self.num_blocks + 1, dtype=INDEX_DTYPE, device=DEVICE)
            row_counts = torch.bincount(sorted_rows, minlength=self.num_blocks)
            torch.cumsum(row_counts, dim=0, out=row_ptr[1:])
            self.row_ptr = row_ptr
            self._topology_changed = False
            self._update_synapses()

    def _update_synapses(self):
        """
        Gather weight blocks into CSR-sorted order for kernel consumption.

        Uses a Triton kernel to efficiently copy weight blocks from the COO
        weight_values array into sorted_values in CSR order. The gather
        operation is performed via the active_slots_sorted mapping.

        This is called after topology changes or when weights are marked stale
        (e.g., after plasticity updates). The sorted buffer is resized if the
        number of active connections changes.
        """
        with torch.no_grad():
            self._values_stale = False
            if self.active_slots_sorted is None:
                if not self._topology_changed:
                    raise RuntimeError("Topology not built.")
                self._update_topology()
                return
            num_active = self.active_slots_sorted.shape[0]
            if self.sorted_values.shape[0] != num_active:
                self.sorted_values.resize_(
                    num_active, self.neurons_per_block, self.neurons_per_block
                )
            if num_active > 0:
                _gather_weights_kernel[(num_active,)](
                    self.weight_values,
                    self.sorted_values,
                    self.active_slots_sorted,
                    num_active,
                    self.neurons_per_block,
                    self.weight_values.stride(0),
                    self.weight_values.stride(1),
                    self.weight_values.stride(2),
                    self.sorted_values.stride(0),
                    self.sorted_values.stride(1),
                    self.sorted_values.stride(2),
                )
            self._values_stale = False

    def forward(
        self,
        initial_state_tuple: SystemStateTuple,
        external_field_sequence: Tensor,
        noise_std: float,
        dt: float,
        tau_fast: float,
        tau_activity: float,
        tau_eligibility: float,
        num_steps_per_input: int,
    ) -> Tuple[SystemStateTuple, Tensor]:
        """
        Execute fused Triton kernel for block-sparse recurrent state evolution.

        Processes an entire input sequence via double-buffered state updates.
        For each input timestep, performs num_steps_per_input internal substeps
        with exponential Euler integration:

            α_fast = exp(-dt / τ_fast)
            s ← α_fast * s + (1 - α_fast) * tanh(W @ s + u + b + noise)

        Eligibility and activity traces evolve with their respective time constants.
        Noise is injected via deterministic RNG (Philox algorithm) indexed by
        batch and timestep to ensure reproducibility across chunk boundaries.

        Parameters
        ----------
        initial_state_tuple : SystemStateTuple
            Starting state with activations, eligibility_trace, etc.
        external_field_sequence : Tensor [T, B, N]
            Projected inputs (already passed through weight_in and tanh).
        noise_std : float
            Standard deviation for Gaussian noise injection (0 disables).
        dt : float
            Integration timestep.
        tau_fast, tau_activity, tau_eligibility : float
            Time constants for state, activity trace, and eligibility trace.
        num_steps_per_input : int
            Internal substeps per external timestep.

        Returns
        -------
        final_state_tuple : SystemStateTuple
            State after processing full sequence (noise incremented).
        state_trajectory : Tensor [T, B, N]
            Hidden states at each external timestep (post-substep).

        Notes
        -----
        Scratch buffers (scratch_a_*, scratch_b_*) are lazily allocated on
        first call and reused thereafter. The kernel uses CSR format for
        efficient sparse matrix-vector products.
        """
        if self._topology_changed:
            self._update_topology()
        if self._values_stale:
            self._update_synapses()

        sequence_length, batch_size, total_neurons = external_field_sequence.shape

        if self.scratch_a_state.shape != (batch_size, total_neurons):  # type: ignore[has-type]
            self.scratch_a_state = torch.empty(
                (batch_size, total_neurons), dtype=self.dtype, device=DEVICE
            )
            self.scratch_a_elig_trace = torch.empty_like(self.scratch_a_state)
            self.scratch_a_act_trace = torch.empty_like(self.scratch_a_state)
            self.scratch_b_state = torch.empty_like(self.scratch_a_state)
            self.scratch_b_elig_trace = torch.empty_like(self.scratch_a_state)
            self.scratch_b_act_trace = torch.empty_like(self.scratch_a_state)

        self.scratch_a_state.copy_(initial_state_tuple.activations)
        self.scratch_a_elig_trace.copy_(initial_state_tuple.eligibility_trace)
        self.scratch_a_act_trace.copy_(initial_state_tuple.homeostatic_trace)

        # Prepare base-step (rng) per batch (required)
        base_step_per_batch = initial_state_tuple.noise.to(DEVICE, dtype=INDEX_DTYPE)

        # Run the kernel (external_field_sequence already projected and tanh'd in Machine)
        state_trajectory_output = torch.empty_like(external_field_sequence)
        final_state_tuple = SystemStateTuple(
            activations=torch.empty_like(initial_state_tuple.activations),
            eligibility_trace=torch.empty_like(initial_state_tuple.eligibility_trace),
            homeostatic_trace=torch.empty_like(initial_state_tuple.homeostatic_trace),
            bias=initial_state_tuple.bias,  # pass-through
            input_projection=initial_state_tuple.input_projection,  # pass-through
            noise=initial_state_tuple.noise,  # placeholder; overwritten below
        )

        BLOCK_SIZE_K = 32 if self.neurons_per_block >= 32 else 16

        grid = (batch_size, self.num_blocks)
        _advance_sequence_kernel[grid](
            state_a_ptr=self.scratch_a_state,
            eligibility_trace_a_ptr=self.scratch_a_elig_trace,
            activity_trace_a_ptr=self.scratch_a_act_trace,
            state_b_ptr=self.scratch_b_state,
            eligibility_trace_b_ptr=self.scratch_b_elig_trace,
            activity_trace_b_ptr=self.scratch_b_act_trace,
            activity_bias_ptr=initial_state_tuple.bias,
            external_field_sequence_ptr=external_field_sequence,
            final_state_ptr=final_state_tuple.activations,
            final_elig_trace_ptr=final_state_tuple.eligibility_trace,
            final_act_trace_ptr=final_state_tuple.homeostatic_trace,
            state_trajectory_ptr=state_trajectory_output,
            weight_val_ptr=self.sorted_values,
            row_ptr_ptr=self.row_ptr,
            col_idx_ptr=self.col_indices,
            base_step_ptr=base_step_per_batch,
            BATCH_SIZE=batch_size,
            N_CTX=total_neurons,
            SEQUENCE_LENGTH=sequence_length,
            NUM_STEPS_PER_INPUT=num_steps_per_input,
            SEED=self.seed,
            NEURONS_PER_BLOCK=self.neurons_per_block,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            NOISE_STD=noise_std,
            DT=dt,
            TAU_FAST=tau_fast,
            TAU_ACTIVITY=tau_activity,
            TAU_ELIGIBILITY=tau_eligibility,
            EPS=EPS,
        )

        # Advance noise for chunking-invariant noise across calls
        final_state_tuple.noise = (
            initial_state_tuple.noise + sequence_length * num_steps_per_input
        )

        return final_state_tuple, state_trajectory_output

    def train(self, mode: bool = True):
        return self
