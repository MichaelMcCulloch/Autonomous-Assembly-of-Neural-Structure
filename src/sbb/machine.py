from torch.nn import Parameter, Module
from torch import Tensor
from typing import Tuple

from .bsr import BlockSparseRecurrentCore
from .const import DEVICE
from .types import SystemStateTuple


class Machine(Module):
    """
    Coordinates state evolution for the recurrent dynamical system.

    This module wraps BlockSparseRecurrentCore and handles the input projection.
    It implements the continuous-time state dynamics discretized via exponential
    Euler integration:

        τ_fast * ds/dt = -s + tanh(W_sparse @ s + gain ⊗ (u @ W_in) + bias)

    with Gaussian noise injection during training. Multiple substeps per input
    timestep ensure numerical stability for small τ_fast.

    The state tuple includes fast variables (activations), medium-timescale
    eligibility traces, and slow homeostatic activity traces, each with their
    own time constants.

    Parameters
    ----------
    cfg : BaseConfig
        Configuration with time constants, noise level, and dimensions.
    weight_in : Parameter
        Input projection weights [N, input_features].
    active_blocks : Tensor
        Boolean mask for active sparse connections.
    weight_rows, weight_cols : Tensor
        Sparse connectivity coordinate lists.
    weight_values : Parameter
        Block-sparse recurrent weight tensor.

    Attributes
    ----------
    bsr : BlockSparseRecurrentCore
        Triton-based kernel for efficient block-sparse matrix multiplication.
    """

    def __init__(
        self,
        cfg,
        weight_in: Parameter,
        active_blocks: Tensor,
        weight_rows: Tensor,
        weight_cols: Tensor,
        weight_values: Parameter,
    ):
        super().__init__()
        self.dtype, self.N = cfg.dtype, cfg.total_neurons
        self.num_blocks, self.neurons_per_block = cfg.num_blocks, cfg.neurons_per_block
        self.noise = cfg.noise
        self.dt, self.tau_fast = cfg.time_step_delta, cfg.tau_fast
        self.tau_activity, self.tau_eligibility, self.seed = (
            cfg.tau_activity,
            cfg.tau_eligibility,
            cfg.seed,
        )
        self.weight_in = weight_in
        self.bsr = BlockSparseRecurrentCore(
            active_blocks=active_blocks,
            dtype=self.dtype,
            neurons_per_block=self.neurons_per_block,
            num_blocks=self.num_blocks,
            weight_cols=weight_cols,
            weight_rows=weight_rows,
            weight_values=weight_values,
            seed=self.seed,
        )

    def forward(
        self,
        external_field_sequence: Tensor,
        initial_system_state: SystemStateTuple,
    ) -> Tuple[SystemStateTuple, Tensor]:
        """
        Advance the dynamical system through an input sequence.

        Projects external inputs through weight_in with tanh nonlinearity, then
        evolves all state variables (fast state, eligibility trace, activity trace)
        via the block-sparse recurrent kernel. Multiple internal substeps per input
        ensure stability for short time constants.

        Noise is only injected during training mode. RNG state advances deterministically
        to maintain reproducibility across different sequence chunk sizes.

        Parameters
        ----------
        external_field_sequence : Tensor [T, B, input_features]
            Input sequence to process.
        initial_system_state : SystemStateTuple
            Initial state (from previous sequence or zero initialization).

        Returns
        -------
        final_system_state : SystemStateTuple
            State after processing entire sequence.
        system_state_trajectory : Tensor [T, B, N]
            Hidden state at each input timestep (after all substeps).

        Notes
        -----
        The projected external field is stored in final_system_state for use
        during plasticity updates (needed for input gain adaptation).
        """
        external_field_sequence = external_field_sequence.to(DEVICE)
        input_projection = external_field_sequence @ self.weight_in.T

        noise_std_val = self.noise if self.training else 0.0
        substeps_per_evolution_step = max(1, int(round(self.tau_fast / self.dt)))

        (
            final_system_state,
            system_state_trajectory,
        ) = self.bsr.forward(
            initial_state_tuple=initial_system_state,
            external_field_sequence=input_projection,
            noise_std=noise_std_val,
            dt=self.dt,
            tau_fast=self.tau_fast,
            tau_activity=self.tau_activity,
            tau_eligibility=self.tau_eligibility,
            num_steps_per_input=substeps_per_evolution_step,
        )
        final_system_state.input_projection = input_projection[-1]
        return final_system_state, system_state_trajectory

    def train(self, mode: bool = True):
        return self
