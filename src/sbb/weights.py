"""
Synaptic weight adaptation via composite potential function gradient descent.

This module implements continuous plasticity rules for the block-sparse recurrent
weights. All updates follow the principle: Δθ ∝ -∇_θ V, where V is a hand-crafted
potential function serving as a computationally tractable proxy for variational
free energy minimization.

The composite potential includes:
- Modulator term: Error-gated eligibility trace (three-factor rule)
- Hebbian term: Local correlation-based strengthening
- Oja term: Anti-Hebbian normalization for stability
- Decay term: L2 regularization scaled by inverse accumulated norm

A fused Triton kernel applies all rules simultaneously with per-block
norm clipping for numerical stability.
"""

from torch.nn import Parameter, Module
import torch
import triton.language as tl
from torch import Tensor

from sbb.kernels.update_weights import _update_weights_kernel

from .const import EPS


class SynapticPlasticity(Module):
    """
    Orchestrate continuous synaptic weight adaptation across all plasticity rules.

    This module implements the continuous portion of the learning algorithm,
    applying gradient-based updates to:
    1. Recurrent weights (via _update_weights_kernel with modulator, Hebbian, and Oja terms)
    2. Trophic support map (via EMA of eligibility-error correlation)

    Credit assignment uses exact backpropagation through the recurrent weight matrix (W^T)
    to route error signals. No learnable approximation matrices needed.

    All updates are designed for per-timestep application to ensure chunking
    invariance. Accumulated statistics (modulator_grad, hebbian_grad, etc.)
    are summed across batch/time without averaging, allowing arbitrary chunking.

    The trophic support map tracks correlation between pre-synaptic eligibility
    traces and post-synaptic error signals, providing a growth potential field
    for structural plasticity.

    Parameters
    ----------
    dtype : torch.dtype
        Weight precision.
    batch_size : int
        Expected batch size (for shape validation).
    neurons_per_block : int
        Block dimension.
    num_blocks : int
        Number of blocks in network.
    weight_values : Parameter
        Block-sparse weight tensor (shared with network core).
    weight_rows, weight_cols : Tensor
        Sparse connectivity coordinates (shared).
    active_blocks : Tensor
        Connection active mask (shared).
    in_degree, out_degree : Tensor
        Per-block connection counts (shared, unused here).
    trophic_support_map : Tensor
        Growth potential field (shared, updated here).
    trophic_map_ema_alpha : float
        EMA coefficient for trophic map updates.
    max_norm : float
        Maximum parameter norm.
    delta_max_norm : float
        Maximum update norm.
    """

    def __init__(
        self,
        dtype,
        batch_size,
        neurons_per_block,
        num_blocks,
        weight_values: Parameter,
        weight_rows: Tensor,
        weight_cols: Tensor,
        active_blocks: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        trophic_support_map: Tensor,
        trophic_map_ema_alpha,
        max_norm,
        delta_max_norm,
    ):
        super().__init__()
        (
            self.dtype,
            self.batch_size,
            self.neurons_per_block,
            self.num_blocks,
        ) = (dtype, batch_size, neurons_per_block, num_blocks)
        self.N = num_blocks * neurons_per_block
        (
            self.weight_values,
            self.weight_rows,
            self.weight_cols,
            self.active_blocks,
            self.in_degree,
            self.out_degree,
            self.trophic_support_map,
        ) = (
            weight_values,
            weight_rows,
            weight_cols,
            active_blocks,
            in_degree,
            out_degree,
            trophic_support_map,
        )
        self.trophic_map_ema_alpha = trophic_map_ema_alpha
        self.max_norm, self.delta_max_norm = (max_norm, delta_max_norm)
        self.triton_dtype = tl.bfloat16 if self.dtype == torch.bfloat16 else tl.float32

    def backward(
        self,
        system_states_trajectory: Tensor,
        eligibility_traces_trajectory: Tensor,
        inverse_state_norms_trajectory: Tensor,
        variational_gradient_trajectory: Tensor,
        pruning_threshold: float,
        post_gain: Tensor,
    ):
        """
        Apply all continuous plasticity updates for a batch of timesteps.

        This is the main entry point called by PlasticityController. It orchestrates:
        1. Trophic support map update (growth potential field)
        2. Synaptic weight update (Hebbian + Oja + Decay via Triton kernel)

        All statistics are accumulated additively across time/batch dimensions,
        ensuring chunking invariance: sum([chunk1, chunk2]) == sum(full_sequence).

        Parameters
        ----------
        system_states_trajectory : Tensor [T, B, N] or [1, B, N]
            Hidden states for Oja updates.
        eligibility_traces_trajectory : Tensor [T, B, N] or [1, B, N]
            Eligibility traces for Hebbian and trophic map updates.
        inverse_state_norms_trajectory : Tensor [T, B, 1] or [1, B, 1]
            Normalization factors: 1 / (||state||² + ε).
        variational_gradient_trajectory : Tensor [T, B, N] or [1, B, N]
            Error/reward signal for trophic map updates.
        pruning_threshold : float
            Dynamic pruning threshold for proximity-based decay.

        Notes
        -----
        Typically called with [1, B, N] inputs in the per-timestep loop from
        PlasticityController.step(). The additive design allows this.
        """
        # All inputs can be [T,B,N] or [1,B,N]; the math is additive across T.
        # Trophic map uses the same W^T backpropagation as weight updates
        # to correctly predict gradient magnitudes for structural plasticity.
        self._trophics(
            eligibility_traces_trajectory,
            variational_gradient_trajectory,
            inverse_state_norms_trajectory,
            post_gain,
        )
        self._synapses(
            system_states_trajectory,
            eligibility_traces_trajectory,
            inverse_state_norms_trajectory,
            variational_gradient_trajectory,
            pruning_threshold,
            post_gain,
        )

    def _trophics(
        self,
        eligibility_traces_trajectory: Tensor,
        variational_gradient_trajectory: Tensor,
        inverse_state_norms_trajectory: Tensor,
        post_gain: Tensor,
    ):
        with torch.no_grad():
            # Compute the same transformed modulator that drives weight updates
            # This ensures trophic map predicts actual gradient magnitudes
            raw_feedback = torch.sum(
                variational_gradient_trajectory * inverse_state_norms_trajectory,
                dim=(0, 1),
            )
            # Jacobian #3: Apply post-gain to trophic map (consistent with weight updates)
            post_gain_aggregated = post_gain.mean(dim=(0, 1))  # [N]
            raw_feedback = raw_feedback * post_gain_aggregated

            # mean vector fields over time/batch; for T=1 this is that step
            axonal_growth_field = eligibility_traces_trajectory.mean(dim=(0, 1)).view(
                self.num_blocks, self.neurons_per_block
            )

            # Use transformed modulator (same as weight updates), not raw variational gradient
            dendritic_seeking_field = raw_feedback.view(
                self.num_blocks, self.neurons_per_block
            )

            # Block-level trophic map: absolute value of outer product of block-averaged vectors
            # Note: This differs from element-wise sum (see paper correction)
            trophic_interaction = axonal_growth_field @ dendritic_seeking_field.T
            trophic_interaction = trophic_interaction.abs()
            trophic_interaction /= self.neurons_per_block + EPS

            self.trophic_support_map.mul_(1.0 - self.trophic_map_ema_alpha).add_(
                trophic_interaction, alpha=self.trophic_map_ema_alpha
            )
            self.trophic_support_map.fill_diagonal_(0)

    def _synapses(
        self,
        system_states_trajectory: Tensor,
        eligibility_traces_trajectory: Tensor,
        inverse_state_norms_trajectory: Tensor,
        variational_gradient_trajectory: Tensor,
        pruning_threshold: float,
        post_gain: Tensor,
    ):
        with torch.no_grad():
            active_slots = self.active_blocks.nonzero(as_tuple=False).squeeze(-1)
            num_active_slots = active_slots.shape[0]
            if num_active_slots == 0:
                return

            # Jacobian #2: diag_factorized eligibility with post_gain
            hebbian_grad = torch.sum(
                eligibility_traces_trajectory
                * post_gain
                * inverse_state_norms_trajectory,
                dim=(0, 1),
            )

            oja_grad = torch.sum(
                system_states_trajectory * inverse_state_norms_trajectory, dim=(0, 1)
            )
            # Aggregate error signal across time and batch
            raw_feedback = torch.sum(
                variational_gradient_trajectory * inverse_state_norms_trajectory,
                dim=(0, 1),
            )

            def grid(meta):
                return (num_active_slots,)

            _update_weights_kernel[grid](
                self.weight_values,
                raw_feedback,
                hebbian_grad,
                oja_grad,
                active_slots,
                self.weight_rows,
                self.weight_cols,
                NEURONS_PER_BLOCK=self.neurons_per_block,
                NUM_ACTIVE_SLOTS=num_active_slots,
                pruning_threshold=pruning_threshold,
                max_norm=self.max_norm,
                delta_max_norm=self.delta_max_norm,
                eps=EPS,
                stride_wv_slot=self.weight_values.stride(0),
                stride_wv_row=self.weight_values.stride(1),
                stride_wv_col=self.weight_values.stride(2),
                DTYPE=self.triton_dtype,
                num_warps=4,
            )
