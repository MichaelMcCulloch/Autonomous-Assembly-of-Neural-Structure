"""
Triton kernel for fused synaptic weight updates with composite plasticity rules.

This kernel implements a biologically-inspired plasticity algorithm combining
multiple learning rules in a single fused operation for efficiency. The composite
gradient includes gated Hebbian and Oja terms modulated by error signals, plus
adaptive decay for stability.
"""

import triton
import triton.language as tl
from .tanh import tanh


@triton.jit
def _update_weights_kernel(
    weight_values_ptr,
    modulator_grad_ptr,
    hebbian_grad_ptr,
    oja_grad_ptr,
    active_slots_ptr,
    weight_rows_ptr,
    weight_cols_ptr,
    NEURONS_PER_BLOCK: tl.constexpr,
    NUM_ACTIVE_SLOTS,
    pruning_threshold,
    max_norm,
    delta_max_norm,
    eps,
    stride_wv_slot,
    stride_wv_row,
    stride_wv_col,
    DTYPE: tl.constexpr,
):
    """
    Fused Triton kernel for block-sparse weight updates with local plasticity rules.

    Applies composite gradient descent: ΔW = composite_delta

    The composite delta includes:
    1. Gated plasticity: feedback * (Hebbian + Oja)
       - Modulatory gate (costate/error signal) controls learning magnitude
       - Hebbian: trace_row ⊗ trace_col, correlation strengthening
       - Oja: state_row ⊗ (state_col - state_row * W), anti-Hebbian normalization
    2. Decay: hybrid fixed + magnitude-adaptive L2 regularization (ungated)

    The hybrid decay term combines two mechanisms:
    - Fixed component (90%): decay_delta = -W * 0.9
    - Magnitude-adaptive (10%): decay_delta = -W * (||W|| / θ_prune) * 0.01
    - Total decay is normalized by NEURONS_PER_BLOCK⁴ for architectural scaling
    - Final form: decay_term = (-W * 0.9 - W * 0.01 * ||W|| / θ_prune) / NEURONS_PER_BLOCK⁴

    For 32x32 blocks with pruning_threshold=0.1:
    - Fixed decay coefficient: 0.9/1,048,576 ≈ 8.6e-7 (ultra-gentle baseline)
    - Magnitude decay scales with block strength, providing adaptive pressure
    - Stronger blocks experience proportionally more decay
    - Prevents long-term norm explosion while preserving plasticity headroom

    Hebbian and Oja use architecturally-scaled updates (normalized by block size).
    Per-block norm clipping is applied to both the update delta and final weights
    to prevent numerical instability. Diagonal blocks (self-connections) are zeroed.

    This kernel is launched with one thread block per active connection slot,
    processing NEURONS_PER_BLOCK x NEURONS_PER_BLOCK weight elements per block.
    """
    pid = tl.program_id(axis=0)
    if pid >= NUM_ACTIVE_SLOTS:
        return

    active_slot_idx = tl.load(active_slots_ptr + pid)
    s = active_slot_idx

    row_block_idx = tl.load(weight_rows_ptr + s)
    col_block_idx = tl.load(weight_cols_ptr + s)

    offs_m = tl.arange(0, NEURONS_PER_BLOCK)
    offs_n = tl.arange(0, NEURONS_PER_BLOCK)

    weight_tile_ptr = (
        weight_values_ptr
        + s * stride_wv_slot
        + offs_m[:, None] * stride_wv_row
        + offs_n[None, :] * stride_wv_col
    )

    current_weights = tl.load(weight_tile_ptr).to(tl.float32)

    # Load modulatory signal (costate/error) for post-synaptic block
    # Biologically: neuromodulator receptors are on the post-synaptic neuron
    # Three-factor rule: ΔW_ij ∝ pre_i × post_j × modulator_j
    mod_grad_col = tl.load(
        modulator_grad_ptr + (col_block_idx * NEURONS_PER_BLOCK + offs_n)
    )

    # Modulatory gate: post-synaptic signal only (broadcast to all pre-synaptic)
    # Each post-synaptic neuron gates all its incoming connections
    # Use tanh value
    # Higher error magnitude → more plasticity
    modulatory_gate = tanh(mod_grad_col)[None, :]

    hebb_trace_row = tl.load(
        hebbian_grad_ptr + (row_block_idx * NEURONS_PER_BLOCK + offs_m)
    )
    hebb_trace_col = tl.load(
        hebbian_grad_ptr + (col_block_idx * NEURONS_PER_BLOCK + offs_n)
    )

    oja_state_row = tl.load(oja_grad_ptr + (row_block_idx * NEURONS_PER_BLOCK + offs_m))
    oja_state_col = tl.load(oja_grad_ptr + (col_block_idx * NEURONS_PER_BLOCK + offs_n))

    hebb_delta = hebb_trace_row[:, None] * hebb_trace_col[None, :]
    oja_delta = oja_state_row[:, None] * (
        oja_state_col[None, :] - oja_state_row[:, None] * current_weights
    )

    # Normalize plasticity terms
    hebbian_term = hebb_delta / (NEURONS_PER_BLOCK * NEURONS_PER_BLOCK)
    oja_term = oja_delta / NEURONS_PER_BLOCK
    block_capacity_sq = (NEURONS_PER_BLOCK * NEURONS_PER_BLOCK) * (
        NEURONS_PER_BLOCK * NEURONS_PER_BLOCK
    )
    decay_term = -current_weights / block_capacity_sq

    # Gate Hebbian and Oja learning by the modulatory signal
    # Modulatory signal acts as a gain control on plasticity
    gated_plasticity = modulatory_gate * (hebbian_term + oja_term)
    final_delta = gated_plasticity + decay_term

    delta_norm = tl.sqrt(tl.sum(final_delta * final_delta))
    scale = tl.where(
        delta_norm > delta_max_norm, delta_max_norm / (delta_norm + eps), 1.0
    )
    final_delta *= scale

    new_weights = current_weights + final_delta

    if row_block_idx == col_block_idx:
        new_weights = tl.where(offs_m[:, None] == offs_n[None, :], 0.0, new_weights)

    weight_norm = tl.sqrt(tl.sum(new_weights * new_weights))
    scale = tl.where(weight_norm > max_norm, max_norm / (weight_norm + eps), 1.0)
    new_weights *= scale

    tl.store(weight_tile_ptr, new_weights.to(DTYPE))
