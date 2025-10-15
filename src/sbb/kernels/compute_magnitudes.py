"""
Triton kernel to compute L2 norms of weight blocks for pruning decisions.

This kernel calculates the Frobenius norm (L2 norm) of each specified weight
block. The result is used by StructuralPlasticity to identify weak connections
for pruning. Magnitude-based pruning is a simple yet effective heuristic that
removes connections with small weights, which typically contribute little to
the network's computation (Han et al., 2015).

Formula:
    magnitude = sqrt(Σ_{i,j} W_{i,j}²)

The computation uses float32 accumulation for numerical stability, then converts
to the output dtype.

Grid parallelization: (num_slots_to_check,)
Each thread block computes the norm of one weight block (NEURONS_PER_BLOCK² elements).
"""

import triton
import triton.language as tl


@triton.jit
def _compute_magnitudes_kernel(
    weight_values_ptr,
    slots_ptr,
    magnitudes_out_ptr,
    NUM_SLOTS_TO_CHECK,
    NEURONS_PER_BLOCK: tl.constexpr,
    stride_wv_slot,
    stride_wv_row,
    stride_wv_col,
):
    pid = tl.program_id(axis=0)
    if pid >= NUM_SLOTS_TO_CHECK:
        return

    slot_idx = tl.load(slots_ptr + pid)
    weight_block_ptr = weight_values_ptr + slot_idx * stride_wv_slot

    acc_f32 = tl.zeros((), dtype=tl.float32)
    offs_m = tl.arange(0, NEURONS_PER_BLOCK)
    offs_n = tl.arange(0, NEURONS_PER_BLOCK)

    tile_ptr = (
        weight_block_ptr
        + offs_m[:, None] * stride_wv_row
        + offs_n[None, :] * stride_wv_col
    )
    weights = tl.load(tile_ptr)

    acc_f32 += tl.sum(weights.to(tl.float32) * weights.to(tl.float32))
    magnitude_f32 = tl.sqrt(acc_f32)
    tl.store(
        magnitudes_out_ptr + pid, magnitude_f32.to(magnitudes_out_ptr.dtype.element_ty)
    )
