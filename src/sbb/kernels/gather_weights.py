"""
Triton kernel to gather block-sparse weights into CSR-sorted order.

This kernel reorganizes weight blocks from COO (coordinate) storage into
CSR-sorted order for efficient consumption by the advance_sequence kernel.
Each thread block copies one weight block (NEURONS_PER_BLOCK x NEURONS_PER_BLOCK).

The mapping is:
    sorted_values[csr_index] = weight_values[active_slots_sorted[csr_index]]

where active_slots_sorted provides the indirection from CSR order to COO slots.

This gather operation is needed because:
1. Weights are updated in-place in COO order (faster for plasticity)
2. Forward pass requires CSR order (faster for sparse matmul)
3. A full conversion would be wasteful; we only copy active blocks

Grid parallelization: (num_active_connections,)
Each thread block processes one weight block (NEURONS_PER_BLOCK² elements).
"""

import triton
import triton.language as tl


@triton.jit
def _gather_weights_kernel(
    weight_values_ptr,
    sorted_values_ptr,
    active_slots_sorted_ptr,
    NUM_ACTIVE_SLOTS,
    NEURONS_PER_BLOCK: tl.constexpr,
    stride_wv_slot,
    stride_wv_row,
    stride_wv_col,
    stride_sv_slot,
    stride_sv_row,
    stride_sv_col,
):
    pid = tl.program_id(axis=0)
    if pid >= NUM_ACTIVE_SLOTS:
        return
    source_slot_idx = tl.load(active_slots_sorted_ptr + pid)
    source_block_ptr = weight_values_ptr + source_slot_idx * stride_wv_slot
    dest_block_ptr = sorted_values_ptr + pid * stride_sv_slot
    offs_m = tl.arange(0, NEURONS_PER_BLOCK)
    offs_n = tl.arange(0, NEURONS_PER_BLOCK)
    source_ptr = (
        source_block_ptr
        + offs_m[:, None] * stride_wv_row
        + offs_n[None, :] * stride_wv_col
    )
    dest_ptr = (
        dest_block_ptr
        + offs_m[:, None] * stride_sv_row
        + offs_n[None, :] * stride_sv_col
    )
    source_data = tl.load(source_ptr)
    tl.store(dest_ptr, source_data)
