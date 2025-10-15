"""
Fused Triton kernel for block-sparse recurrent state evolution.

This kernel integrates three coupled ODEs with different time constants:
    1. Fast state: τ_fast * ds/dt = -s + tanh(W@s + u + b + noise)
    2. Eligibility trace: τ_elig * de/dt = -e + s
    3. Activity trace: τ_act * da/dt = -a + |s|

The kernel uses:
- CSR format for efficient sparse matrix multiplication
- Double-buffering (ping-pong) for in-place state updates
- Deterministic RNG with chunking-invariant indexing
- Exponential Euler integration for numerical stability

Grid parallelization: (batch_size, num_blocks)
Each thread block processes one neural block (NEURONS_PER_BLOCK neurons) for one batch item.

Mathematical formulas (discrete exponential Euler):
    α = exp(-dt/τ)
    s[t+1] = α * s[t] + (1-α) * tanh(W@s[t] + u[t] + b + ε)
    e[t+1] = α_elig * e[t] + (1-α_fast) * s[t]
    a[t+1] = α_act * a[t] + (1-α_act) * |s[t+1]|

The eligibility trace uses (1-α_fast) injection to align with state dynamics,
implementing a diagonal-Jacobian approximation for credit assignment.
"""

import triton
import triton.language as tl
from .tanh import tanh


@triton.jit
def _advance_sequence_kernel(
    # Pointers to I/O tensors
    state_a_ptr,
    eligibility_trace_a_ptr,
    activity_trace_a_ptr,
    state_b_ptr,
    eligibility_trace_b_ptr,
    activity_trace_b_ptr,
    activity_bias_ptr,
    external_field_sequence_ptr,
    final_state_ptr,
    final_elig_trace_ptr,
    final_act_trace_ptr,
    state_trajectory_ptr,
    weight_val_ptr,
    row_ptr_ptr,
    col_idx_ptr,
    # Per-batch RNG base micro-step (for chunking-invariant noise)
    base_step_ptr,
    # Dimensions and parameters
    BATCH_SIZE,
    N_CTX,
    SEQUENCE_LENGTH,
    NUM_STEPS_PER_INPUT,
    SEED,
    # Constants
    NEURONS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NOISE_STD: tl.constexpr,
    DT: tl.constexpr,
    TAU_FAST: tl.constexpr,
    TAU_ACTIVITY: tl.constexpr,
    TAU_ELIGIBILITY: tl.constexpr,
    EPS: tl.constexpr,
):
    # One program per (batch_item, block_row)
    pid_batch = tl.program_id(0)
    pid_m_block = tl.program_id(1)

    # Neuron offsets for the current block
    offs_m = tl.arange(0, NEURONS_PER_BLOCK)
    block_row_offset = pid_batch * N_CTX + pid_m_block * NEURONS_PER_BLOCK + offs_m

    # Types and constants
    COMPUTE_DTYPE = state_a_ptr.dtype.element_ty
    zero = tl.full((), 0, COMPUTE_DTYPE)
    one = tl.full((), 1, COMPUTE_DTYPE)

    # Load initial state for this block
    state = tl.load(state_a_ptr + block_row_offset)  # s_0
    elig_trace = tl.load(eligibility_trace_a_ptr + block_row_offset)  # e_0
    act_trace = tl.load(activity_trace_a_ptr + block_row_offset)
    act_bias = tl.load(
        activity_bias_ptr + block_row_offset
    )  # bias is constant over sequence

    # Leak/decays
    decay_fast = tl.exp(-DT / (TAU_FAST + EPS)).to(COMPUTE_DTYPE)  # α (state)
    decay_elig = tl.exp(-DT / (TAU_ELIGIBILITY + EPS)).to(
        COMPUTE_DTYPE
    )  # α_e (eligibility)
    decay_activity = tl.exp(-DT / (TAU_ACTIVITY + EPS)).to(COMPUTE_DTYPE)

    # CSR row pointers for this block row
    row_start_ptr = row_ptr_ptr + pid_m_block
    row_end_ptr = row_ptr_ptr + pid_m_block + 1
    start_col_idx = tl.load(row_start_ptr)
    end_col_idx = tl.load(row_end_ptr)

    # Per-batch RNG base micro-step
    base_micro = tl.load(base_step_ptr + pid_batch)

    # External-step loop
    for t in range(SEQUENCE_LENGTH):
        # Ping-pong buffers
        even = (t & 1) == 0
        read_state_ptr = state_a_ptr if even else state_b_ptr
        write_state_ptr = state_b_ptr if even else state_a_ptr
        write_elig_ptr = eligibility_trace_b_ptr if even else eligibility_trace_a_ptr
        write_act_ptr = activity_trace_b_ptr if even else activity_trace_a_ptr

        # Block-sparse recurrent drive: rec = Σ_k W_{block}(row,k) * s_k
        rec = tl.zeros((NEURONS_PER_BLOCK,), dtype=COMPUTE_DTYPE)
        for col_idx_offset in range(start_col_idx, end_col_idx):
            k_block_idx = tl.load(col_idx_ptr + col_idx_offset)
            state_k_ptr = (
                read_state_ptr + pid_batch * N_CTX + k_block_idx * NEURONS_PER_BLOCK
            )
            weight_block_ptr = (
                weight_val_ptr + col_idx_offset * NEURONS_PER_BLOCK * NEURONS_PER_BLOCK
            )

            for k_start in range(0, NEURONS_PER_BLOCK, BLOCK_SIZE_K):
                offs_k_tile = k_start + tl.arange(0, BLOCK_SIZE_K)
                mask_k_tile = offs_k_tile < NEURONS_PER_BLOCK
                state_k_tile = tl.load(
                    state_k_ptr + offs_k_tile, mask=mask_k_tile, other=zero
                )
                weight_slab_ptr = (
                    weight_block_ptr
                    + offs_m[:, None] * NEURONS_PER_BLOCK
                    + offs_k_tile[None, :]
                )
                weight_slab = tl.load(
                    weight_slab_ptr, mask=mask_k_tile[None, :], other=zero
                )
                rec += tl.sum(weight_slab * state_k_tile[None, :], axis=1)

        # External input (already projected outside the kernel) + bias
        ext_input_ptr = (
            external_field_sequence_ptr + t * BATCH_SIZE * N_CTX + block_row_offset
        )
        ext_input = tl.load(ext_input_ptr)

        base_potential = rec + ext_input + act_bias

        # Micro-integration over NUM_STEPS_PER_INPUT
        for step in range(NUM_STEPS_PER_INPUT):
            # Save pre-update state s_t for eligibility update
            state_prev = state

            # Chunking-invariant noise: offset includes per-batch base micro-step
            noise_offset = (
                (base_micro + t * NUM_STEPS_PER_INPUT + step) * BATCH_SIZE * N_CTX
                + pid_batch * N_CTX
                + pid_m_block * NEURONS_PER_BLOCK
                + offs_m
            )
            noise = tl.randn(SEED, noise_offset).to(tl.float32)
            noise = (noise * NOISE_STD).to(COMPUTE_DTYPE)

            pot = tanh(base_potential + noise)  # tanh(a_{t+1})
            one_minus_alpha = one - decay_fast

            # State update: s_{t+1} = α s_t + (1−α) tanh(a_{t+1})
            state = state_prev * decay_fast + pot * one_minus_alpha

            # Eligibility (diagonal-J): retention uses α_e, injection uses (1−α_fast)
            #   e ← α_e e + (1−α_fast) · s_t
            elig_trace = elig_trace * decay_elig + state_prev * one_minus_alpha

            # Activity trace: EMA of |s|
            act_trace = act_trace * decay_activity + tl.abs(state) * (
                one - decay_activity
            )

        # Write updated block row to ping-pong buffers for next external step
        tl.store(write_state_ptr + block_row_offset, state)
        tl.store(write_elig_ptr + block_row_offset, elig_trace)
        tl.store(write_act_ptr + block_row_offset, act_trace)

        # Save state trajectory at external step resolution
        state_trajectory_block_row_ptr = (
            state_trajectory_ptr + t * BATCH_SIZE * N_CTX + block_row_offset
        )
        tl.store(state_trajectory_block_row_ptr, state)

    # Final outputs
    tl.store(final_state_ptr + block_row_offset, state)
    tl.store(final_elig_trace_ptr + block_row_offset, elig_trace)
    tl.store(final_act_trace_ptr + block_row_offset, act_trace)
