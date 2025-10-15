import pytest
import torch

from src.sbb.bsr import BlockSparseRecurrentCore
from src.sbb.const import DEVICE, INDEX_DTYPE
from src.sbb.types import SystemStateTuple

cuda_available = torch.cuda.is_available()


def build_core(
    num_blocks,
    neurons_per_block,
    active_pattern="diag_plus_ring",
    seed=0,
    dtype=torch.float32,
):
    torch.device("cuda")
    max_slots = num_blocks * num_blocks

    active_blocks = torch.zeros(max_slots, dtype=torch.bool, device=DEVICE)
    indices = []
    if active_pattern in ("diag", "diag_plus_ring"):
        for i in range(num_blocks):
            indices.append(i * num_blocks + i)
    if active_pattern in ("ring", "diag_plus_ring"):
        for i in range(num_blocks):
            row = i
            col = (i + 1) % num_blocks
            indices.append(row * num_blocks + col)
    if active_pattern == "full_row0":
        row = 0
        indices.extend([row * num_blocks + j for j in range(num_blocks)])
    active_blocks[indices] = True

    weight_rows = (
        torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) // num_blocks
    )
    weight_cols = torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) % num_blocks

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    weight_values = torch.nn.Parameter(
        torch.randn(
            max_slots,
            neurons_per_block,
            neurons_per_block,
            device=DEVICE,
            dtype=dtype,
            generator=gen,
        )
        * 0.1
    )
    return BlockSparseRecurrentCore(
        active_blocks=active_blocks,
        dtype=dtype,
        neurons_per_block=neurons_per_block,
        num_blocks=num_blocks,
        weight_cols=weight_cols,
        weight_rows=weight_rows,
        weight_values=weight_values,
        seed=seed,
    )


@pytest.mark.skipif(not cuda_available, reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("batch_size", [1, 2, 7])
@pytest.mark.parametrize("sequence_length", [1, 3, 9])
@pytest.mark.parametrize("num_steps_per_input", [1, 2, 5])
@pytest.mark.parametrize("neurons_per_block", [16, 32, 64])
def test_kernel_shapes_and_stability(
    batch_size, sequence_length, num_steps_per_input, neurons_per_block
):
    num_blocks = 4
    core = build_core(num_blocks=num_blocks, neurons_per_block=neurons_per_block)
    N = num_blocks * neurons_per_block
    _device, dtype = DEVICE, core.dtype

    init_state = torch.zeros(batch_size, N, device=DEVICE, dtype=dtype)
    init_tuple = SystemStateTuple(
        activations=init_state.clone(),
        eligibility_trace=torch.zeros_like(init_state),
        homeostatic_trace=torch.zeros_like(init_state),
        bias=torch.zeros_like(init_state),
        input_projection=torch.zeros_like(init_state),
        noise=torch.zeros_like(init_state),
    )

    ext = torch.randn(sequence_length, batch_size, N, device=DEVICE, dtype=dtype) * 0.01

    final_state, traj = core.forward(
        initial_state_tuple=init_tuple,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=0.002,
        tau_fast=0.02,
        tau_activity=1.0,
        num_steps_per_input=num_steps_per_input,
    )

    assert traj.shape == (sequence_length, batch_size, N)
    assert final_state.activations.shape == (batch_size, N)
    assert final_state.eligibility_trace.shape == (batch_size, N)
    assert final_state.homeostatic_trace.shape == (batch_size, N)
    assert final_state.bias.shape == (batch_size, N)

    assert torch.isfinite(traj).all()
    assert torch.isfinite(final_state.activations).all()
    assert torch.isfinite(final_state.eligibility_trace).all()
    assert torch.isfinite(final_state.homeostatic_trace).all()

    assert traj.abs().max() < 5.0


@pytest.mark.skipif(not cuda_available, reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("num_steps_per_input", [1, 3, 7])
def test_kernel_inner_steps_effect(num_steps_per_input):

    num_blocks, neurons_per_block = 3, 64
    core = build_core(num_blocks=num_blocks, neurons_per_block=neurons_per_block)
    N = num_blocks * neurons_per_block
    _device, dtype = DEVICE, core.dtype

    batch_size, T = 2, 2
    init = torch.zeros(batch_size, N, device=DEVICE, dtype=dtype)
    st = SystemStateTuple(
        activations=init.clone(),
        eligibility_trace=torch.zeros_like(init),
        homeostatic_trace=torch.zeros_like(init),
        bias=torch.zeros_like(init),
        input_projection=torch.zeros_like(init),
        noise=torch.zeros_like(init),
    )

    ext = torch.ones(T, batch_size, N, device=DEVICE, dtype=dtype) * 0.05

    dt = 0.002
    tau_fast = 0.02
    final_state, traj = core.forward(
        initial_state_tuple=st,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=dt,
        tau_fast=tau_fast,
        tau_activity=1.0,
        num_steps_per_input=num_steps_per_input,
    )

    st2 = SystemStateTuple(
        activations=torch.zeros_like(init),
        eligibility_trace=torch.zeros_like(init),
        homeostatic_trace=torch.zeros_like(init),
        bias=torch.zeros_like(init),
        input_projection=torch.zeros_like(init),
        noise=torch.zeros_like(init),
    )
    final_state_base, _ = core.forward(
        initial_state_tuple=st2,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=dt,
        tau_fast=tau_fast,
        tau_activity=1.0,
        num_steps_per_input=1,
    )

    assert (
        final_state.activations.abs().mean() + 1e-7
        >= final_state_base.activations.abs().mean()
    )


@pytest.mark.skipif(not cuda_available, reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("neurons_per_block", [32, 64])
@pytest.mark.parametrize("sequence_length", [1, 4])
def test_kernel_respects_sorted_rows_cols(neurons_per_block, sequence_length):

    num_blocks = 32
    torch.device("cuda")
    dtype = torch.float32
    max_slots = num_blocks * num_blocks

    active_blocks = torch.zeros(max_slots, dtype=torch.bool, device=DEVICE)

    slot = 0 * num_blocks + 1
    active_blocks[slot] = True

    weight_rows = (
        torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) // num_blocks
    )
    weight_cols = torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) % num_blocks

    W = torch.zeros(
        max_slots, neurons_per_block, neurons_per_block, device=DEVICE, dtype=dtype
    )

    W[slot] += torch.eye(neurons_per_block, device=DEVICE, dtype=dtype) * 0.5
    weight_values = torch.nn.Parameter(W)

    core = BlockSparseRecurrentCore(
        active_blocks=active_blocks,
        dtype=dtype,
        neurons_per_block=neurons_per_block,
        num_blocks=num_blocks,
        weight_cols=weight_cols,
        weight_rows=weight_rows,
        weight_values=weight_values,
        seed=0,
    )

    N = num_blocks * neurons_per_block
    B = 2
    init_state = torch.zeros(B, N, device=DEVICE, dtype=dtype)

    col_block = 1
    start = col_block * neurons_per_block
    end = start + neurons_per_block
    init_state[:, start:end] = 1.0

    st = SystemStateTuple(
        activations=init_state.clone(),
        eligibility_trace=torch.zeros_like(init_state),
        homeostatic_trace=torch.zeros_like(init_state),
        bias=torch.zeros_like(init_state),
        input_projection=torch.zeros_like(init_state),
        noise=torch.zeros_like(init_state),
    )
    ext = torch.zeros(sequence_length, B, N, device=DEVICE, dtype=dtype)
    final_state, traj = core.forward(
        initial_state_tuple=st,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=0.002,
        tau_fast=0.02,
        tau_activity=1.0,
        num_steps_per_input=1,
    )

    row_block = 0
    rstart = row_block * neurons_per_block
    rend = rstart + neurons_per_block

    other_mask = torch.ones_like(final_state.activations, dtype=torch.bool)
    other_mask[:, rstart:rend] = False
    other_mask[:, start:end] = False

    assert final_state.activations[:, rstart:rend].abs().mean() > 1e-6

    assert final_state.activations[other_mask].abs().max() < 1e-5


@pytest.mark.skipif(not cuda_available, reason="CUDA required for Triton kernels")
def test_noise_std_effect_and_determinism_seed():
    num_blocks, neurons_per_block = 4, 16
    core = build_core(
        num_blocks=num_blocks, neurons_per_block=neurons_per_block, seed=123
    )
    N = num_blocks * neurons_per_block
    B, T = 3, 5
    _device, dtype = DEVICE, core.dtype

    init = torch.zeros(B, N, device=DEVICE, dtype=dtype)
    st = SystemStateTuple(
        activations=init.clone(),
        eligibility_trace=torch.zeros_like(init),
        homeostatic_trace=torch.zeros_like(init),
        bias=torch.zeros_like(init),
        input_projection=torch.zeros_like(init),
        noise=torch.zeros_like(init),
    )
    ext = torch.zeros(T, B, N, device=DEVICE, dtype=dtype)

    f0, _ = core.forward(
        initial_state_tuple=st,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=0.002,
        tau_fast=0.02,
        tau_activity=1.0,
        num_steps_per_input=2,
    )

    f1, _ = core.forward(
        initial_state_tuple=st,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.1,
        dt=0.002,
        tau_fast=0.02,
        tau_activity=1.0,
        num_steps_per_input=2,
    )
    f2, _ = core.forward(
        initial_state_tuple=st,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.1,
        dt=0.002,
        tau_fast=0.02,
        tau_activity=1.0,
        num_steps_per_input=2,
    )

    assert not torch.allclose(f1.activations, f0.activations)
    assert torch.allclose(f1.activations, f2.activations, atol=0, rtol=0)


@pytest.mark.skipif(not cuda_available, reason="CUDA required for Triton kernels")
def test_diagonal_block_zero_enforcement():

    num_blocks, neurons_per_block = 3, 32
    torch.device("cuda")
    dtype = torch.float32
    max_slots = num_blocks * num_blocks

    active_blocks = torch.zeros(max_slots, dtype=torch.bool, device=DEVICE)

    slot = 1 * num_blocks + 1
    active_blocks[slot] = True

    weight_rows = (
        torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) // num_blocks
    )
    weight_cols = torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) % num_blocks
    W = torch.zeros(
        max_slots, neurons_per_block, neurons_per_block, device=DEVICE, dtype=dtype
    )

    W[slot].fill_(0.0)
    for i in range(neurons_per_block):
        for j in range(neurons_per_block):
            if i != j:
                W[slot, i, j] = 0.2
            else:
                W[slot, i, j] = 0.0
    weight_values = torch.nn.Parameter(W)

    core = BlockSparseRecurrentCore(
        active_blocks=active_blocks,
        dtype=dtype,
        neurons_per_block=neurons_per_block,
        num_blocks=num_blocks,
        weight_cols=weight_cols,
        weight_rows=weight_rows,
        weight_values=weight_values,
        seed=0,
    )

    N = num_blocks * neurons_per_block
    B, T = 2, 3
    init = torch.randn(B, N, device=DEVICE, dtype=dtype) * 0.1
    st = SystemStateTuple(
        activations=init.clone(),
        eligibility_trace=torch.zeros_like(init),
        homeostatic_trace=torch.zeros_like(init),
        bias=torch.zeros_like(init),
        input_projection=torch.zeros_like(init),
        noise=torch.zeros_like(init),
    )
    ext = torch.zeros(T, B, N, device=DEVICE, dtype=dtype)
    f, traj = core.forward(
        initial_state_tuple=st,
        external_field_sequence=ext,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=0.002,
        tau_fast=0.02,
        tau_activity=1.0,
        num_steps_per_input=3,
    )

    start = 1 * neurons_per_block
    end = start + neurons_per_block
    assert traj[-1, :, start:end].abs().mean() > 1e-6

    assert torch.isfinite(traj).all()
