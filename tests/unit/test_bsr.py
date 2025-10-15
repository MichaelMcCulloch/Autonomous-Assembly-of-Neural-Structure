from torch.nn import Parameter
import torch
import pytest

from src.sbb.bsr import BlockSparseRecurrentCore
from src.sbb.const import DEVICE, INDEX_DTYPE, EPS
from src.sbb.types import SystemStateTuple


@pytest.fixture
def core_setup():
    """Provides a standard setup for the BlockSparseRecurrentCore."""
    torch.device("cuda")
    dtype = torch.float32
    neurons_per_block = 16
    num_blocks = 8
    max_slots = num_blocks * num_blocks

    active_indices = []
    for i in range(num_blocks):
        active_indices.append(i * num_blocks + i)
        row = i
        col = (i + 1) % num_blocks
        active_indices.append(row * num_blocks + col)

    active_blocks = torch.zeros(max_slots, dtype=torch.bool, device=DEVICE)
    active_blocks[active_indices] = True

    weight_rows = (
        torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) // num_blocks
    )
    weight_cols = torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) % num_blocks

    weight_values = Parameter(
        torch.randn(
            max_slots,
            neurons_per_block,
            neurons_per_block,
            device=DEVICE,
            dtype=dtype,
        )
    )

    return {
        "dtype": dtype,
        "weight_values": weight_values,
        "weight_rows": weight_rows,
        "weight_cols": weight_cols,
        "active_blocks": active_blocks,
        "neurons_per_block": neurons_per_block,
        "num_blocks": num_blocks,
        "seed": 0,
    }


def test_forward_pass_correctness(core_setup):
    """
    Tests that the recurrent matrix multiplication within the Triton-based
    forward pass produces the correct output. This is done by isolating the
    matmul operation for a single internal step.
    """
    core = BlockSparseRecurrentCore(**core_setup)
    batch_size = 1
    total_neurons = core.num_blocks * core.neurons_per_block

    initial_state_tensor = torch.randn(
        batch_size, total_neurons, device=DEVICE, dtype=core.dtype
    )
    initial_state_tuple = SystemStateTuple(
        activations=initial_state_tensor.clone(),
        eligibility_trace=torch.zeros_like(initial_state_tensor),
        homeostatic_trace=torch.zeros_like(initial_state_tensor),
        bias=torch.zeros_like(initial_state_tensor),
        input_projection=torch.zeros_like(initial_state_tensor),
        noise=torch.zeros_like(initial_state_tensor),
    )

    expected_recurrent_term = torch.zeros_like(initial_state_tensor)
    core._update_topology()
    for i in range(core.num_blocks):
        start_idx = core.row_ptr[i]
        end_idx = core.row_ptr[i + 1]
        for sorted_idx in range(start_idx, end_idx):
            col_block_idx = core.col_indices[sorted_idx]
            weights = core.sorted_values[sorted_idx]
            row_block_idx = i
            r_start, r_end = (
                row_block_idx * core.neurons_per_block,
                (row_block_idx + 1) * core.neurons_per_block,
            )
            c_start, c_end = (
                col_block_idx * core.neurons_per_block,
                (col_block_idx + 1) * core.neurons_per_block,
            )
            input_segment = initial_state_tensor[:, c_start:c_end]
            output_segment = input_segment @ weights.T
            expected_recurrent_term[:, r_start:r_end] += output_segment

    tau_fast = 0.02
    dt = 0.002
    decay_fast = torch.exp(-torch.tensor(dt / (tau_fast + EPS)))
    expected_final_state = initial_state_tensor * decay_fast + torch.tanh(
        expected_recurrent_term
    ) * (1.0 - decay_fast)

    input_sequence = torch.zeros(
        1, batch_size, total_neurons, device=DEVICE, dtype=core.dtype
    )

    final_state_tuple, state_trajectory = core.forward(
        initial_state_tuple=initial_state_tuple,
        external_field_sequence=input_sequence,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=dt,
        tau_fast=tau_fast,
        tau_activity=1.0,
        num_steps_per_input=1,
    )
    actual_output = final_state_tuple.activations

    assert torch.allclose(actual_output, expected_final_state, atol=1e-5)


def test_rebuild_after_change(core_setup):
    """
    Tests that the forward pass reflects changes after a topology modification.
    """
    core = BlockSparseRecurrentCore(**core_setup)
    batch_size = 1
    total_neurons = core.num_blocks * core.neurons_per_block

    initial_state_tensor = torch.randn(
        batch_size, total_neurons, device=DEVICE, dtype=core.dtype
    )
    initial_state_tuple = SystemStateTuple(
        activations=initial_state_tensor.clone(),
        eligibility_trace=torch.zeros_like(initial_state_tensor),
        homeostatic_trace=torch.zeros_like(initial_state_tensor),
        bias=torch.zeros_like(initial_state_tensor),
        input_projection=torch.zeros_like(initial_state_tensor),
        noise=torch.zeros_like(initial_state_tensor),
    )
    input_sequence = torch.zeros(
        1, batch_size, total_neurons, device=DEVICE, dtype=core.dtype
    )

    final_state_before, _ = core.forward(
        initial_state_tuple=initial_state_tuple,
        external_field_sequence=input_sequence,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=0.01,
        tau_fast=0.1,
        tau_activity=1.0,
        num_steps_per_input=1,
    )

    inactive_indices = (~core.active_blocks).nonzero().squeeze()
    index_to_activate = inactive_indices[0]
    core.active_blocks[index_to_activate] = True
    core.topology_stale()

    final_state_after, _ = core.forward(
        initial_state_tuple=initial_state_tuple,
        external_field_sequence=input_sequence,
        tau_eligibility=0.2,
        noise_std=0.0,
        dt=0.01,
        tau_fast=0.1,
        tau_activity=1.0,
        num_steps_per_input=1,
    )

    assert not torch.allclose(
        final_state_before.activations, final_state_after.activations
    )
