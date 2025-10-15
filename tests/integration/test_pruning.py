import pytest
import torch
from torch.nn import Parameter

from src.sbb.topology import StructuralPlasticity
from src.sbb.const import DEVICE, INDEX_DTYPE


@pytest.fixture
def core_setup():
    """Provides a standard setup for the BlockSparseRecurrentCore."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    neurons_per_block = 4
    num_blocks = 4
    max_slots = num_blocks * num_blocks

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

    active_blocks = torch.zeros(max_slots, dtype=torch.bool, device=DEVICE)

    return {
        "device": device,
        "dtype": dtype,
        "weight_values": weight_values,
        "weight_rows": weight_rows,
        "weight_cols": weight_cols,
        "active_blocks": active_blocks,
        "neurons_per_block": neurons_per_block,
        "num_blocks": num_blocks,
        "seed": 42,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
def test_pruning_rule_preserves_positive_degrees_when_initially_positive(core_setup):
    """
    StructuralPlasticity only prunes edges (r->c) when out_degree[r] > 1 and in_degree[c] > 1
    (for off-diagonal edges). Therefore, it cannot reduce any node's off-diagonal in/out degree
    from 1 to 0. This test checks that invariant rather than requiring every node to have
    off-diagonal degree >=1 (some nodes may start at 0, which is allowed).
    """
    num_blocks = core_setup["num_blocks"]
    neurons_per_block = core_setup["neurons_per_block"]
    max_slots = num_blocks * num_blocks
    core_setup["device"]
    dtype = core_setup["dtype"]

    active_blocks = torch.zeros(max_slots, dtype=torch.bool, device=DEVICE)
    initial_connections = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 2),
        (0, 3),
    ]
    in_degree = torch.zeros(num_blocks, device=DEVICE, dtype=INDEX_DTYPE)
    out_degree = torch.zeros(num_blocks, device=DEVICE, dtype=INDEX_DTYPE)
    for r, c in initial_connections:
        idx = r * num_blocks + c
        active_blocks[idx] = True
        if r != c:
            in_degree[c] += 1
            out_degree[r] += 1

    core_setup["active_blocks"] = active_blocks

    core_setup["weight_values"].data.fill_(0.00001)

    initial_in_deg = in_degree.clone()
    initial_out_deg = out_degree.clone()

    trophic_support_map = torch.zeros(
        num_blocks, num_blocks, device=DEVICE, dtype=dtype
    )

    sp = StructuralPlasticity(
        dtype=dtype,
        weight_values=core_setup["weight_values"],
        weight_rows=core_setup["weight_rows"],
        weight_cols=core_setup["weight_cols"],
        active_blocks=core_setup["active_blocks"],
        in_degree=in_degree,
        out_degree=out_degree,
        trophic_support_map=trophic_support_map,
        initial_synaptic_efficacy=0.01,
        initial_synaptic_polarity=0.0005,
        target_connectivity=4,
        structural_plasticity=True,
        neurons_per_block=neurons_per_block,
        num_blocks=num_blocks,
    )

    _ = sp.backward()

    final_in_deg = sp.in_degree
    final_out_deg = sp.out_degree

    assert (final_in_deg >= 0).all()
    assert (final_out_deg >= 0).all()

    mask_in_pos = initial_in_deg > 0
    if mask_in_pos.any():
        assert (final_in_deg[mask_in_pos] > 0).all()

    mask_out_pos = initial_out_deg > 0
    if mask_out_pos.any():
        assert (final_out_deg[mask_out_pos] > 0).all()

    for i in range(num_blocks):
        diag_idx = i * num_blocks + i
        assert sp.active_blocks[diag_idx].item() is True
