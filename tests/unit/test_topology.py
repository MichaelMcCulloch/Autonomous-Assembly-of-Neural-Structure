import torch
from torch.nn import Parameter
from src.sbb.topology import StructuralPlasticity
from src.sbb.const import DEVICE, INDEX_DTYPE


def setup_modifier(
    num_blocks,
    neurons_per_block,
    max_connections,
    structural_plasticity,
    target_connectivity,
    initial_connections,
    initial_potentials,
):
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    active_blocks = torch.zeros(max_connections, device=DEVICE, dtype=torch.bool)
    weight_rows = torch.full((max_connections,), -1, device=DEVICE, dtype=INDEX_DTYPE)
    weight_cols = torch.full((max_connections,), -1, device=DEVICE, dtype=INDEX_DTYPE)
    weight_values = torch.zeros(
        max_connections,
        neurons_per_block,
        neurons_per_block,
        device=DEVICE,
        dtype=dtype,
    )

    in_degree = torch.zeros(num_blocks, device=DEVICE, dtype=INDEX_DTYPE)
    out_degree = torch.zeros(num_blocks, device=DEVICE, dtype=INDEX_DTYPE)

    for i, (r, c, mag) in enumerate(initial_connections):
        active_blocks[i] = True
        weight_rows[i] = r
        weight_cols[i] = c
        weight_values[i, 0, 0] = mag
        if r != c:
            in_degree[c] += 1
            out_degree[r] += 1

    trophic_support_map = torch.zeros(
        num_blocks, num_blocks, device=DEVICE, dtype=dtype
    )
    for (r, c), pot in initial_potentials.items():
        trophic_support_map[r, c] = pot

    modifier = StructuralPlasticity(
        dtype=dtype,
        weight_values=Parameter(weight_values, requires_grad=False),
        weight_rows=weight_rows,
        weight_cols=weight_cols,
        active_blocks=active_blocks,
        in_degree=in_degree,
        out_degree=out_degree,
        trophic_support_map=trophic_support_map,
        initial_synaptic_efficacy=0.1,
        initial_synaptic_polarity=0.1,
        target_connectivity=target_connectivity,
        structural_plasticity=True,
        neurons_per_block=neurons_per_block,
        num_blocks=num_blocks,
    )
    return modifier


def test_no_change_when_no_candidates():
    """Tests that no rewiring occurs if no connections are weak enough to prune."""
    modifier = setup_modifier(
        num_blocks=4,
        neurons_per_block=2,
        max_connections=10,
        structural_plasticity=True,
        target_connectivity=2,
        initial_connections=[(0, 1, 2.0), (1, 2, 2.0)],
        initial_potentials={},
    )

    result = modifier.backward()

    assert result.pruned == 0
    assert result.grown == 0


def test_prune_selects_weakest():
    """Tests that one of the weakest candidates is chosen for pruning."""
    modifier = setup_modifier(
        num_blocks=4,
        neurons_per_block=2,
        max_connections=10,
        structural_plasticity=True,
        target_connectivity=2,
        initial_connections=[
            (0, 1, 0.4),
            (1, 2, 0.2),
            (0, 3, 2.0),
            (3, 0, 2.0),
            (1, 3, 2.0),
            (3, 1, 2.0),
            (2, 3, 2.0),
            (3, 2, 2.0),
        ],
        initial_potentials={},
    )

    initial_active_count = modifier.active_blocks.sum().item()
    result = modifier.backward()

    assert result.pruned == 1
    assert result.grown == 0

    assert modifier.active_blocks.sum().item() == initial_active_count - 1
