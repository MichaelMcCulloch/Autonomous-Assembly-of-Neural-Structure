from torch.nn import Parameter
import torch

from src.sbb.const import DEVICE, INDEX_DTYPE
from src.sbb.weights import SynapticPlasticity


def setup_updater(
    num_blocks=4,
    neurons_per_block=16,
    batch_size=1,
    active_indices=None,
    plasticity_lrs=None,
    initial_vals_magnitude=0.5,
):
    torch.device("cuda")
    dtype = torch.float32
    num_blocks * neurons_per_block
    max_slots = num_blocks * num_blocks
    if plasticity_lrs is None:
        plasticity_lrs = {}
    weight_values = Parameter(
        torch.randn(
            (max_slots, neurons_per_block, neurons_per_block),
            device=DEVICE,
            dtype=dtype,
        )
        * initial_vals_magnitude,
        requires_grad=False,
    )
    weight_rows = (
        torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) // num_blocks
    )
    weight_cols = torch.arange(max_slots, device=DEVICE, dtype=INDEX_DTYPE) % num_blocks
    active_blocks = torch.zeros(max_slots, device=DEVICE, dtype=torch.bool)
    if active_indices:
        active_blocks[active_indices] = True
    in_degree, out_degree = torch.zeros(
        num_blocks, device=DEVICE, dtype=INDEX_DTYPE
    ), torch.zeros(num_blocks, device=DEVICE, dtype=INDEX_DTYPE)
    trophic_support_map = torch.zeros(
        num_blocks, num_blocks, device=DEVICE, dtype=dtype
    )
    total_neurons = neurons_per_block * num_blocks
    torch.eye(total_neurons, device=DEVICE, dtype=dtype)
    torch.eye(total_neurons, device=DEVICE, dtype=dtype)
    return SynapticPlasticity(
        dtype=dtype,
        batch_size=batch_size,
        neurons_per_block=neurons_per_block,
        num_blocks=num_blocks,
        weight_values=weight_values,
        weight_rows=weight_rows,
        weight_cols=weight_cols,
        active_blocks=active_blocks,
        in_degree=in_degree,
        out_degree=out_degree,
        trophic_support_map=trophic_support_map,
        trophic_map_ema_alpha=0.5,
        max_norm=10.0,
        delta_max_norm=1.0,
    )


def test_accumulate_growth_diagonal_zeroing():
    updater = setup_updater(num_blocks=2)
    T, B, N = (
        10,
        updater.batch_size,
        updater.num_blocks * updater.neurons_per_block,
    )
    traces = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype)
    feedback_signals = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype)
    inv_norms = torch.ones(T, B, 1, device=DEVICE, dtype=updater.dtype)
    post_gain = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype)

    updater.trophic_map_ema_alpha = 0.5
    assert torch.all(updater.trophic_support_map == 0.0)
    updater._trophics(traces, feedback_signals, inv_norms, post_gain)
    assert torch.all(updater.trophic_support_map.diag() == 0.0)
    assert torch.any(updater.trophic_support_map.triu(diagonal=1) > 0.0) or torch.any(
        updater.trophic_support_map.tril(diagonal=-1) > 0.0
    )


def test_oja_plasticity():
    torch.manual_seed(42)
    updater = setup_updater(plasticity_lrs={"oja": 0.1}, active_indices=[5])
    T, B, N = (
        10,
        updater.batch_size,
        updater.num_blocks * updater.neurons_per_block,
    )
    states = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype) * 0.1
    traces = torch.zeros(T, B, N, device=DEVICE, dtype=updater.dtype)
    inv_norms = torch.ones(T, B, 1, device=DEVICE, dtype=updater.dtype)
    feedback_signals = torch.zeros_like(states)

    vals_initial = updater.weight_values.clone()
    post_gain = torch.ones_like(states)
    updater.backward(
        states,
        traces,
        inv_norms,
        feedback_signals,
        pruning_threshold=0.1,
        post_gain=post_gain,
    )

    mask = torch.eye(updater.neurons_per_block, device=DEVICE).bool()
    assert torch.allclose(
        updater.weight_values[5][mask], torch.zeros_like(updater.weight_values[5][mask])
    )
    off = ~mask
    assert not torch.allclose(updater.weight_values[5][off], vals_initial[5][off])
    assert torch.allclose(updater.weight_values[0], vals_initial[0])


def test_hebbian_plasticity():
    torch.manual_seed(42)
    updater = setup_updater(plasticity_lrs={"hebbian": 0.1}, active_indices=[5])
    T, B, N = (
        10,
        updater.batch_size,
        updater.num_blocks * updater.neurons_per_block,
    )
    states = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype) * 0.1
    traces = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype) * 0.1
    inv_norms = torch.ones(T, B, 1, device=DEVICE, dtype=updater.dtype)
    feedback_signals = torch.zeros_like(states)

    vals_initial = updater.weight_values.clone()
    post_gain = torch.ones_like(states)
    updater.backward(
        states,
        traces,
        inv_norms,
        feedback_signals,
        pruning_threshold=0.1,
        post_gain=post_gain,
    )

    mask = torch.eye(updater.neurons_per_block, device=DEVICE).bool()
    assert torch.allclose(
        updater.weight_values[5][mask], torch.zeros_like(updater.weight_values[5][mask])
    )
    off = ~mask
    assert not torch.allclose(updater.weight_values[5][off], vals_initial[5][off])
    assert torch.allclose(updater.weight_values[0], vals_initial[0])


def test_adaptive_decay_plasticity():
    torch.manual_seed(42)
    updater = setup_updater(plasticity_lrs={"decay": 0.01}, active_indices=[5])
    T, B, N = (
        10,
        updater.batch_size,
        updater.num_blocks * updater.neurons_per_block,
    )
    states = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype) * 0.1
    traces = torch.zeros(T, B, N, device=DEVICE, dtype=updater.dtype)
    inv_norms = torch.ones(T, B, 1, device=DEVICE, dtype=updater.dtype)
    feedback_signals = torch.zeros_like(states)

    vals_initial = updater.weight_values.clone()
    post_gain = torch.ones_like(states)
    updater.backward(
        states,
        traces,
        inv_norms,
        feedback_signals,
        pruning_threshold=0.1,
        post_gain=post_gain,
    )

    mask = torch.eye(updater.neurons_per_block, device=DEVICE).bool()
    assert torch.allclose(
        updater.weight_values[5][mask], torch.zeros_like(updater.weight_values[5][mask])
    )
    off = ~mask
    assert not torch.allclose(updater.weight_values[5][off], vals_initial[5][off])
    assert torch.allclose(updater.weight_values[0], vals_initial[0])


def test_normalization_and_clipping():
    torch.manual_seed(42)
    updater = setup_updater(
        plasticity_lrs={"hebbian": 100.0},
        active_indices=[5],
        initial_vals_magnitude=0.1,
    )
    T, B, N = (
        10,
        updater.batch_size,
        updater.num_blocks * updater.neurons_per_block,
    )
    states = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype) * 10
    traces = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype) * 10
    inv_norms = torch.ones(T, B, 1, device=DEVICE, dtype=updater.dtype)
    feedback_signals = torch.zeros_like(states)

    updater.max_norm, updater.delta_max_norm = 2.0, 5.0
    post_gain = torch.ones_like(states)
    updater.backward(
        states,
        traces,
        inv_norms,
        feedback_signals,
        pruning_threshold=0.1,
        post_gain=post_gain,
    )

    final_norm = torch.linalg.norm(updater.weight_values[5])
    assert final_norm <= updater.max_norm + 1e-5


def test_no_active_blocks_no_weight_update():
    updater = setup_updater(
        plasticity_lrs={"hebbian": 1.0},
        active_indices=[],
    )
    T, B, N = (
        10,
        updater.batch_size,
        updater.num_blocks * updater.neurons_per_block,
    )
    states = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype)
    traces = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype)
    inv_norms = torch.ones(T, B, 1, device=DEVICE, dtype=updater.dtype)
    feedback_signals = torch.ones(T, B, N, device=DEVICE, dtype=updater.dtype)

    vals_i, gt_i = (
        updater.weight_values.clone(),
        updater.trophic_support_map.clone(),
    )
    post_gain = torch.ones_like(states)
    updater.backward(
        states,
        traces,
        inv_norms,
        feedback_signals,
        pruning_threshold=0.1,
        post_gain=post_gain,
    )

    assert torch.allclose(updater.weight_values, vals_i)
    assert not torch.allclose(updater.trophic_support_map, gt_i)
    assert torch.all(updater.trophic_support_map.diag() == 0.0)


def test_combined_step():
    lrs = {"oja": 0.01, "hebbian": 0.01, "decay": 0.001}
    updater = setup_updater(plasticity_lrs=lrs, active_indices=[5])
    T, B, N = (
        10,
        updater.batch_size,
        updater.num_blocks * updater.neurons_per_block,
    )
    states = torch.randn(T, B, N, device=DEVICE, dtype=updater.dtype) * 0.1
    traces = torch.randn(T, B, N, device=DEVICE, dtype=updater.dtype) * 0.1
    inv_norms = torch.ones(T, B, 1, device=DEVICE, dtype=updater.dtype)
    feedback_signals = torch.randn(T, B, N, device=DEVICE, dtype=updater.dtype)

    vals_i, gt_i = (
        updater.weight_values.clone(),
        updater.trophic_support_map.clone(),
    )
    post_gain = torch.ones_like(states)
    updater.backward(
        states,
        traces,
        inv_norms,
        feedback_signals,
        pruning_threshold=0.1,
        post_gain=post_gain,
    )

    assert not torch.allclose(updater.weight_values[5], vals_i[5])
    assert torch.all(updater.weight_values[5].diag() == 0.0)
    assert not torch.allclose(updater.trophic_support_map, gt_i)
    assert torch.all(updater.trophic_support_map.diag() == 0.0)
