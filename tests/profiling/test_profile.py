import pytest
import torch
import copy
from torch.profiler import profile, record_function, ProfilerActivity

from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import SupervisedConfig
from src.sbb.paradigms.predictive_coding import PredictiveCoding


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_accumulation = True
torch.backends.cudnn.allow_tf32 = True


def format_hertz(sps):
    """Formats a number into Hz, kHz, MHz, or GHz."""
    if sps >= 1_000_000_000:
        return f"{sps / 1_000_000_000:.2f} GHz"
    if sps >= 1_000_000:
        return f"{sps / 1_000_000:.2f} MHz"
    if sps >= 1_000:
        return f"{sps / 1_000:.2f} kHz"
    return f"{sps:.2f} Hz"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("neurons_per_block", [32])
@pytest.mark.parametrize("num_blocks", [1024, 10240])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("evolution_substeps", [16])
def test_torch_profiler_learn(
    neurons_per_block,
    num_blocks,
    batch_size,
    evolution_substeps,
):
    """
    Runs a detailed PyTorch profiler analysis, isolating the `backward` method
    to correctly identify its internal bottlenecks.
    """
    cfg = SupervisedConfig(
        num_blocks=num_blocks,
        neurons_per_block=neurons_per_block,
        batch_size=batch_size,
        input_features=2,
        output_features=1,
        seed=42,
        evolution_substeps=evolution_substeps,
    )
    model = PredictiveCoding(cfg=cfg)

    total_steps = 50
    warmup_steps = 10

    preds = [
        torch.randn(cfg.batch_size, cfg.output_features, device=DEVICE, dtype=cfg.dtype)
        for _ in range(total_steps)
    ]
    targets = [
        torch.randn(cfg.batch_size, cfg.output_features, device=DEVICE, dtype=cfg.dtype)
        for _ in range(total_steps)
    ]

    state_tuple = model.base.new_state(cfg.batch_size)
    next_state_tuple = copy.deepcopy(state_tuple)
    next_state_tuple.activations.normal_()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:

        for i in range(warmup_steps):
            model.backward(preds[i], targets[i], state_tuple, next_state_tuple)

        for i in range(warmup_steps, total_steps):
            with record_function("learn_step_call"):
                model.backward(preds[i], targets[i], state_tuple, next_state_tuple)

    print("\n--- PyTorch Profiler Results (Learn Step, Top 15 CUDA time) ---")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=15
        )
    )

    events = prof.key_averages()
    update_kernel_events = [e for e in events if "_update_weights_kernel" in e.key]

    assert (
        len(update_kernel_events) > 0
    ), "_update_weights_kernel not found in profiler output. The learn step may not be executing as expected."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("neurons_per_block", [32])
@pytest.mark.parametrize("num_blocks", [1024, 10240])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("evolution_substeps", [16])
def test_torch_profiler_forward(
    neurons_per_block,
    num_blocks,
    batch_size,
    evolution_substeps,
):
    """
    Runs a detailed PyTorch profiler analysis, isolating the `forward` method
    to correctly identify its internal bottlenecks.
    """
    cfg = SupervisedConfig(
        num_blocks=num_blocks,
        neurons_per_block=neurons_per_block,
        batch_size=batch_size,
        input_features=2,
        output_features=1,
        seed=42,
        evolution_substeps=evolution_substeps,
    )
    model = PredictiveCoding(cfg=cfg)

    total_steps = 50
    warmup_steps = 10

    inputs = [
        torch.randn(cfg.batch_size, cfg.input_features, device=DEVICE, dtype=cfg.dtype)
        for _ in range(total_steps)
    ]
    state_tuple = model.base.new_state(cfg.batch_size)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:

        for i in range(warmup_steps):
            _, state_tuple = model.forward(inputs[i], state_tuple)

        for i in range(warmup_steps, total_steps):
            with record_function("forward_step_call"):
                _, state_tuple = model.forward(inputs[i], state_tuple)

    print("\n--- PyTorch Profiler Results (Forward Step, Top 15 CUDA time) ---")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=15
        )
    )

    events = prof.key_averages()
    forward_kernel_events = [e for e in events if "_advance_sequence_kernel" in e.key]

    assert (
        len(forward_kernel_events) > 0
    ), "_advance_sequence_kernel not found in profiler output. The forward step may not be executing as expected."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("neurons_per_block", [32])
@pytest.mark.parametrize("num_blocks", [1024, 10240])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("evolution_substeps", [16])
def test_torch_profiler_full_loop(
    neurons_per_block,
    num_blocks,
    batch_size,
    evolution_substeps,
):
    """
    Profiles a single complete forward + learn iteration including all overhead:
    state management, tensor creation, density calculations, and evolution substeps.
    This shows where the time goes in test_performance.py's full loop.
    """
    cfg = SupervisedConfig(
        num_blocks=num_blocks,
        neurons_per_block=neurons_per_block,
        batch_size=batch_size,
        input_features=2,
        output_features=1,
        seed=42,
        evolution_substeps=evolution_substeps,
    )
    model = PredictiveCoding(cfg=cfg)

    # Warmup
    warmup_steps = 10
    state_tuple = model.base.new_state(cfg.batch_size)

    for _ in range(warmup_steps):
        u = torch.randn(
            cfg.batch_size, cfg.input_features, device=DEVICE, dtype=cfg.dtype
        )
        y_target = torch.randn(
            cfg.batch_size, cfg.output_features, device=DEVICE, dtype=cfg.dtype
        )
        pred, next_state_tuple = model.forward(u, state_tuple)
        _, state_tuple = model.backward(pred, y_target, state_tuple, next_state_tuple)

    # Profile a single iteration
    u = torch.randn(cfg.batch_size, cfg.input_features, device=DEVICE, dtype=cfg.dtype)
    y_target = torch.randn(
        cfg.batch_size, cfg.output_features, device=DEVICE, dtype=cfg.dtype
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with record_function("full_iteration"):
            pred, next_state_tuple = model.forward(u, state_tuple)
            _, state_tuple = model.backward(
                pred, y_target, state_tuple, next_state_tuple
            )
            num_active_blocks = model.base.active_blocks.sum()
            num_active_blocks / (cfg.num_blocks**2)

    print("\n--- PyTorch Profiler Results (Full Loop, Top 20 CUDA time) ---")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=20
        )
    )

    # Get timing for full_iteration to compare with test_performance
    events = prof.key_averages()
    full_iteration_events = [e for e in events if "full_iteration" in e.key]

    if full_iteration_events:
        cpu_time_ms = full_iteration_events[0].cpu_time_total / 1000
        cuda_time_ms = full_iteration_events[0].device_time_total / 1000
        print("\n--- Full iteration timing breakdown ---")
        print(f"CPU time (total): {cpu_time_ms:.2f}ms")
        print(f"CUDA time (total): {cuda_time_ms:.2f}ms")
        print(
            f"CPU overhead: {cpu_time_ms - cuda_time_ms:.2f}ms ({100*(cpu_time_ms - cuda_time_ms)/cpu_time_ms:.1f}% of total)"
        )
        print(
            "--- This should match test_performance.py latency (apples to apples) ---"
        )

    # Show top CPU time consumers
    print("\n--- PyTorch Profiler Results (Full Loop, Top 20 CPU time) ---")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=20
        )
    )

    assert (
        len(full_iteration_events) > 0
    ), "full_iteration not found in profiler output."
