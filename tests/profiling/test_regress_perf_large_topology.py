import os
import pytest
import torch
import time

from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import SupervisedConfig
from src.sbb.paradigms.predictive_coding import PredictiveCoding

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _format_hz(x):
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f} G"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    if x >= 1_000:
        return f"{x/1_000:.2f} k"
    return f"{x:.2f} "


def _run_loop(model, total_steps, warmup_steps):
    dtype = model.cfg.dtype
    B = model.cfg.batch_size
    Fi = model.cfg.input_features
    Fo = model.cfg.output_features

    xs = [torch.randn(B, Fi, device=DEVICE, dtype=dtype) for _ in range(total_steps)]
    ys = [torch.randn(B, Fo, device=DEVICE, dtype=dtype) for _ in range(total_steps)]

    st = model.base.new_state(B)
    density_sum = 0.0
    measured = 0

    for t in range(total_steps):
        pred, st_next = model.forward(xs[t], st)
        if t >= warmup_steps:
            _, st = model.backward(pred, ys[t], st, st_next)
            num_active = model.base.active_blocks.sum()
            density = num_active / (model.cfg.num_blocks**2)
            density_sum += float(density)
            measured += 1
        else:
            st = st_next

    return (density_sum / measured) if measured > 0 else 0.0


large_params = dict(
    neurons_per_block=32,
    num_blocks=8192,
    evolution_substeps=16,
)

batches = [64, 128]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", batches)
def test_large_topology_illegal_memory_and_perf(batch_size):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    cfg = SupervisedConfig(
        num_blocks=large_params["num_blocks"],
        neurons_per_block=large_params["neurons_per_block"],
        batch_size=batch_size,
        input_features=2,
        output_features=1,
        seed=42,
        evolution_substeps=large_params["evolution_substeps"],
        dtype=torch.float32,
    )
    model = PredictiveCoding(cfg=cfg)
    model.train()

    warmup_steps = 10
    total_steps = 30

    _run_loop(model, warmup_steps, 0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    avg_density = _run_loop(model, total_steps, warmup_steps)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    measured = total_steps - warmup_steps
    items_per_sec = (measured * cfg.batch_size) / max(dt, 1e-9)
    latency_ms = (dt / measured) * 1000.0

    print(
        f"N/B={cfg.neurons_per_block} B={cfg.num_blocks} batch={batch_size} "
        f"res={cfg.evolution_substeps} "
        f"thr={_format_hz(items_per_sec)}items/s lat={latency_ms:.3f}ms avgDens={avg_density:.4f}"
    )

    assert torch.isfinite(model.base.weight_values).all()

    assert (
        items_per_sec < 1_000_000
    ), "Unexpectedly high throughput; regression not reproduced"
    assert latency_ms > 5.0, "Latency too low; regression not reproduced"
