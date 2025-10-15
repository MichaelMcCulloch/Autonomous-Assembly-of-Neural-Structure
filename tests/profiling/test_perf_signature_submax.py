import pytest
import torch
import time
from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import SupervisedConfig
from src.sbb.paradigms.predictive_coding import PredictiveCoding


def cap_slots(cfg, cap_slots):

    rows = cfg.initial_connectivity_map_rows.detach().cpu()
    cols = cfg.initial_connectivity_map_cols.detach().cpu()
    if rows.numel() > cap_slots:
        rows = rows[:cap_slots]
        cols = cols[:cap_slots]
    cfg.max_synaptic_connections = int(cap_slots)
    cfg.initial_connectivity_map_rows = rows.to(DEVICE)
    cfg.initial_connectivity_map_cols = cols.to(DEVICE)

    cfg.target_connectivity = max(1, min(2, cfg.target_connectivity or 1))
    cfg.structural_plasticity = True
    return cfg


def sps(model, total_steps=100, warmup=20):
    B = model.cfg.batch_size
    Fi, Fo = model.cfg.input_features, model.cfg.output_features
    _dev, dt = DEVICE, model.cfg.dtype
    xs = [torch.randn(B, Fi, device=DEVICE, dtype=dt) for _ in range(total_steps)]
    ys = [torch.randn(B, Fo, device=DEVICE, dtype=dt) for _ in range(total_steps)]
    st = model.base.new_state(B)
    for t in range(warmup):
        st = model.forward(xs[t], st)[1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(warmup, total_steps):
        pred, stn = model.forward(xs[t], st)
        _, st = model.backward(pred, ys[t], st, stn)
    torch.cuda.synchronize()
    dt_wall = time.perf_counter() - t0
    return (total_steps - warmup) * B / max(dt_wall, 1e-9), (
        dt_wall / (total_steps - warmup)
    ) * 1000.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_blocks,bs,cap", ((4096, 32, 8192), (2048, 32, 4096)))
@pytest.mark.parametrize("batch_size", [32])
def test_perf_signature(num_blocks, bs, cap, batch_size):
    cfg = SupervisedConfig(
        num_blocks=num_blocks,
        neurons_per_block=bs,
        batch_size=batch_size,
        input_features=2,
        output_features=1,
        seed=0,
        evolution_substeps=16,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    cfg = cap_slots(cfg, cap)
    cfg.max_norm = 2.0
    cfg.delta_max_norm = 0.2

    model = PredictiveCoding(cfg)
    model.train()
    thr, lat = sps(model)

    assert thr < 200000
    assert lat > 2.0
