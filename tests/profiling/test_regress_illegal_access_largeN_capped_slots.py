import os
import pytest
import torch

from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import SupervisedConfig
from src.sbb.paradigms.predictive_coding import PredictiveCoding

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


def run_loop(model, steps, warmup):
    B = model.cfg.batch_size
    Fi, Fo = model.cfg.input_features, model.cfg.output_features
    _dev, dt = DEVICE, model.cfg.dtype
    xs = [torch.randn(B, Fi, device=DEVICE, dtype=dt) for _ in range(steps)]
    ys = [torch.randn(B, Fo, device=DEVICE, dtype=dt) for _ in range(steps)]
    st = model.base.new_state(B)
    for t in range(steps):
        pred, stn = model.forward(xs[t], st)
        if t >= warmup:
            _, st = model.backward(pred, ys[t], st, stn)
        else:
            st = stn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("num_blocks", [4096, 8192])
@pytest.mark.parametrize("neurons_per_block", [32])
def test_illegal_access_repro_with_capped_slots(
    batch_size, num_blocks, neurons_per_block
):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    cfg = SupervisedConfig(
        num_blocks=num_blocks,
        neurons_per_block=neurons_per_block,
        batch_size=batch_size,
        input_features=2,
        output_features=1,
        seed=42,
        evolution_substeps=16,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    cfg = cap_slots(cfg, cap_slots=min(16384, num_blocks * 4))

    cfg.max_norm = 1.0
    cfg.delta_max_norm = 0.1

    model = PredictiveCoding(cfg=cfg)
    model.train()

    run_loop(model, steps=8, warmup=0)
    torch.cuda.synchronize()

    run_loop(model, steps=32, warmup=8)
    torch.cuda.synchronize()

    assert torch.isfinite(model.base.weight_values).all()
    assert model.base.machine.bsr.row_ptr.numel() == num_blocks + 1
