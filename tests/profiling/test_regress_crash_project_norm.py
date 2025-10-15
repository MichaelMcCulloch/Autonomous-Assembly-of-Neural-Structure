import os
import pytest
import torch

from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import SupervisedConfig
from src.sbb.paradigms.predictive_coding import PredictiveCoding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_project_l2_norm_illegal_memory_large_graph():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    cfg = SupervisedConfig(
        num_blocks=8192,
        neurons_per_block=32,
        batch_size=16,
        input_features=2,
        output_features=1,
        seed=42,
        evolution_substeps=16,
        dtype=torch.float32,
    )

    cfg.max_norm = 0.1
    cfg.delta_max_norm = 0.05

    model = PredictiveCoding(cfg=cfg)
    model.train()

    x = torch.randn(cfg.batch_size, cfg.input_features, device=DEVICE, dtype=cfg.dtype)
    y = torch.randn(cfg.batch_size, cfg.output_features, device=DEVICE, dtype=cfg.dtype)

    st0 = model.base.new_state(cfg.batch_size)
    pred, st1 = model.forward(x, st0)

    with pytest.raises(RuntimeError):

        model.backward(pred, y, st0, st1)
        torch.cuda.synchronize()
