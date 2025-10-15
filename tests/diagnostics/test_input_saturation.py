import torch
from sbb.const import DEVICE
from sbb.paradigms.predictive_coding import SupervisedConfig
from sbb.base import BaseModel


def test_input_projection_not_saturated():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    # Small but representative core
    cfg = SupervisedConfig(
        num_blocks=32,
        neurons_per_block=32,
        input_features=1,
        output_features=1,
        batch_size=8,
        dtype=dtype,
        seed=123,
        noise=0.0,
    )
    integ = BaseModel(cfg).to(device, dtype)
    # Synthetic inputs with variance similar to tasks (MG/NARMA in [-0.9,0.9])
    T = 256
    x = (
        torch.rand(T, cfg.batch_size, cfg.input_features, device=DEVICE, dtype=dtype)
        - 0.5
    ) * 1.8
    # Pre-tanh activations (no nonlinearity)
    z = x @ integ.machine.weight_in.T
    frac_pre = (z.abs() > 2.0).float().mean().item()  # tanh(|z|=2)≈0.964
    # Projected input actually used
    y = torch.tanh(z)
    frac_post = (y.abs() > 0.98).float().mean().item()
    # Assert: large swaths of network must remain in linear/weakly nonlinear regime
    assert frac_pre <= 0.15, f"Pre-tanh saturation too high: {frac_pre:.3f}"
    assert frac_post <= 0.50, f"Tanh output saturation too high: {frac_post:.3f}"


def test_state_gain_not_collapsed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    cfg = SupervisedConfig(
        num_blocks=32,
        neurons_per_block=32,
        input_features=3,
        output_features=3,
        batch_size=8,
        dtype=dtype,
        seed=321,
        noise=0.0,
    )
    integ = BaseModel(cfg).to(device, dtype)
    T = 128
    x = (
        torch.randn(T, cfg.batch_size, cfg.input_features, device=DEVICE, dtype=dtype)
        * 0.5
    )
    st0 = integ.new_state(cfg.batch_size)
    stf, traj = integ.forward(x, st0)
    # Post-gain proxy: 1 - s^2; if tanh is saturating everywhere this collapses near 0
    gain = 1.0 - traj.pow(2)
    mean_gain = gain.mean().item()
    frac_low_gain = (gain < 0.1).float().mean().item()
    assert mean_gain >= 0.2, f"Mean post-gain too small: {mean_gain:.3f}"
    assert frac_low_gain <= 0.5, f"Too many units with tiny gain: {frac_low_gain:.3f}"
