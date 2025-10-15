import pytest
import torch

from sbb.const import DEVICE
from sbb.types import SystemStateTuple
from src.sbb.paradigms.predictive_coding import (
    PredictiveCoding,
    SupervisedConfig,
)


@pytest.fixture
def supervised_config():
    """Provides a default configuration for the PredictiveCoding model."""
    cfg = SupervisedConfig(
        num_blocks=4,
        neurons_per_block=16,
        input_features=16,
        output_features=4,
        batch_size=2,
        dtype=torch.float32,
        seed=42,
    )

    cfg.activity_lr = 0.01
    return cfg


@pytest.fixture
def supervised_model(supervised_config):
    """Creates a PredictiveCoding model instance."""
    model = PredictiveCoding(cfg=supervised_config)
    return model


def test_supervised_initialization(supervised_model, supervised_config):
    """Tests if the PredictiveCoding model initializes its components correctly."""
    c = supervised_config
    model = supervised_model

    assert model.readout is not None
    assert model.readout.weight.shape == (c.output_features, c.total_neurons)
    assert model.base.machine is not None
    assert model.base.plasticity is not None

    initial_state_tuple = model.base.new_state(c.batch_size)
    assert isinstance(initial_state_tuple, SystemStateTuple)
    assert initial_state_tuple.activations.shape == (c.batch_size, c.total_neurons)


def test_supervised_forward_pass(supervised_model, supervised_config):
    """Tests the forward pass, which now includes the readout."""
    c = supervised_config
    model = supervised_model
    input_tensor = torch.randn(
        c.batch_size, c.input_features, device=DEVICE, dtype=c.dtype
    )

    initial_state_tuple = model.base.new_state(c.batch_size)
    predictions, next_state_tuple = model.forward(input_tensor, initial_state_tuple)

    assert predictions.shape == (c.batch_size, c.output_features)
    assert not torch.allclose(
        next_state_tuple.activations, initial_state_tuple.activations
    )


def test_supervised_learn_step(supervised_model, supervised_config):
    """Tests a single learning step for the PredictiveCoding model."""
    c = supervised_config
    model = supervised_model
    model.train()

    input_tensor = torch.ones(
        c.batch_size, c.input_features, device=DEVICE, dtype=c.dtype
    )

    initial_state_tuple = model.base.new_state(c.batch_size)
    predictions, next_state_tuple = model.forward(input_tensor, initial_state_tuple)

    initial_readout = model.readout.weight.clone()
    initial_weight_values = model.base.weight_values.clone()
    initial_trophic_support_map = model.base.trophic_support_map.clone()
    initial_bias = initial_state_tuple.bias.clone()

    targets = torch.randn(c.batch_size, c.output_features, device=DEVICE, dtype=c.dtype)

    loss, final_state_tuple = model.backward(
        predictions, targets, initial_state_tuple, next_state_tuple
    )

    assert loss.ndim == 0

    assert not torch.allclose(model.readout.weight, initial_readout)
    assert not torch.allclose(final_state_tuple.bias, initial_bias, atol=1e-10)

    error = predictions - targets
    if error.norm() > 1e-6:
        assert not torch.allclose(model.base.weight_values, initial_weight_values)
        assert not torch.allclose(
            model.base.trophic_support_map,
            initial_trophic_support_map,
            atol=1e-20,
        )

    assert model.base.machine.bsr._values_stale
