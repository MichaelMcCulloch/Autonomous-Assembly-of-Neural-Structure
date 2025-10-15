import pytest
import torch


from sbb.const import DEVICE
from src.sbb.paradigms.policy_gradient import (
    ContinuousPolicyGradientAgent,
    DiscretePolicyGradientAgent,
    GAEReturnEstimator,
    ReinforcementLearningConfig,
    RLTrajectoryBuffer,
)

from src.sbb.types import SystemStateTuple


@pytest.fixture(scope="module")
def rl_config():
    """Provides a default configuration for the RL model."""
    cfg = ReinforcementLearningConfig(
        num_blocks=4,
        neurons_per_block=16,
        input_features=16,
        output_features=2,
        batch_size=2,
        dtype=torch.float32,
        seed=42,
        value_lr=0.01,
        policy_lr=0.01,
    )

    cfg.decay_lr = 0.001
    cfg.activity_lr = 0.01
    return cfg


@pytest.fixture(scope="module")
def continuous_rl_model(rl_config):
    """Creates a ContinuousPolicyGradientAgent model instance."""
    return ContinuousPolicyGradientAgent(
        rl_config,
        GAEReturnEstimator(gae_lambda=0.95, gamma=0.99, dtype=rl_config.dtype),
    )


@pytest.fixture(scope="module")
def discrete_rl_model(rl_config):
    """Creates a DiscretePolicyGradientAgent model instance."""
    return DiscretePolicyGradientAgent(
        rl_config,
        GAEReturnEstimator(gae_lambda=0.95, gamma=0.99, dtype=rl_config.dtype),
    )


@pytest.fixture(scope="module")
def sample_rollout_buffer(rl_config):
    """Creates a sample PolicyGradientTrajectoryBuffer for testing the learn method."""
    steps = 5
    B = rl_config.batch_size
    N = rl_config.total_neurons
    action_dim = rl_config.output_features
    dtype = rl_config.dtype

    states = torch.randn(steps + 1, B, N, device=DEVICE, dtype=dtype)
    activity_traces = torch.full_like(states, rl_config.homeostatic_setpoint * 0.5)
    activity_biases = torch.zeros_like(states)

    return RLTrajectoryBuffer(
        system_states=states,
        actions=torch.randn(steps, B, action_dim, device=DEVICE, dtype=dtype),
        rewards=torch.ones(steps, B, device=DEVICE, dtype=dtype),
        state_value_estimates=torch.zeros(steps + 1, B, device=DEVICE, dtype=dtype),
        action_log_probabilities=torch.randn(steps, B, device=DEVICE, dtype=dtype),
        terminations=torch.zeros(steps, B, device=DEVICE, dtype=torch.bool),
        eligibility_traces=torch.randn(steps + 1, B, N, device=DEVICE, dtype=dtype),
        projected_fields=torch.randn(steps + 1, B, N, device=DEVICE, dtype=dtype),
        homeostatic_traces=activity_traces,
        biases=activity_biases,
    )


def test_reinforcement_initialization(continuous_rl_model, rl_config):
    """Tests if the RL model initializes its components correctly."""
    model = continuous_rl_model
    assert model.recurrent_policy_network.machine is not None
    assert model.recurrent_policy_network.plasticity is not None
    assert model.policy_head is not None
    assert model.value_head.weight.shape == (1, rl_config.total_neurons)


def test_get_value(continuous_rl_model, rl_config):
    """Tests the value function computation."""
    model = continuous_rl_model
    state = torch.randn(rl_config.batch_size, rl_config.total_neurons, device=DEVICE)
    value = model._estimate_state_value(state)
    assert value.shape == (rl_config.batch_size,)


def test_act_continuous(continuous_rl_model, rl_config):
    """Tests continuous action selection."""
    model = continuous_rl_model
    state_tensor = torch.randn(
        rl_config.batch_size, rl_config.total_neurons, device=DEVICE
    )
    state_tuple = SystemStateTuple(
        activations=state_tensor,
        eligibility_trace=torch.zeros_like(state_tensor),
        homeostatic_trace=torch.zeros_like(state_tensor),
        bias=torch.zeros_like(state_tensor),
        input_projection=torch.zeros_like(state_tensor),
        noise=torch.zeros_like(state_tensor),
    )

    action, log_prob, value = model.act(state_tuple)
    assert action.shape == (rl_config.batch_size, rl_config.output_features)
    assert log_prob.shape == (rl_config.batch_size,)
    assert value.shape == (rl_config.batch_size,)


def test_reinforcement_learn_step(continuous_rl_model, sample_rollout_buffer):
    """Tests a single learning step for the PolicyGradientAgent model."""
    model = continuous_rl_model
    model.train()

    initial_value_head = model.value_head.weight.clone()
    initial_policy_head_mean = model.policy_head.mean_weight.clone()
    initial_weight_values = model.recurrent_policy_network.weight_values.clone()

    loss = model.backward(sample_rollout_buffer)

    assert loss.ndim == 0

    assert not torch.allclose(model.value_head.weight, initial_value_head)
    assert not torch.allclose(model.policy_head.mean_weight, initial_policy_head_mean)
    assert not torch.allclose(
        model.recurrent_policy_network.weight_values, initial_weight_values
    )

    assert model.recurrent_policy_network.machine.bsr._topology_changed
