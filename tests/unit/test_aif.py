import pytest
import torch
from torch.distributions import Normal, Independent
from unittest.mock import MagicMock

from sbb import (
    ActiveInferenceHyperparameters,
    ActiveInferenceAgent,
    BaseModel,
)
from sbb.const import DEVICE

DEFAULT_PARAMS = {
    "num_blocks": 4,
    "neurons_per_block": 16,
    "input_features": 4,
    "output_features": 9,
}
DTYPES = [torch.float32]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)


@pytest.fixture(params=DTYPES)
def dtype(request):
    return request.param


@pytest.fixture
def aif_config(dtype):
    return ActiveInferenceHyperparameters(
        **DEFAULT_PARAMS,
        seed=42,
        dtype=dtype,
    )


@pytest.fixture
def aif_agent(aif_config):
    agent = ActiveInferenceAgent(aif_config)
    agent.to(DEVICE, aif_config.dtype)
    return agent


class TestActiveInferenceAgentUnit:
    def test_initialization(self, aif_agent, aif_config):
        assert isinstance(aif_agent.base, BaseModel)
        assert isinstance(aif_agent.readout.weight, torch.nn.Parameter)
        expected_shape = (aif_config.output_features, aif_config.total_neurons)
        assert aif_agent.readout.weight.shape == expected_shape
        assert aif_agent.readout.weight.dtype == aif_config.dtype
        assert aif_agent.cfg == aif_config

    @pytest.mark.parametrize("num_policies, horizon", [(1, 1), (4, 5), (4, 10)])
    def test_simulate_policy_outcomes_shapes(
        self, aif_agent, aif_config, num_policies, horizon
    ):
        initial_state = aif_agent.base.new_state(1)
        candidate_actions = torch.randn(
            num_policies,
            horizon,
            aif_config.input_features,
            device=DEVICE,
            dtype=aif_config.dtype,
        )
        internal_states, sensory_outcomes = aif_agent._simulate_policy_outcomes(
            initial_state, candidate_actions
        )
        assert internal_states.shape == (
            num_policies,
            horizon,
            aif_config.total_neurons,
        )
        assert sensory_outcomes.shape == (
            num_policies,
            horizon,
            aif_config.output_features,
        )
        assert internal_states.dtype == aif_config.dtype
        assert sensory_outcomes.dtype == aif_config.dtype

    def test_evaluate_efe_instrumental_value_driven(self, aif_agent, aif_config):
        horizon = aif_config.prediction_horizon
        num_policies = 4

        goal_loc = torch.ones(
            aif_config.output_features, device=DEVICE, dtype=aif_config.dtype
        )
        goal_dist = Independent(Normal(loc=goal_loc, scale=0.1), 1)
        best_policy_idx = 1

        internal_states = torch.randn(
            num_policies,
            horizon,
            aif_config.total_neurons,
            device=DEVICE,
            dtype=aif_config.dtype,
        )
        internal_states[:, -1, :] = torch.ones_like(internal_states[0, -1, :])

        sensory_outcomes = torch.full(
            (num_policies, horizon, aif_config.output_features),
            -1.0,
            device=DEVICE,
            dtype=aif_config.dtype,
        )
        sensory_outcomes[best_policy_idx] = goal_loc

        efe, _, _ = aif_agent._evaluate_expected_free_energy(
            internal_states, sensory_outcomes, goal_dist
        )

        assert efe.shape == (num_policies,)
        assert torch.argmin(efe).item() == best_policy_idx

    def test_evaluate_efe_epistemic_value_driven(self, aif_agent, aif_config):
        horizon = aif_config.prediction_horizon
        num_policies = 4

        goal_loc = torch.zeros(
            aif_config.output_features, device=DEVICE, dtype=aif_config.dtype
        )
        goal_scale = torch.ones(
            aif_config.output_features, device=DEVICE, dtype=aif_config.dtype
        )
        goal_dist = Independent(Normal(loc=goal_loc, scale=goal_scale), 1)

        sensory_outcomes = torch.zeros(
            num_policies,
            horizon,
            aif_config.output_features,
            device=DEVICE,
            dtype=aif_config.dtype,
        )

        low_variance_states = torch.randn(
            num_policies,
            horizon,
            aif_config.total_neurons,
            device=DEVICE,
            dtype=aif_config.dtype,
        )
        low_variance_states[:, -1, :] *= 1e-3

        high_variance_states = low_variance_states.clone()
        high_variance_states[:, -1, :] = (
            torch.randn_like(high_variance_states[:, -1, :]) * 10
        )

        efe_low_variance, _, _ = aif_agent._evaluate_expected_free_energy(
            low_variance_states, sensory_outcomes, goal_dist
        )
        efe_high_variance, _, _ = aif_agent._evaluate_expected_free_energy(
            high_variance_states, sensory_outcomes, goal_dist
        )

        assert efe_high_variance.sum() < efe_low_variance.sum()

    def test_act_runs_without_error_and_returns_valid_tensor(
        self, aif_agent, aif_config
    ):
        aif_agent.eval()

        initial_state = aif_agent.base.new_state(1)
        goal_loc = torch.zeros(
            aif_config.output_features, device=DEVICE, dtype=aif_config.dtype
        )
        goal_dist = Independent(Normal(loc=goal_loc, scale=1.0), 1)

        chosen_action, diagnostics = aif_agent.forward(initial_state, goal_dist)

        assert isinstance(chosen_action, torch.Tensor)
        assert chosen_action.shape == (aif_config.input_features,)
        assert chosen_action.dtype == aif_config.dtype
        assert torch.isfinite(chosen_action).all()
        assert torch.all(chosen_action >= 0.0) and torch.all(chosen_action <= 1.0)
        assert isinstance(diagnostics, dict)
        assert "efe" in diagnostics

    def test_backward_modifies_parameters_and_calls_scheduler(
        self, aif_agent, aif_config
    ):
        aif_agent.train()

        initial_head_weights = aif_agent.readout.weight.clone()
        mock_scheduler_step = MagicMock()
        aif_agent.base.plasticity.backward = mock_scheduler_step

        state_tuple = aif_agent.base.new_state(1)
        action = torch.randn(
            1,
            aif_config.input_features,
            device=DEVICE,
            dtype=aif_config.dtype,
        )
        observation = torch.randn(
            1,
            aif_config.output_features,
            device=DEVICE,
            dtype=aif_config.dtype,
        )

        state_tuple, _ = aif_agent.backward(state_tuple, action, observation)

        assert not torch.allclose(aif_agent.readout.weight, initial_head_weights)
        mock_scheduler_step.assert_called_once()
        _, call_kwargs = mock_scheduler_step.call_args
        assert "variational_gradient_trajectory" in call_kwargs
        feedback_signal = call_kwargs["variational_gradient_trajectory"]
        assert feedback_signal.shape == (1, 1, aif_config.total_neurons)
        assert torch.isfinite(feedback_signal).all()
