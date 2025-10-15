from unittest.mock import MagicMock, patch
import pytest
import torch
from torch.distributions import Normal, Independent
from sbb import ActiveInferenceHyperparameters, ActiveInferenceAgent
from sbb.const import DEVICE


FLAT_OBS_DIM = 9
ACTION_DIM = 4
BATCH_SIZE = 1
HORIZON = 5

DTYPES_TO_TEST = [torch.float32]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPES_TO_TEST.append(torch.bfloat16)


@pytest.fixture(params=DTYPES_TO_TEST)
def aif_config(request):
    return ActiveInferenceHyperparameters(
        prediction_horizon=HORIZON,
        num_stochastic_policies=0,
        include_deterministic_per_action=True,
        input_features=ACTION_DIM,
        output_features=FLAT_OBS_DIM,
        num_blocks=4,
        neurons_per_block=16,
        seed=42,
        dtype=request.param,
    )


@pytest.fixture
def aif_agent(aif_config):
    agent = ActiveInferenceAgent(aif_config)
    agent.to(DEVICE, aif_config.dtype)
    return agent


class TestActiveInferenceAgentIntegration:
    def test_world_model_learns_from_surprise(self, aif_agent, aif_config):
        aif_agent.train()

        initial_state = aif_agent.base.new_state(BATCH_SIZE)
        action_taken = torch.randn(
            BATCH_SIZE, ACTION_DIM, device=DEVICE, dtype=aif_config.dtype
        ).tanh()
        true_observation = torch.randn(
            BATCH_SIZE, FLAT_OBS_DIM, device=DEVICE, dtype=aif_config.dtype
        ).tanh()

        with torch.no_grad():
            _, initial_state_traj = aif_agent.base.forward(
                action_taken.unsqueeze(0), initial_state
            )

            initial_pred = torch.tanh(aif_agent.readout(initial_state_traj[0]))
            initial_surprise = torch.linalg.norm(initial_pred - true_observation).item()

        current_state = initial_state
        for _ in range(5):
            current_state, _ = aif_agent.backward(
                current_state, action_taken, true_observation
            )

        with torch.no_grad():
            _, final_state_traj = aif_agent.base.forward(
                action_taken.unsqueeze(0), initial_state
            )
            final_pred = torch.tanh(aif_agent.readout(final_state_traj[0]))
            final_surprise = torch.linalg.norm(final_pred - true_observation).item()

        assert final_surprise < initial_surprise
        assert initial_surprise > 1e-4
        assert torch.isfinite(torch.tensor(final_surprise))

    def test_planning_selects_goal_directed_action(self, aif_agent, aif_config):
        aif_agent.eval()

        goal_location = torch.ones(FLAT_OBS_DIM, device=DEVICE, dtype=aif_config.dtype)
        goal_distribution = Independent(Normal(loc=goal_location, scale=0.1), 1)

        bad_outcome = -torch.ones(
            1, HORIZON, FLAT_OBS_DIM, device=DEVICE, dtype=aif_config.dtype
        )
        good_outcome = (
            goal_location.clone().unsqueeze(0).unsqueeze(0).expand(-1, HORIZON, -1)
        )

        dummy_internal_states = torch.randn(
            ACTION_DIM,
            HORIZON,
            aif_config.total_neurons,
            device=DEVICE,
            dtype=aif_config.dtype,
        )

        mock_simulation = MagicMock()

        def side_effect_func(initial_state, candidate_actions):

            n_policies = candidate_actions.shape[0]
            winning_policy_idx = 1 if n_policies > 1 else 0

            sensory_outcomes = bad_outcome.repeat(n_policies, 1, 1)
            sensory_outcomes[winning_policy_idx] = good_outcome
            return dummy_internal_states, sensory_outcomes

        mock_simulation.side_effect = side_effect_func

        with patch.object(aif_agent, "_simulate_policy_outcomes", new=mock_simulation):
            initial_state = aif_agent.base.new_state(BATCH_SIZE)
            best_action, diags = aif_agent.forward(initial_state, goal_distribution)

        winning_idx = 1 if ACTION_DIM > 1 else 0
        expected_action = torch.nn.functional.one_hot(
            torch.tensor(winning_idx, device=DEVICE), num_classes=ACTION_DIM
        ).to(aif_config.dtype)

        assert torch.allclose(best_action, expected_action)
        mock_simulation.assert_called_once()


@pytest.fixture
def aif_bfloat16_config():
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        pytest.skip("bfloat16 not supported on this device.")
    return ActiveInferenceHyperparameters(
        prediction_horizon=5,
        num_stochastic_policies=0,
        include_deterministic_per_action=True,
        input_features=ACTION_DIM,
        output_features=FLAT_OBS_DIM,
        num_blocks=8,
        neurons_per_block=16,
        seed=1337,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def aif_agent_for_stability(aif_bfloat16_config):
    agent = ActiveInferenceAgent(aif_bfloat16_config)
    agent.to(DEVICE, aif_bfloat16_config.dtype)
    return agent


class TestActiveInferenceAgentStability:
    def test_agent_remains_stable_over_episode(
        self, aif_agent_for_stability, aif_bfloat16_config
    ):
        cfg = aif_bfloat16_config
        agent = aif_agent_for_stability
        agent.train()

        goal_location = torch.zeros(FLAT_OBS_DIM, device=DEVICE, dtype=cfg.dtype)
        goal_distribution = Independent(Normal(loc=goal_location, scale=0.5), 1)

        state_tuple = agent.base.new_state(batch_size=BATCH_SIZE)

        num_steps = 100
        for step in range(num_steps):
            action_tensor, _ = agent.forward(state_tuple, goal_distribution)
            assert torch.isfinite(
                action_tensor
            ).all(), f"Action became non-finite at step {step}"

            action_one_hot = torch.nn.functional.one_hot(
                torch.argmax(action_tensor, dim=-1).unsqueeze(0), num_classes=ACTION_DIM
            ).to(device=DEVICE, dtype=cfg.dtype)

            next_obs_tensor = torch.randn(
                BATCH_SIZE, FLAT_OBS_DIM, device=DEVICE, dtype=cfg.dtype
            ).tanh()

            state_tuple, _ = agent.backward(
                state_tuple, action_one_hot, next_obs_tensor
            )

            assert torch.isfinite(
                state_tuple.activations
            ).all(), f"Agent state became non-finite at step {step}"
