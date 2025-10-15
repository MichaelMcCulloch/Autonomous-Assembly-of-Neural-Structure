from dataclasses import dataclass

from torch import Tensor


@dataclass
class _TrajectoryDataBuffer:
    system_states: Tensor
    actions: Tensor
    terminations: Tensor
    eligibility_traces: Tensor
    projected_fields: Tensor
    homeostatic_traces: Tensor
    biases: Tensor

    def __post_init__(self):

        self.dtype = self.system_states.dtype
        self.batch_size = self.system_states.shape[1]


@dataclass
class RLTrajectoryBuffer(_TrajectoryDataBuffer):
    rewards: Tensor
    state_value_estimates: Tensor
    action_log_probabilities: Tensor

    def __post_init__(self):
        super().__post_init__()
        self.steps = self.rewards.shape[0]
