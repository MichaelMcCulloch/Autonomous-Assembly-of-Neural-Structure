from dataclasses import dataclass
from torch import Tensor


@dataclass
class SystemStateTuple:
    """
    Complete state snapshot for the recurrent dynamical system.

    This dataclass encapsulates all transient variables needed to resume
    computation from a given point. The system's forward pass is a pure
    function: identical inputs and initial state always produce identical
    outputs, enabling reproducible chunked sequence processing.

    Attributes
    ----------
    activations : Tensor [B, N]
        Fast-timescale neural activations (τ_fast ≈ 0.02s).
        Represents the current "thought" or computational state.
    eligibility_trace : Tensor [B, N]
        Medium-timescale trace for credit assignment (τ_eligibility ≈ 0.2s).
        Tags recently active synapses for modification when error arrives.
    homeostatic_trace : Tensor [B, N]
        Slow-timescale running average for homeostasis (τ_activity ≈ 10s).
        Drives long-term regulation toward target firing rate.
    bias : Tensor [B, N]
        Per-neuron adaptive bias learned to maintain target activity.
        Evolves slowly to compensate for network-wide imbalances.
    input_projection : Tensor [B, N]
        Most recent input after projection: tanh(u @ weight_in.T).
        Cached for use in plasticity updates (gain modulation).
    noise : Tensor [B]
        Deterministic RNG counter for noise injection.
        Incremented each timestep to ensure chunk-invariant randomness.

    Notes
    -----
    All tensors use the network's configured dtype (typically bfloat16 or float32)
    except noise (int64). Batch dimension B allows parallel sequence processing.
    The tuple is immutable after creation (use replace() to modify fields).
    """

    activations: Tensor
    eligibility_trace: Tensor
    homeostatic_trace: Tensor
    bias: Tensor
    input_projection: Tensor
    noise: Tensor
