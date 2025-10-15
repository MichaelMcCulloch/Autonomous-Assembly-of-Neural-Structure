from torch.nn import Parameter, Module
from torch import Tensor

from .topology import StructuralPlasticity, StructuralChangeReport
from .weights import SynapticPlasticity


class PlasticityController(Module):
    """
    Coordinate all forms of network adaptation in a per-timestep loop.

    This module integrates three adaptation mechanisms:
    1. Continuous synaptic plasticity (weight updates via SynapticPlasticity)
    2. Discrete structural plasticity (topology changes via StructuralPlasticity)
    3. Homeostatic regulation (bias adaptation toward target activity)

    Updates are applied per-timestep (not per-batch) to ensure chunking
    invariance: processing a sequence in chunks produces identical results
    to processing it whole. This design supports online learning and avoids
    artifacts from mini-batch averaging.

    The variational signal (error/reward) drives weight updates directly,
    supporting online continual learning without static gradient correction.

    Parameters
    ----------
    cfg : BaseConfig
        Configuration with learning rates and time constants.
    weight_values : Parameter
        Block-sparse recurrent weight tensor (shared with core network).
    active_blocks : Tensor
        Boolean mask for active connections (shared with core network).
    weight_rows, weight_cols : Tensor
        Sparse connectivity coordinates (shared with core network).
    in_degree, out_degree : Tensor
        Per-block connection counts (shared, updated during topology changes).
    trophic_support_map : Tensor
        Growth potential field [num_blocks, num_blocks].
    activity_bias : Parameter
        Homeostatic bias vector (shared with core network).

    Attributes
    ----------
    synaptic : SynapticPlasticity
        Applies continuous weight update rules.
    structural : StructuralPlasticity
        Applies discrete growth and pruning.
    """

    def __init__(
        self,
        cfg,
        weight_values: Parameter,
        active_blocks: Tensor,
        weight_rows: Tensor,
        weight_cols: Tensor,
        in_degree: Tensor,
        out_degree: Tensor,
        trophic_support_map: Tensor,
        activity_bias: Parameter,
    ):
        super().__init__()
        self.cfg = cfg

        self.dtype = cfg.dtype
        self.batch_size = cfg.batch_size
        self.num_blocks = cfg.num_blocks
        self.neurons_per_block = cfg.neurons_per_block

        self.activity_bias = activity_bias

        self.synaptic = SynapticPlasticity(
            dtype=cfg.dtype,
            batch_size=cfg.batch_size,
            neurons_per_block=cfg.neurons_per_block,
            num_blocks=cfg.num_blocks,
            weight_values=weight_values,
            weight_rows=weight_rows,
            weight_cols=weight_cols,
            active_blocks=active_blocks,
            in_degree=in_degree,
            out_degree=out_degree,
            trophic_support_map=trophic_support_map,
            trophic_map_ema_alpha=cfg.trophic_map_ema_alpha,
            max_norm=cfg.max_norm,
            delta_max_norm=cfg.delta_max_norm,
        )

        self.structural = StructuralPlasticity(
            dtype=cfg.dtype,
            weight_values=weight_values,
            weight_rows=weight_rows,
            weight_cols=weight_cols,
            active_blocks=active_blocks,
            in_degree=in_degree,
            out_degree=out_degree,
            trophic_support_map=trophic_support_map,
            initial_synaptic_efficacy=cfg.initial_synaptic_efficacy,
            initial_synaptic_polarity=cfg.initial_synaptic_polarity,
            target_connectivity=cfg.target_connectivity,
            structural_plasticity=True,
            neurons_per_block=cfg.neurons_per_block,
            num_blocks=cfg.num_blocks,
        )

    def backward(
        self,
        system_states_trajectory: Tensor,  # [T,B,N]
        eligibility_traces_trajectory: Tensor,  # [T,B,N]
        activity_traces_trajectory: Tensor,  # [T,B,N]
        inverse_state_norms_trajectory: Tensor,  # [T,B,1]
        variational_gradient_trajectory: Tensor,  # [T,B,N]
    ) -> StructuralChangeReport:
        """
        Apply all adaptation mechanisms for a sequence of timesteps.

        Loops over timesteps (not batches) to ensure chunking invariance.
        For each timestep:
        1. Update homeostatic bias toward target activity
        2. Apply synaptic weight updates (modulator, Hebbian, Oja, decay)
        3. Update eligibility trace factors (U, V)
        4. Attempt structural topology change (growth or pruning)

        Parameters
        ----------
        system_states_trajectory : Tensor [T, B, N]
            Hidden states from forward pass.
        eligibility_traces_trajectory : Tensor [T, B, N]
            Medium-timescale eligibility traces (τ_eligibility).
        activity_traces_trajectory : Tensor [T, B, N]
            Slow activity averages for homeostasis (τ_activity).
        projected_field_trajectory : Tensor [T, B, N]
            Projected external inputs (for gain modulation, currently unused).
        inverse_state_norms_trajectory : Tensor [T, B, 1]
            Precomputed 1/(||s||² + ε) for normalization.
        variational_gradient_trajectory : Tensor [T, B, N]
            Error/reward signal from paradigm (e.g., prediction error).

        Returns
        -------
        StructuralChangeReport
            Summary of final topology changes (grown, pruned counts).

        Notes
        -----
        In eval mode, no updates are applied and the last report is returned.
        The per-timestep loop ensures that the system behaves identically
        regardless of how sequences are chunked during training.
        """
        if not self.training:
            return self.structural.last_structural_change_report

        T, B, N = system_states_trajectory.shape

        dynamic_threshold = self.structural._cached_survival_threshold

        # Per-step loop for chunking invariance and pure-online learning
        last_report = StructuralChangeReport(grown=0, pruned=0)
        for t in range(T):
            s_t = system_states_trajectory[t]  # [B,N]
            e_t = eligibility_traces_trajectory[t]  # [B,N]
            inv_t = inverse_state_norms_trajectory[t]  # [B,1]
            act_t = activity_traces_trajectory[t]  # [B,N]
            var_t = variational_gradient_trajectory[t]  # [B,N]

            # Homeostasis per step: bias += lr * mean_batch( (setpoint - activity) * inv_norm )
            if self.cfg.activity_lr and self.cfg.activity_lr > 0:
                delta_bias = (self.cfg.homeostatic_setpoint - act_t) * inv_t
                batch_averaged_update = delta_bias.mean(dim=0, keepdim=True)  # [1,N]
                self.activity_bias.add_(
                    batch_averaged_update, alpha=self.cfg.activity_lr
                )

            # Post-synaptic Jacobian gate (instantaneous): 1 - s^2
            post_gain_t = 1.0 - s_t.pow(2)
            mod_eff_t = var_t * post_gain_t  # Jacobian-gated error

            # Synaptic update per step (no per-window averaging)
            self.synaptic.backward(
                system_states_trajectory=s_t.unsqueeze(0),  # [1,B,N]
                eligibility_traces_trajectory=e_t.unsqueeze(0),  # [1,B,N]
                inverse_state_norms_trajectory=inv_t.unsqueeze(0),  # [1,B,1]
                variational_gradient_trajectory=mod_eff_t.unsqueeze(0),  # [1,B,N]
                pruning_threshold=dynamic_threshold,
                post_gain=post_gain_t.unsqueeze(0),  # [1,B,N]
            )

            # Structural topology update per step
            last_report = self.structural.backward()

        return last_report

    def train(self, mode: bool = True):
        return self
