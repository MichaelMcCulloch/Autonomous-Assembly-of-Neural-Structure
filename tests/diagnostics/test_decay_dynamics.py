# """
# Diagnostic tests for evaluating weight decay dynamics.

# This module provides systematic tests to evaluate different decay rules by measuring:
# 1. Weight norm stability (does it prevent explosion?)
# 2. Plasticity headroom (can Hebbian/Oja forces rescue useful connections?)
# 3. Pruning alignment (does decay pressure align with structural pruning?)
# 4. Block size sensitivity (do larger blocks get appropriate treatment?)

# These diagnostics are faster than full RL training and provide quantitative metrics
# for comparing decay strategies.
# """

# import numpy as np
# import torch
# from sbb.const import DEVICE
# from tests.common import (
#     DecayDiagnosticResults,
#     create_test_network,
#     measure_rescue_capability,
#     simulate_plasticity_dynamics,
# )


# def test_current_decay_rule():
#     """Test the currently implemented decay rule.
#     Run full diagnostic suite and compute all metrics.
#     """
#     network = create_test_network()

#     # 1. Simulate dynamics
#     weight_norms, thresholds, near_threshold_counts = simulate_plasticity_dynamics(
#         network, num_steps=1000, hebbian_strength=0.1
#     )

#     # 2. Weight norm dynamics
#     initial_norm = weight_norms[0]
#     final_norm = weight_norms[-1]
#     max_norm = max(weight_norms)

#     # Growth rate (log scale to handle both growth and decay)
#     norm_ratio = final_norm / (initial_norm + 1e-8)
#     norm_growth_rate = (norm_ratio - 1.0) / len(weight_norms)

#     # Stability score: penalize both explosion and collapse
#     # Ideal is modest growth (1.0 to 1.5 range)
#     if 1.0 <= norm_ratio <= 1.5:
#         norm_stability_score = 1.0
#     elif norm_ratio < 1.0:
#         # Collapse
#         norm_stability_score = max(0.0, norm_ratio)
#     else:
#         # Explosion
#         norm_stability_score = max(0.0, 1.0 - (norm_ratio - 1.5) / 10.0)

#     # 3. Rescue capability (test on fresh network to avoid equilibrium artifacts)
#     fresh_network = create_test_network()
#     rescue_rate = measure_rescue_capability(fresh_network, num_trials=500)

#     # False prune rate (assume 10% if rescue rate is low)
#     false_prune_rate = max(0.0, 0.1 - rescue_rate * 0.1)

#     # 4. Pruning alignment
#     threshold_mean = np.mean(thresholds)
#     near_threshold_mean = np.mean(near_threshold_counts)

#     # Count blocks far above threshold
#     with torch.no_grad():
#         active_slots = network.active_blocks.nonzero().squeeze(-1)
#         if active_slots.numel() > 0:
#             active_weights = network.weight_values[active_slots]
#             block_norms = torch.linalg.norm(
#                 active_weights.flatten(start_dim=-2), dim=-1
#             )
#             far_above = (block_norms > threshold_mean * 2.0).sum().item()
#         else:
#             far_above = 0

#     # 5. Block size sensitivity (requires creating networks with different sizes)
#     network_128 = create_test_network(num_blocks=16, neurons_per_block=128)
#     network_64 = create_test_network(num_blocks=16, neurons_per_block=64)
#     network_32 = create_test_network(num_blocks=16, neurons_per_block=32)
#     network_16 = create_test_network(num_blocks=16, neurons_per_block=16)

#     # Measure decay force per parameter by running one step
#     def measure_decay_force(net):
#         state = net.new_state(net.cfg.batch_size)
#         input_seq = (
#             torch.randn(
#                 1,
#                 net.cfg.batch_size,
#                 net.cfg.input_features,
#                 device=DEVICE,
#                 dtype=net.dtype,
#             )
#             * 0.1
#         )

#         # Store initial weights
#         with torch.no_grad():
#             active_slots = net.active_blocks.nonzero().squeeze(-1)
#             if active_slots.numel() == 0:
#                 return 0.0
#             initial_weights = net.weight_values[active_slots].clone()

#         # One plasticity step with zero Hebbian (pure decay)
#         state, trajectory = net.forward(input_seq, state)
#         state_norms_sq = torch.sum(trajectory**2, dim=-1, keepdim=True)
#         inv_norms = 1.0 / (state_norms_sq + 1e-8)

#         net.apply_plasticity(
#             system_states=trajectory,
#             eligibility_traces=state.eligibility_trace.unsqueeze(0),
#             activity_traces=state.homeostatic_trace.unsqueeze(0),
#             projected_fields=state.input_projection.unsqueeze(0),
#             variational_signal=torch.zeros_like(trajectory),  # No Hebbian
#             inverse_state_norms=inv_norms,
#         )

#         # Measure change per parameter
#         with torch.no_grad():
#             final_weights = net.weight_values[active_slots]
#             delta = (final_weights - initial_weights).abs().mean().item()
#             num_params = final_weights.numel()
#             return delta / num_params if num_params > 0 else 0.0

#     decay_force_128 = measure_decay_force(network_128)
#     decay_force_64 = measure_decay_force(network_64)
#     decay_force_32 = measure_decay_force(network_32)
#     decay_force_16 = measure_decay_force(network_16)

#     # Larger blocks should have smaller decay force per param
#     size_sensitivity_ratio = decay_force_16 / (decay_force_32 + 1e-10)

#     # 6. Overall health score (weighted composite)
#     overall_score = (
#         norm_stability_score * 0.4  # Weight norm stability is critical
#         + rescue_rate * 0.3  # Must allow plasticity to work
#         + (1.0 - false_prune_rate) * 0.2  # Don't kill useful connections
#         + min(1.0, size_sensitivity_ratio / 2.0) * 0.1  # Bonus for size-awareness
#     )

#     results = DecayDiagnosticResults(
#         initial_weight_norm=initial_norm,
#         final_weight_norm=final_norm,
#         max_weight_norm=max_norm,
#         norm_growth_rate=norm_growth_rate,
#         norm_stability_score=norm_stability_score,
#         rescue_success_rate=rescue_rate,
#         false_prune_rate=false_prune_rate,
#         pruning_threshold_mean=float(threshold_mean),
#         blocks_near_threshold=int(near_threshold_mean),
#         blocks_far_above_threshold=far_above,
#         decay_force_per_param_128x128=decay_force_128,
#         decay_force_per_param_64x64=decay_force_64,
#         decay_force_per_param_32x32=decay_force_32,
#         decay_force_per_param_16x16=decay_force_16,
#         size_sensitivity_ratio=size_sensitivity_ratio,
#         overall_score=overall_score,
#     )

#     """Print a human-readable diagnostic report."""
#     print("\n" + "=" * 70)
#     print("DECAY DYNAMICS DIAGNOSTIC REPORT")
#     print("=" * 70)

#     print("\n[1] Weight Norm Dynamics")
#     print(f"    Initial norm:      {results.initial_weight_norm:.3f}")
#     print(f"    Final norm:        {results.final_weight_norm:.3f}")
#     print(f"    Max norm:          {results.max_weight_norm:.3f}")
#     print(f"    Growth rate:       {results.norm_growth_rate:+.6f} per step")
#     print(f"    Stability score:   {results.norm_stability_score:.3f} / 1.0")

#     status = "✓ STABLE" if results.norm_stability_score > 0.8 else "⚠ UNSTABLE"
#     print(f"    Status:            {status}")

#     print("\n[2] Plasticity Headroom")
#     print(f"    Rescue success:    {results.rescue_success_rate:.1%}")
#     print(f"    False prune rate:  {results.false_prune_rate:.1%}")

#     status = "✓ GOOD" if results.rescue_success_rate > 0.5 else "⚠ LIMITED"
#     print(f"    Status:            {status}")

#     print("\n[3] Pruning Alignment")
#     print(f"    Avg threshold:     {results.pruning_threshold_mean:.4f}")
#     print(f"    Blocks near threshold: {results.blocks_near_threshold}")
#     print(f"    Blocks far above:      {results.blocks_far_above_threshold}")

#     print("\n[4] Block Size Sensitivity")
#     print(f"    Decay/param (128x128): {results.decay_force_per_param_128x128:.8f}")
#     print(f"    Decay/param (64x64):   {results.decay_force_per_param_64x64:.8f}")
#     print(f"    Decay/param (32x32):   {results.decay_force_per_param_32x32:.8f}")
#     print(f"    Decay/param (16x16):   {results.decay_force_per_param_16x16:.8f}")
#     print(f"    Sensitivity ratio:     {results.size_sensitivity_ratio:.2f}x")

#     status = "✓ SIZE-AWARE" if results.size_sensitivity_ratio > 1.5 else "⚠ SIZE-BLIND"
#     print(f"    Status:              {status}")

#     print("\n[5] Overall Health")
#     print(f"    Composite score:   {results.overall_score:.3f} / 1.0")

#     if results.overall_score > 0.8:
#         grade = "A (Excellent)"
#     elif results.overall_score > 0.6:
#         grade = "B (Good)"
#     elif results.overall_score > 0.4:
#         grade = "C (Fair)"
#     else:
#         grade = "D (Poor)"

#     print(f"    Grade:             {grade}")
#     print("\n" + "=" * 70 + "\n")

#     # Assert minimum requirements
#     assert results.norm_stability_score > 0.5, "Weight norms must be reasonably stable"
#     assert results.overall_score > 0.4, "Overall health must be at least fair"
