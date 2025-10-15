import pytest
import torch
import gc
from tabulate import tabulate
from tqdm import tqdm

from sbb.const import DEVICE
from src.sbb.paradigms.predictive_coding import SupervisedConfig
from src.sbb.paradigms.predictive_coding import PredictiveCoding


def validate_model_step(
    num_blocks: int,
    neurons_per_block: int,
    dtype: torch.dtype,
    device: str,
    num_growth_steps: int = 100,
) -> tuple[bool, float, float]:
    """
    Instantiates the model, performs multiple learning steps to allow structural
    plasticity to grow connections, and measures peak memory and final density.

    This gives a realistic capacity measurement accounting for network growth.

    Parameters
    ----------
    num_blocks : int
        Number of blocks in the network
    neurons_per_block : int
        Neurons per block
    dtype : torch.dtype
        Data type (bfloat16/float32)
    device : str
        Device ('cuda' or 'cpu')
    num_growth_steps : int
        Number of learning steps to run for structural growth (default: 100)

    Returns
    -------
    tuple[bool, float, float]
        (success, peak_memory_mb, final_density)
    """
    model = None
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        cfg = SupervisedConfig(
            num_blocks=num_blocks,
            neurons_per_block=neurons_per_block,
            dtype=dtype,
            input_features=2,
            output_features=1,
            batch_size=32,
        )
        model = PredictiveCoding(cfg=cfg)
        model.train()

        state_tuple = model.base.new_state(cfg.batch_size)

        # Run multiple steps to allow structural plasticity to grow connections
        # This gives a realistic memory measurement
        for _ in range(num_growth_steps):
            dummy_input = torch.randn(
                cfg.batch_size, cfg.input_features, device=DEVICE, dtype=dtype
            )
            dummy_target = torch.randn(
                cfg.batch_size, cfg.output_features, device=DEVICE, dtype=dtype
            )
            pred, next_state_tuple = model.forward(dummy_input, state_tuple)
            model.backward(pred, dummy_target, state_tuple, next_state_tuple)
            state_tuple = next_state_tuple

        # Calculate final network density
        num_active_blocks = model.base.active_blocks.sum().item()
        final_density = num_active_blocks / (cfg.num_blocks**2)

        if device == "cuda":
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_memory_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024**2)

        del model
        return True, peak_memory_mb, final_density

    except Exception:
        gc.collect()
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            return False, 0.0, 0.0
        return False, 0.0, 0.0


def find_max_n_for_params(
    neurons_per_block: int,
    dtype: torch.dtype,
    device: str,
    n_search_range: tuple[int, int],
    pbar: tqdm,
) -> tuple[int, float, float]:
    """Performs a binary search to find the maximum supportable N."""
    low, high = n_search_range
    last_successful_n = 0
    last_successful_mem = 0.0
    last_successful_density = 0.0

    low_b = low // neurons_per_block
    high_b = high // neurons_per_block

    while low_b <= high_b:
        mid_b = low_b + (high_b - low_b) // 2
        if mid_b == 0:
            low_b = mid_b + 1
            continue

        pbar.set_postfix_str(f"Searching N={mid_b:,}")

        success, mem_used, density = validate_model_step(
            num_blocks=mid_b,
            neurons_per_block=neurons_per_block,
            dtype=dtype,
            device=DEVICE,
        )

        if success:
            last_successful_n = mid_b * neurons_per_block
            last_successful_mem = mem_used
            last_successful_density = density
            low_b = mid_b + 1
        else:
            high_b = mid_b - 1

    return last_successful_n, last_successful_mem, last_successful_density


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_max_network_size():
    """
    Finds the maximum supportable network size accounting for structural growth.

    Runs 100 learning steps to allow structural plasticity to grow connections,
    providing a realistic capacity measurement rather than just initial connectivity.
    """
    dtype = torch.bfloat16
    n_search_range = (32 * 10240, 32 * 20480)

    block_sizes = [32]

    results = []

    with tqdm(block_sizes, desc="Testing Configurations", unit="config") as pbar:
        for bs in pbar:
            pbar.set_description(f"Config (BS={bs})")
            max_n, mem_used, density = find_max_n_for_params(
                neurons_per_block=bs,
                dtype=dtype,
                device=DEVICE,
                n_search_range=n_search_range,
                pbar=pbar,
            )
            if max_n > 0:
                results.append((bs, max_n, mem_used, density))

    print("\n--- Maximum Supportable N (VRAM, with structural growth) ---")
    print("Note: Measured after 100 learning steps with structural plasticity")
    headers = ["Block Size", "Max N", "Peak Memory (MB)", "Final Density"]
    table_data = [(r[0], f"{r[1]:,}", f"{r[2]:.2f}", f"{r[3]:.4f}") for r in results]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    assert len(results) > 0, "No successful configurations found"

    assert any(
        r[1] >= 32768 for r in results
    ), "Failed to support a large network size (N>=32768)"
