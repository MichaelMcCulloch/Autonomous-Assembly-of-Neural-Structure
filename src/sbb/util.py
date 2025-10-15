"""
Utility functions for matrix initialization, normalization, and configuration.

This module provides helper functions for orthogonal initialization,
block-level operations, L2 norm projection, and configuration serialization.
"""

from torch import Tensor
import torch
import dataclasses
import numpy as np
import yaml
from .const import DEVICE, EPS


def _orthogonal(dtype: torch.dtype, rows: int, columns: int) -> Tensor:
    """
    Generate an orthogonal matrix via QR decomposition.

    Creates a random Gaussian matrix and orthonormalizes its columns (if rows >= columns)
    or rows (if columns > rows) using numpy's QR factorization. This initialization
    preserves variance through linear projections and avoids vanishing/exploding signals.

    Parameters
    ----------
    dtype : torch.dtype
        Target data type (e.g., float32, bfloat16).
    rows : int
        Number of rows in output matrix.
    columns : int
        Number of columns in output matrix.

    Returns
    -------
    Tensor [rows, columns]
        Orthogonal matrix where columns (or rows) have unit norm and are mutually orthogonal.

    Notes
    -----
    Uses numpy for QR decomposition then converts to torch. Returns empty tensor if
    either dimension is zero.
    """
    if rows == 0 or columns == 0:
        return torch.empty(rows, columns, dtype=dtype, device=DEVICE)

    m = np.random.randn(rows, columns).astype(np.float32, copy=False)

    if rows >= columns:
        q, _ = np.linalg.qr(m, mode="reduced")
    else:
        qt, _ = np.linalg.qr(m.T, mode="reduced")
        q = qt.T

    return torch.from_numpy(q).to(device=DEVICE, dtype=dtype, non_blocking=True)


def _zero_blocks(
    dtype: torch.dtype,
    block_values: Tensor,
    target_blocks: Tensor,
    neurons_per_block: int,
):
    """
    Zero diagonal elements within specified weight blocks.

    Applies an identity mask to remove self-connections (diagonal elements) from
    weight blocks, typically used for diagonal blocks (row == col) where we want
    to prevent direct self-recurrence within a neural block.

    Parameters
    ----------
    dtype : torch.dtype
        Data type for mask.
    block_values : Tensor [num_blocks, block_size, block_size]
        Weight blocks to modify.
    target_blocks : Tensor [num_target]
        Indices of blocks to zero (typically diagonal blocks).
    neurons_per_block : int
        Size of each square block.

    Returns
    -------
    Tensor
        Modified block_values with diagonals zeroed in target_blocks.
    """
    if target_blocks.numel() == 0:
        return block_values
    identity_mask = torch.eye(neurons_per_block, dtype=dtype, device=DEVICE)
    block_values[target_blocks] *= 1.0 - identity_mask
    return block_values


def _project_l2_norm(
    tensor: Tensor,
    max_norm: float,
    eps: float = EPS,
):
    """
    Clip tensor to maximum L2 norm.

    If ||tensor||_2 > max_norm, scales tensor by (max_norm / ||tensor||_2).
    Otherwise, leaves tensor unchanged. Prevents parameter explosion during
    learning.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to project (modified in-place).
    max_norm : float
        Maximum allowed L2 norm.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        The input tensor (modified in-place).
    """
    if max_norm <= 0:
        return tensor
    norm = torch.linalg.norm(tensor)
    if norm > max_norm:
        tensor.mul_(max_norm / (norm + eps))
    return tensor


def _project_per_block_l2_norm(block_batch: Tensor, max_norm: float, eps: float = EPS):
    """
    Clip each block's L2 norm to maximum value.

    Applies per-block norm clipping for a batch of weight blocks. Each block
    is treated independently, with norms computed over the flattened block
    elements.

    Parameters
    ----------
    block_batch : Tensor [num_blocks, block_size, block_size]
        Batch of weight blocks to clip.
    max_norm : float
        Maximum allowed L2 norm per block.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        The input tensor with clipped blocks (modified in-place).
    """
    if max_norm <= 0 or block_batch.numel() == 0 or block_batch.dim() < 3:
        return block_batch

    norms = torch.linalg.norm(block_batch.flatten(start_dim=-2), dim=-1)
    clip_mask = norms > max_norm
    if not torch.any(clip_mask):
        return block_batch

    scale = (max_norm / (norms[clip_mask] + eps)).view(-1, 1, 1)
    block_batch[clip_mask] *= scale
    return block_batch


def config_to_dict(obj):
    """
    Convert a dataclass to a dictionary recursively.

    Handles nested dataclasses, tensors, numpy arrays, and standard Python types.
    Large tensors (>32 elements) are converted to shape/dtype strings.

    Parameters
    ----------
    obj : any
        Object to convert (typically a dataclass instance).

    Returns
    -------
    dict or any
        Dictionary representation with all nested structures converted.
    """
    if dataclasses.is_dataclass(obj):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = config_to_dict(value)
        return result
    elif isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [config_to_dict(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() > 32:
            return f"tensor(shape={list(obj.shape)}, dtype={obj.dtype})"
        return obj.tolist()
    elif isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def print_config(cfg, title: str = "Computed Hyperparameters"):
    """
    Pretty-print configuration with YAML formatting.

    Converts a configuration dataclass to YAML format and prints it with
    a formatted header. Falls back to JSON if YAML serialization fails.

    Parameters
    ----------
    cfg : BaseConfig or dataclass
        Configuration object to display.
    title : str, optional
        Header title for the output. Default "Computed Hyperparameters".
    """
    config_dict = config_to_dict(cfg)

    print("\n" + "=" * 80)
    print(title)
    print("-" * len(title))
    try:
        print(
            yaml.dump(
                config_dict,
                Dumper=yaml.SafeDumper,
                sort_keys=False,
                indent=2,
                default_flow_style=False,
            )
        )
    except Exception as e:
        print(f"An error occurred during YAML serialization: {e}")
        print("Falling back to basic print:")
        import json

        print(json.dumps(config_dict, indent=2))

    print("=" * 80 + "\n")
