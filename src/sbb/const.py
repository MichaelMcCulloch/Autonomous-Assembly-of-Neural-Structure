"""
Global constants for the SBB (Self-Building Box) system.

This module defines device configuration, numerical precision constants,
and limits for various computational operations.
"""

import torch

DEVICE = "cuda"
EPS = 1e-12
INDEX_DTYPE = torch.long
CUDA_WARP_SIZE = 32
MAX_QUANTILE_SIZE = 16_777_216
