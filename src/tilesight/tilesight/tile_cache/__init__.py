"""tile_cache — Multi-level cache hit rate modeling.

Provides two-level SDCM cascade (L1.5 + L2) for tile reuse distance analysis.
L1.5 models passive per-group cache where ~8 physically adjacent SMs share a cache.

Two interfaces:
- General: multilevel_hit_rate_general(grid_shape, tensors, ...) — any access pattern
- GEMM:   multi_level_hit_rate(M, N, K, ..., arch) — auto-detects L1.5 from arch
"""

from .cache_model import multi_level_hit_rate
from .reuse_distance_multilevel import (
    multilevel_hit_rate_general,
    multilevel_hit_rate_reuse_distance,
    TensorAccess,
    CacheConfig,
)
