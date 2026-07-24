# 非门控 2-GEMM MLP: y = W2 @ gelu(W1 @ x)
# W1 [hidden, up_hidden], W2 [up_hidden, out_hidden]; out_hidden 默认等于 hidden,
# 也可不同 (如 vision patch merger 4608->4608->2048, audio 出口 proj 1280->1280->2048)。
from mosaic.parallelism import ParallelScheme
from mosaic.noc.noc_topo import Hierarchy
from mosaic.utils import OpBytes, Modeling_Granularity
from tilesight.arch import *
import logging
from .mlp_gelu_coarse import mlp_gelu_coarse

log = logging.getLogger(__name__)


def mlp_gelu_top(bs:int, seq:int, hidden:int, up_hidden:int, parallel:ParallelScheme, next_parallel:ParallelScheme, mlp_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, out_hidden:"int | None" = None):
    if seq == 1:
        assert parallel.sp == 1

    if out_hidden is None:
        out_hidden = hidden

    if granularity.get_mode() == "coarse":
        total_time, stats = mlp_gelu_coarse(bs, seq, hidden, up_hidden, out_hidden, parallel, next_parallel, mlp_bytes, granularity, single_chip, noc_hierarchy)
        return total_time, stats
    elif granularity.get_mode() == "fine":
        pass
    elif granularity.get_mode() == "roof":
        pass
