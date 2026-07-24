# conv 系列算子的 top 层入口 (granularity 分发), 与其他 kernel 的 *_top 约定一致:
# 返回 (time, stats)。
from mosaic.parallelism import ParallelScheme
from mosaic.noc.noc_topo import Hierarchy
from mosaic.utils import OpBytes, Modeling_Granularity
from tilesight.arch import *
import logging
from .conv_coarse import (
    activation_1d_coarse,
    conv1d_coarse,
    conv2d_coarse,
    conv3d_patch_embed_coarse,
    conv_transpose1d_coarse,
    depthwise_conv1d_coarse,
)

log = logging.getLogger(__name__)


def conv1d_top(bs:int, length:int, cin:int, cout:int, kernel:int, stride:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, dilation:int=1):
    if granularity.get_mode() == "coarse":
        return conv1d_coarse(bs, length, cin, cout, kernel, stride, parallel, conv_bytes, granularity, single_chip, noc_hierarchy, dilation=dilation)
    raise NotImplementedError(granularity.get_mode())


def conv2d_top(bs:int, height:int, width:int, cin:int, cout:int, kernel:int, stride:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    if granularity.get_mode() == "coarse":
        return conv2d_coarse(bs, height, width, cin, cout, kernel, stride, parallel, conv_bytes, granularity, single_chip, noc_hierarchy)
    raise NotImplementedError(granularity.get_mode())


def conv3d_patch_embed_top(num_patches:int, cin:int, embed_dim:int, kernel_elems:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    if granularity.get_mode() == "coarse":
        return conv3d_patch_embed_coarse(num_patches, cin, embed_dim, kernel_elems, parallel, conv_bytes, granularity, single_chip, noc_hierarchy)
    raise NotImplementedError(granularity.get_mode())


def conv_transpose1d_top(bs:int, length:int, cin:int, cout:int, kernel:int, stride:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    if granularity.get_mode() == "coarse":
        return conv_transpose1d_coarse(bs, length, cin, cout, kernel, stride, parallel, conv_bytes, granularity, single_chip, noc_hierarchy)
    raise NotImplementedError(granularity.get_mode())


def depthwise_conv1d_top(bs:int, length:int, channels:int, kernel:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    if granularity.get_mode() == "coarse":
        return depthwise_conv1d_coarse(bs, length, channels, kernel, parallel, conv_bytes, granularity, single_chip, noc_hierarchy)
    raise NotImplementedError(granularity.get_mode())


def activation_1d_top(bs:int, length:int, channels:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, ops_per_elem:int=4):
    if granularity.get_mode() == "coarse":
        return activation_1d_coarse(bs, length, channels, parallel, conv_bytes, granularity, single_chip, noc_hierarchy, ops_per_elem=ops_per_elem)
    raise NotImplementedError(granularity.get_mode())
