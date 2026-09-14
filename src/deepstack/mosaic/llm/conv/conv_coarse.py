#
#   Conv1d(cin,cout,k,s):   M = bs*L_out,          N = cout, K = cin*k
#   Conv2d(cin,cout,kh,kw): M = bs*H_out*W_out,    N = cout, K = cin*kh*kw
#
#
import math
import numpy as np
from mosaic.parallelism import ParallelScheme
from mosaic.noc.noc_topo import Hierarchy
from mosaic.utils import OpBytes, Tensor_Loc, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import gemm_wrapper
from mosaic.op_dtype.element_wrapper import element_wrapper

from tilesight.arch import *
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
import logging
log = logging.getLogger(__name__)
from mosaic.cost.op_perf_stats import OpPerfStats


def _shard_m(m: int, parallel: ParallelScheme) -> int:
    return math.ceil(m / (parallel.dp * parallel.sp))


def _shard_n(n: int, parallel: ParallelScheme) -> int:
    return math.ceil(n / parallel.tp)


def conv_gemm_coarse(m:int, n:int, k:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, op_name:str="conv"):
    """Model the convolution core as implicit GEMM [m,k] @ [k,n], sharding m across
    dp*sp and n across tp.
    """
    assert granularity.get_mode() == "coarse"

    stats = OpPerfStats(op_name=op_name, dump_perf_log=granularity.dump_perf_log) if granularity.dump_perf_log else None

    shard_m = _shard_m(m, parallel)
    shard_n = _shard_n(n, parallel)

    gemm_bytes = OpBytes(
        input1=Tensor_Loc(conv_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(conv_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(conv_bytes.output.dtype, 'ddr'),
    )
    hete_post_data, smem_fusion_post_data, tiling_config = gemm_wrapper(M=shard_m, N=shard_n, K=k, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip)
    single_chip_time = smem_fusion_post_data[0]
    log.info("%s implicit-gemm M=%s N=%s K=%s (shard M=%s N=%s) time: %s s", op_name, m, n, k, shard_m, shard_n, single_chip_time)

    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return single_chip_time, stats


def _conv_implicit_coarse(n:int, f:int, h_out:int, w_out:int, c:int, kh:int, kw:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, op_name:str):
    from mosaic.op_dtype.conv_wrapper import conv_implicit_gemm_wrapper
    stats = OpPerfStats(op_name=op_name, dump_perf_log=granularity.dump_perf_log) if granularity.dump_perf_log else None
    shard_n = max(1, math.ceil(n / (parallel.dp * parallel.sp)))
    shard_f = _shard_n(f, parallel)
    _, post, _ = conv_implicit_gemm_wrapper(shard_n, shard_f, h_out, w_out, c, kh, kw, conv_bytes, granularity, single_chip)
    if stats is not None:
        stats.append_hete(post)
    log.info("%s (implicit-gemm) n=%s f=%s hw=%sx%s c=%s k=%sx%s time: %s s", op_name, n, f, h_out, w_out, c, kh, kw, post[0])
    return post[0], stats


def conv1d_coarse(bs:int, length:int, cin:int, cout:int, kernel:int, stride:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, dilation:int=1):
    l_out = math.ceil(length / stride)
    if kernel == 1:
        return conv_gemm_coarse(m=bs*l_out, n=cout, k=cin, parallel=parallel, conv_bytes=conv_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, op_name="conv1d_k1")
    return _conv_implicit_coarse(n=bs, f=cout, h_out=1, w_out=l_out, c=cin, kh=1, kw=kernel, parallel=parallel, conv_bytes=conv_bytes, granularity=granularity, single_chip=single_chip, op_name="conv1d")


def conv2d_coarse(bs:int, height:int, width:int, cin:int, cout:int, kernel:int, stride:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    h_out = math.ceil(height / stride)
    w_out = math.ceil(width / stride)
    return _conv_implicit_coarse(n=bs, f=cout, h_out=h_out, w_out=w_out, c=cin, kh=kernel, kw=kernel, parallel=parallel, conv_bytes=conv_bytes, granularity=granularity, single_chip=single_chip, op_name="conv2d")


def conv3d_patch_embed_coarse(num_patches:int, cin:int, embed_dim:int, kernel_elems:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    return conv_gemm_coarse(m=num_patches, n=embed_dim, k=cin*kernel_elems, parallel=parallel, conv_bytes=conv_bytes, granularity=granularity, single_chip=single_chip, noc_hierarchy=noc_hierarchy, op_name="conv3d_patch_embed")


def conv_transpose1d_coarse(bs:int, length:int, cin:int, cout:int, kernel:int, stride:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    assert granularity.get_mode() == "coarse"

    stats = OpPerfStats(op_name="conv_transpose1d", dump_perf_log=granularity.dump_perf_log) if granularity.dump_perf_log else None

    shard_m = _shard_m(bs * length, parallel)
    shard_n = _shard_n(cout * kernel, parallel)

    gemm_bytes = OpBytes(
        input1=Tensor_Loc(conv_bytes.input1.dtype, 'ddr'),
        input2=Tensor_Loc(conv_bytes.input2.dtype, 'ddr'),
        output=Tensor_Loc(conv_bytes.output.dtype, 'ddr'),
    )
    hete_post_data, smem_fusion_post_data, tiling_config = gemm_wrapper(M=shard_m, N=shard_n, K=cin, gemm_bytes=gemm_bytes, granularity=granularity, single_chip=single_chip)
    gemm_time = smem_fusion_post_data[0]
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)

    l_out = length * stride
    shard_cout = _shard_n(cout, parallel)
    shard_bs_l = _shard_m(bs * l_out, parallel)
    overlap_factor = max(1, math.ceil(kernel / stride))
    scatter_bytes = OpBytes(
        input1=Tensor_Loc(conv_bytes.input1.dtype, 'ddr', [shard_bs_l, shard_cout]),
        input2=None,
        output=Tensor_Loc(conv_bytes.output.dtype, 'ddr', [shard_bs_l, shard_cout]),
    )
    hete_post_data_scatter, tb_shape_scatter = element_wrapper(element_op_bytes=scatter_bytes, granularity=granularity, single_chip=single_chip, batch=overlap_factor, type="cuda_core", tb_tiling_config=None)
    reg_fused = hete_reg_fusion([hete_post_data_scatter], single_chip)
    grids = [shard_bs_l / tb_shape_scatter[0], shard_cout / tb_shape_scatter[1]]
    scatter_post_data = hete_smem_fusion([reg_fused], grids, single_chip)
    scatter_time = scatter_post_data[0]
    if stats is not None:
        stats.append_hete(scatter_post_data)

    total_time = gemm_time + scatter_time
    log.info("conv_transpose1d L=%s cin=%s cout=%s k=%s s=%s: gemm %s s + overlap-add %s s = %s s", length, cin, cout, kernel, stride, gemm_time, scatter_time, total_time)
    return total_time, stats


def _elementwise_pass_coarse(bs:int, length:int, channels:int, ops_per_elem:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, op_name:str):
    assert granularity.get_mode() == "coarse"

    stats = OpPerfStats(op_name=op_name, dump_perf_log=granularity.dump_perf_log) if granularity.dump_perf_log else None

    shard_c = _shard_n(channels, parallel)
    shard_bs_l = _shard_m(bs * length, parallel)

    ew_bytes = OpBytes(
        input1=Tensor_Loc(conv_bytes.input1.dtype, 'ddr', [shard_bs_l, shard_c]),
        input2=None,
        output=Tensor_Loc(conv_bytes.output.dtype, 'ddr', [shard_bs_l, shard_c]),
    )
    hete_post_data, tb_shape = element_wrapper(element_op_bytes=ew_bytes, granularity=granularity, single_chip=single_chip, batch=ops_per_elem, type="cuda_core", tb_tiling_config=None)
    reg_fused = hete_reg_fusion([hete_post_data], single_chip)
    grids = [shard_bs_l / tb_shape[0], shard_c / tb_shape[1]]
    smem_fusion_post_data = hete_smem_fusion([reg_fused], grids, single_chip)
    total_time = smem_fusion_post_data[0]
    log.info("%s L=%s C=%s ops/elem=%s time: %s s", op_name, length, channels, ops_per_elem, total_time)
    if stats is not None:
        stats.append_hete(smem_fusion_post_data)
    return total_time, stats


def depthwise_conv1d_coarse(bs:int, length:int, channels:int, kernel:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy):
    return _elementwise_pass_coarse(bs, length, channels, kernel, parallel, conv_bytes, granularity, single_chip, op_name="depthwise_conv1d")


def activation_1d_coarse(bs:int, length:int, channels:int, parallel:ParallelScheme, conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch, noc_hierarchy:Hierarchy, ops_per_elem:int=4):
    return _elementwise_pass_coarse(bs, length, channels, ops_per_elem, parallel, conv_bytes, granularity, single_chip, op_name="activation_1d")
