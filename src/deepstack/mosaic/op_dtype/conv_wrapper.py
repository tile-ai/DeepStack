# conv 的 implicit-GEMM 建模 wrapper: 走 TileSight 的
# conv_implicit_gemm_fused_op (带 conv 专用 L2 复用/halo 命中率模型),
# 结构与 gemm_wrapper 一致 (默认 tiling + hete 后处理)。
import math
import numpy as np
from mosaic.utils import OpBytes, Modeling_Granularity
from mosaic.op_dtype.gemm_wrapper import get_default_tiling
from tilesight.arch import Arch
from tilesight.fused_op_dtype.conv_implicit_gemm_fused_op import (
    calculate_conv_implicit_gemm_resource_utilization,
)
from tilesight.fusion_support.hete_post_process_single_op import hete_post_process_tensor_core_op
from tilesight.fusion_support.hete_reg_fusion import hete_reg_fusion
from tilesight.fusion_support.hete_smem_fusion import hete_smem_fusion
import logging
log = logging.getLogger(__name__)


def conv_implicit_gemm_wrapper(n:int, f:int, h_out:int, w_out:int, c:int, kh:int, kw:int,
                               conv_bytes:OpBytes, granularity:Modeling_Granularity, single_chip:Arch):
    """conv (NCHW, implicit GEMM): m = n*h_out*w_out, n = f(out channels), k = kh*kw*c。
    返回 (hete_post_data, smem_fusion_post_data, tiling_config), 与 gemm_wrapper 同构。"""
    m, nn, k = n * h_out * w_out, f, kh * kw * c
    input_bytes, weight_bytes, output_bytes = conv_bytes.get_dtype_bytes()

    tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages, _ = get_default_tiling(m, nn, k, conv_bytes, single_chip)
    mem_levels = conv_bytes.to_mem_levels()

    ret = calculate_conv_implicit_gemm_resource_utilization(
        op_shape=(n, f, h_out, w_out, c, kh, kw),
        tb_shape=(tb_m, tb_n, tb_k), wp_shape=(wp_m, wp_n, wp_k),
        bytes_per_num=weight_bytes, stage_num=int(stages), arch=single_chip, mem_levels=mem_levels)

    hete_post_data = hete_post_process_tensor_core_op(ret, single_chip, weight_bytes)
    smem_fusion_list = [hete_reg_fusion([hete_post_data], single_chip)]
    grids = [m / tb_m, nn / tb_n]
    smem_fusion_post_data = hete_smem_fusion(smem_fusion_list, grids, single_chip)
    log.info("conv implicit-gemm n%s f%s h%s w%s c%s k%sx%s -> M%s N%s K%s, grids %s, time %s s",
             n, f, h_out, w_out, c, kh, kw, m, nn, k, grids, smem_fusion_post_data[0])
    return hete_post_data, smem_fusion_post_data, (tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, stages)
