from __future__ import annotations

import pytest
import torch




from mosaic.arch import B200  # noqa: E402
from mosaic.cost.op_perf_stats import OpPerfStats  # noqa: E402
from mosaic.op_dtype.element_wrapper import get_transformed_tiling  # noqa: E402
from mosaic.utils import OpBytes, Tensor_Loc  # noqa: E402
from tilesight.fused_op_dtype_wave.matmul_fused_op_new_api_wave import (  # noqa: E402
    calculate_matmul_resource_utilization_new,
)


def _fp16_bytes():
    return OpBytes(
        input1=Tensor_Loc(torch.float16, "ddr"),
        input2=Tensor_Loc(torch.float16, "ddr"),
        output=Tensor_Loc(torch.float16, "smem"),
    )


def test_predicated_bs1_padding_is_preserved():
    tile, _ = get_transformed_tiling(
        (1, 1, 2304), _fp16_bytes(), B200(), (2, 128)
    )
    assert tile == [2, 1, 128]


def test_corrected_collective_padding_emits_useful_and_padding_work():
    result = calculate_matmul_resource_utilization_new(
        [2, 2048, 7168],
        [2, 16, 32],
        [1, 4, 32],
        3,
        B200(),
        {
            "in1": [1, 1, 1, 1],
            "in2": [1, 1, 1, 1],
            "out1": [1, 1, 1, 2],
        },
        mma_type="utcmma_cta2",
    )
    assert len(result) == 11
    assert result[9] == 2 * 2 * 2048 * 7168
    assert result[10] > 0


def test_corrected_padding_energy_uses_point_eight_empty_lane_activity():
    row = [
        1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 300.0,
    ]
    stats = OpPerfStats(op_name="padding-energy", dump_perf_log=False)
    stats.append_hete(row)
    stats.finalize(total_time_s=1.0e-6, arch=B200(), num_devices=1)
    assert stats.tensor_useful_ops == 100.0
    assert stats.tensor_padding_ops == 300.0
    assert stats.tensor_energy_equivalent_ops == pytest.approx(340.0)
    assert stats.tensor_padding_energy_activity == 0.8
