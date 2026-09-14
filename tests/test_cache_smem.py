"""Numerical checks for corrected cache and shared-memory accounting."""

import numpy as np
import pytest
from scipy.stats import norm




from tilesight.model_version import get_model_version  # noqa: E402
from tilesight.util.sdcm import norm_cdf_approx_3, sdcm  # noqa: E402
from tilesight.fused_op_dtype_wave.matmul_fused_op_new_api_wave import (  # noqa: E402
    calculate_matmul_resource_utilization_new,
)


class _Arch:
    core = "B200"
    l2_capacity = 64 * 1024**2
    sm_count = 148
    ddr_wave_bytes = 1024

    def get_tensor_core_minimum_ptx(self, bytes=2):
        return (64, 8, 16)


def test_corrected_cdf_is_continuous_monotone_and_matches_scipy():
    xs = np.linspace(-6.0, 6.0, 2401)
    values = np.array([norm_cdf_approx_3(x) for x in xs])
    assert np.max(np.abs(values - norm.cdf(xs))) < 2e-5
    assert np.all(np.diff(values) >= 0)
    assert norm_cdf_approx_3(0) == pytest.approx(0.5, abs=1e-6)
    assert abs(norm_cdf_approx_3(1e-9) - norm_cdf_approx_3(-1e-9)) < 1e-6




def test_library_defaults_to_corrected_resource_accounting(monkeypatch):
    monkeypatch.delenv("TILESIGHT_MODEL_VERSION", raising=False)
    assert get_model_version() == "current_corrected"
    assert sdcm(1920, 8, 2048) == pytest.approx(0.5, abs=1e-6)


def test_unknown_resource_model_version_is_rejected(monkeypatch):
    monkeypatch.setenv("TILESIGHT_MODEL_VERSION", "typo")
    with pytest.raises(ValueError, match="TILESIGHT_MODEL_VERSION"):
        get_model_version()


def _smem(in1, in2, output, stages, mma_type="wmma"):
    result = calculate_matmul_resource_utilization_new(
        [64, 64, 64], [32, 32, 32], [16, 16, 32], stages, _Arch(),
        {"in1": in1, "in2": in2, "out1": output}, mma_type=mma_type,
    )
    return result[3]


DDR = [1, 1, 1, 2]
SMEM = [0, 1, 1, 2]
REG = [0, 0, 1, 2]


@pytest.mark.parametrize("stages,expected", [(1, 6144), (3, 14336), (5, 22528)])
def test_pipeline_stages_replicate_inputs_but_not_the_output(stages, expected):
    # Each 32x32 fp16 tile is 2048 bytes: two staged inputs + one output.
    assert _smem(DDR, DDR, SMEM, stages) == expected


@pytest.mark.parametrize("stages", [1, 3, 5])
def test_fused_smem_input_is_resident_once_and_register_input_uses_no_smem(stages):
    assert _smem(SMEM, REG, SMEM, stages) == 4096


def test_cta2_retains_half_of_the_second_input_per_cta():
    # A: 6144 bytes; B: 3072 bytes; output: 2048 bytes.
    assert _smem(DDR, DDR, SMEM, 3, "utcmma_cta2") == 11264
