import runpy

import numpy as np
import pytest

from conftest import ROOT
from mosaic.data.routing import load_routing


evaluate = runpy.run_path(str(ROOT / "examples/custom_accelerator.py"))["evaluate"]


def test_hardware_and_network_parameters_change_the_model_result():
    _, baseline = evaluate()
    _, slower_memory = evaluate(hbm_bandwidth_gbyteps=2680)
    _, slower_network = evaluate(noc_bandwidth_gbyteps=225, noc_latency_ns=2000)
    assert 0 < slower_memory < baseline
    assert 0 < slower_network < baseline


def test_parallelism_keeps_the_eight_gpu_budget():
    latency, stps = evaluate(tp=4, dp=2)
    assert latency > 0
    assert stps == pytest.approx(64 / latency)
    with pytest.raises(ValueError, match=r"TP \* DP"):
        evaluate(tp=3, dp=2)


@pytest.mark.parametrize("model,experts", [("qwen3_235b", 128), ("deepseek_v3", 256)])
def test_packaged_routing_arrays_are_numeric_expert_indices(model, experts):
    for array in load_routing(model):
        assert np.issubdtype(array.dtype, np.integer)
        assert array.shape[-1] == 8
        assert array.min() >= 0 and array.max() < experts
