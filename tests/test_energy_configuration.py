from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from ae.paths import DEEPSTACK_SRC, activate_vendored_sources


activate_vendored_sources()

from mosaic.cost.energy import ChipEnergyConfig, EnergyModel  # noqa: E402
from mosaic.cost.energy_record import EnergyRecord  # noqa: E402
from mosaic.noc.custom_profile import make_custom_profile  # noqa: E402
from mosaic.noc.energy_config import NocEnergyConfig  # noqa: E402
from mosaic.noc.noc_energy import build_extended_energy_matrix  # noqa: E402


def _chip_config(scale: float = 1.0) -> ChipEnergyConfig:
    return ChipEnergyConfig(
        register_read_pj_per_bit=0.17 * scale,
        register_write_pj_per_bit=0.31 * scale,
        shared_memory_read_pj_per_bit=0.43 * scale,
        shared_memory_write_pj_per_bit=0.67 * scale,
        l2_read_pj_per_bit=0.89 * scale,
        l2_write_pj_per_bit=1.23 * scale,
        dram_read_pj_per_bit=2.47 * scale,
        dram_write_pj_per_bit=2.71 * scale,
        sfu_pj_per_op=3.19 * scale,
        cuda_core_pj_per_op=4.37 * scale,
        tensor_core_pj_per_op=5.41 * scale,
        static_power_w=6.73 * scale,
    )


def _layer_configs() -> dict[str, dict[str, object]]:
    return {
        "L3": {
            "kind": "torus2d",
            "shape": (2, 2),
            "hop_latency_ns": 19.25,
            "link_bandwidth_gbytes_per_s": 41.5,
        },
        "L2": {
            "kind": "mesh2d",
            "shape": (2, 2),
            "hop_latency_ns": 7.75,
            "link_bandwidth_gbytes_per_s": 83.25,
        },
        "L1": {
            "kind": "switch",
            "shape": (1, 4),
            "hop_latency_ns": 2.125,
            "link_bandwidth_gbytes_per_s": 167.5,
            "switch_center_in_gbytes_per_s": 341.0,
            "switch_center_out_gbytes_per_s": 347.0,
        },
    }


def test_caller_owned_chip_energy_equations() -> None:
    config = _chip_config()
    model = EnergyModel(SimpleNamespace(chip_energy_config=config))
    read_bytes = 123.5
    write_bytes = 456.25
    operations = 789.25

    assert model.get_reg_energy((read_bytes, write_bytes)) == pytest.approx(
        8
        * (
            read_bytes * config.register_read_pj_per_bit
            + write_bytes * config.register_write_pj_per_bit
        )
    )
    assert model.get_smem_energy((read_bytes, write_bytes)) == pytest.approx(
        8
        * (
            read_bytes * config.shared_memory_read_pj_per_bit
            + write_bytes * config.shared_memory_write_pj_per_bit
        )
    )
    assert model.get_l2_energy((read_bytes, write_bytes)) == pytest.approx(
        8
        * (
            read_bytes * config.l2_read_pj_per_bit
            + write_bytes * config.l2_write_pj_per_bit
        )
    )
    assert model.get_dram_energy((read_bytes, write_bytes)) == pytest.approx(
        8
        * (
            read_bytes * config.dram_read_pj_per_bit
            + write_bytes * config.dram_write_pj_per_bit
        )
    )
    assert model.get_sfu_energy(operations) == pytest.approx(
        operations * config.sfu_pj_per_op
    )
    assert model.get_cuda_core_energy(operations) == pytest.approx(
        operations * config.cuda_core_pj_per_op
    )
    assert model.get_tensor_core_energy(operations) == pytest.approx(
        operations * config.tensor_core_pj_per_op
    )
    assert model.get_static_power() == config.static_power_w


def test_energy_record_accumulates_compute_and_memory_inputs() -> None:
    record = EnergyRecord(4)
    record.add_compute_memory(
        smem=(101.0, 83.0),
        l2=(29.0, 31.0),
        dram=(17.0, 19.0),
        sfu_ops=7.0,
        cuda_ops=11.0,
        tensor_ops=14.0,
    )

    compute_reg_ops = 7.0 + 11.0 + 14.0 / 2
    assert record.reg_rw == pytest.approx(
        (
            101.0 - 29.0 + compute_reg_ops,
            83.0 - 31.0 + compute_reg_ops,
        )
    )
    assert record.smem_rw == pytest.approx((101.0, 83.0))
    assert record.l2_rw == pytest.approx((29.0, 31.0))
    assert record.dram_rw == pytest.approx((17.0, 19.0))
    assert record.sfu_ops == 7.0
    assert record.cuda_ops == 11.0
    assert record.tensor_ops == 14.0


def test_explicit_chip_energy_config_overrides_arch_config() -> None:
    inherited = _chip_config()
    explicit = _chip_config(scale=1.7)
    model = EnergyModel(
        SimpleNamespace(chip_energy_config=inherited),
        chip_energy_config=explicit,
    )

    assert model.get_static_power() == explicit.static_power_w


def test_custom_noc_profile_carries_all_three_energy_levels() -> None:
    config = NocEnergyConfig(
        l1_pj_per_bit=0.73,
        l2_pj_per_bit=1.17,
        l3_pj_per_bit=2.59,
    )
    profile = make_custom_profile(
        layers=_layer_configs(),
        name="synthetic-energy-profile",
        energy_config=config,
    )
    matrix = build_extended_energy_matrix(profile)

    assert profile.energy_config is config
    nonzero_values = set(np.unique(matrix[matrix > 0.0]).tolist())
    assert nonzero_values == {
        config.l1_pj_per_bit,
        config.l2_pj_per_bit,
        config.l3_pj_per_bit,
    }


def test_explicit_noc_energy_config_overrides_profile_config() -> None:
    attached = NocEnergyConfig(0.73, 1.17, 2.59)
    explicit = NocEnergyConfig(3.07, 4.13, 5.29)
    profile = make_custom_profile(
        layers=_layer_configs(),
        energy_config=attached,
    )
    matrix = build_extended_energy_matrix(profile, explicit)

    assert set(np.unique(matrix[matrix > 0.0]).tolist()) == {
        explicit.l1_pj_per_bit,
        explicit.l2_pj_per_bit,
        explicit.l3_pj_per_bit,
    }


@pytest.mark.parametrize(
    "field",
    [item.name for item in fields(ChipEnergyConfig)],
)
@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf")])
def test_chip_energy_config_rejects_invalid_values(
    field: str,
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        replace(_chip_config(), **{field: invalid})


@pytest.mark.parametrize(
    "field",
    ("l1_pj_per_bit", "l2_pj_per_bit", "l3_pj_per_bit"),
)
@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf")])
def test_noc_energy_config_rejects_invalid_values(
    field: str,
    invalid: float,
) -> None:
    config = NocEnergyConfig(0.73, 1.17, 2.59)
    with pytest.raises(ValueError):
        replace(config, **{field: invalid})
    with pytest.raises(TypeError):
        replace(config, **{field: True})


def test_synthetic_energy_path_does_not_load_binary_provider() -> None:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, {str(DEEPSTACK_SRC)!r})

        blocked_providers = {{
            "mosaic.arch._reference_model",
            "mosaic.cost.area",
            "mosaic.cost._capacity",
            "mosaic.noc._model_support",
            "tilesight.distributed.noc._model_support",
        }}

        class BlockProvider(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked_providers:
                    raise ImportError("provider blocked by source-only test")
                return None

        sys.meta_path.insert(0, BlockProvider())

        from mosaic.cost.energy import ChipEnergyConfig, EnergyModel
        from mosaic.cost.op_perf_stats import OpPerfStats
        from mosaic.noc.custom_profile import make_custom_profile
        from mosaic.noc.energy_config import NocEnergyConfig
        from mosaic.noc.noc_energy import build_extended_energy_matrix

        chip = ChipEnergyConfig(
            0.17, 0.31, 0.43, 0.67, 0.89, 1.23,
            2.47, 2.71, 3.19, 4.37, 5.41, 6.73,
        )
        noc = NocEnergyConfig(0.73, 1.17, 2.59)
        model = EnergyModel(
            SimpleNamespace(chip_energy_config=chip),
            noc_energy_config=noc,
        )
        assert model.get_static_power() == chip.static_power_w

        profile = make_custom_profile(
            layers={_layer_configs()!r},
            energy_config=noc,
        )
        matrix = build_extended_energy_matrix(profile)
        assert matrix.max() == noc.l3_pj_per_bit

        arch = SimpleNamespace(
            ddr_bandwidth=9.1e9,
            l2_bandwidth=8.2e9,
            smem_bandwidth=7.3e9,
            fp16_tensor_flops=6.4e9,
            fp32_cuda_core_flops=5.5e9,
            sfu_flops=4.6e9,
            get_tensor_core_minimum_ptx=lambda: (5, 7, 11),
        )
        stats = OpPerfStats(dump_perf_log=False)
        stats.append_hete([
            0.01, 0.2, 0.0, 0.3, 101.0, 0.4,
            17.0, 1000.0, 2000.0, 0.5, 0.6, 0.7,
        ])
        stats.finalize(
            total_time_s=0.02,
            h=profile,
            arch=arch,
            chip_energy_config=chip,
        )
        assert stats.route_stats is not None
        assert stats.chip_total_energy_j > stats.chip_static_energy_j > 0.0
        assert not blocked_providers.intersection(sys.modules)
        """
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
