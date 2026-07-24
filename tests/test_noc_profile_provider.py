from __future__ import annotations

import copy
import importlib
import importlib.machinery
import multiprocessing as mp
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import pytest

from ae.cli import doctor
from ae.doctor import (
    ENERGY_PROFILE_API_VERSION,
    NOC_PROFILE_API_VERSION,
    PROPRIETARY_BINARY_LICENSE_ID,
    check_noc_profile_provider,
    check_tilesight_noc_profile_provider,
)
from ae.paths import DEEPSTACK_SRC, activate_vendored_sources


activate_vendored_sources()
_model_support = importlib.import_module("mosaic.noc._model_support")
_noc_config_set = importlib.import_module("mosaic.noc.noc_config_set")
_custom_profile = importlib.import_module("mosaic.noc.custom_profile")


PROFILE_NAMES = (
    "torus_mesh_switch_1",
    "torus_mesh_mesh_3",
    "torus_mesh_switch_cluster8",
    "torus_mesh_switch_cluster16",
    "torus_mesh_switch_cluster32",
)
LAYER_NAMES = ("L3", "L2", "L1")
TOPOLOGY_DISPATCH_CASES = (
    ("switch", (1, 4)),
    ("all_to_all", (2, 2)),
    ("ring", (1, 4)),
    ("chain", (1, 4)),
    ("mesh2d", (2, 4)),
    ("torus2d", (2, 4)),
    ("mchain_nring", (2, 4)),
    ("mring_nchain", (2, 4)),
)


def _toy_layer_configs() -> dict[str, dict[str, object]]:
    """Return deliberately out-of-order, fully synthetic layer inputs."""

    return {
        "L1": {
            "shape": [1, 4],
            "kind": "switch",
            "link_bandwidth_gbytes_per_s": 172.75,
            "hop_latency_ns": 3.125,
            "switch_center_out_gbytes_per_s": 351.25,
            "switch_center_in_gbytes_per_s": 347.5,
        },
        "L3": {
            "link_bandwidth_gbytes_per_s": 43.5,
            "hop_latency_ns": 17.25,
            "shape": (2, 2),
            "kind": "torus2d",
        },
        "L2": {
            "hop_latency_ns": 8.75,
            "kind": "mchain_nring",
            "shape": [2, 4],
            "link_bandwidth_gbytes_per_s": 86.25,
        },
    }


def _construct_profile_in_worker() -> tuple[int, str, int, int]:
    from mosaic.noc import noc_config_set as provider

    profile = provider.torus_mesh_switch_1()
    return (
        provider.model_support_api_version(),
        str(profile),
        len(profile.layers),
        int(profile.num_devices),
    )


def _construct_custom_profile_in_worker(
    layer_latencies_ns: dict[str, float],
) -> tuple[str, tuple[float, ...]]:
    from mosaic.noc.noc_config_set import make_profile_with_latencies

    profile = make_profile_with_latencies(
        "torus_mesh_switch_1",
        layer_latencies_ns=layer_latencies_ns,
        output_name="spawn_custom_profile",
    )
    return (
        str(profile),
        tuple(layer.hop_latency * 1.0e9 for layer in profile.layers),
    )


def _construct_mapping_profile_in_worker(
    layer_configs: dict[str, dict[str, object]],
) -> tuple[str, str, tuple[tuple[str, tuple[int, int], float, float], ...]]:
    from mosaic.noc.custom_profile import make_custom_profile

    profile = make_custom_profile(
        layers=layer_configs,
        name="spawn_mapping_profile",
        port_spread="nearest",
    )
    return (
        str(profile),
        profile.port_spread.value,
        tuple(
            (
                layer.kind.value,
                layer.shape,
                layer.hop_latency * 1.0e9,
                layer.link_bandwidth / 1.0e9,
            )
            for layer in profile.layers
        ),
    )


def test_doctor_accepts_bundled_binary_provider(
    capsys: pytest.CaptureFixture[str],
) -> None:
    failures: list[str] = []
    path = check_noc_profile_provider(failures)
    tilesight_path = check_tilesight_noc_profile_provider(failures)
    assert failures == []
    assert path is not None
    assert tilesight_path is not None
    assert any(
        path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    assert any(
        tilesight_path.name.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )

    assert doctor() == 0
    output = capsys.readouterr()
    assert "noc_energy_profile_provider=binary" in output.out
    assert "tilesight_noc_profile_provider=binary" in output.out
    assert "noc_api=1" in output.out
    assert "energy_api=1" in output.out
    assert (
        f"binary_license={PROPRIETARY_BINARY_LICENSE_ID} "
        "(4/4 sidecars)"
    ) in output.out
    assert "hop_latency" not in output.out
    assert "link_bandwidth" not in output.out
    assert output.err == ""


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_binary_provider_constructs_named_profiles(profile_name: str) -> None:
    factory = getattr(_noc_config_set, profile_name)
    profile = factory()
    assert str(profile) == profile_name
    assert len(profile.layers) == 3
    assert profile.num_devices > 0
    assert (
        _noc_config_set.model_support_api_version()
        == NOC_PROFILE_API_VERSION
    )


def test_binary_provider_exposes_energy_calculation_interfaces() -> None:
    assert _model_support.p97() == ENERGY_PROFILE_API_VERSION
    for name in ("p93", "p94", "p95", "p96"):
        assert callable(getattr(_model_support, name))
    for removed_name in (
        "_ReferenceChipEnergyModel",
        "_reference_noc_energy_for_layer",
        "energy_support_api_version",
    ):
        assert not hasattr(_model_support, removed_name)


@pytest.mark.parametrize(
    "module_name",
    (
        "mosaic.noc.noc_config_set",
        "tilesight.distributed.noc.noc_config_set",
    ),
)
def test_reference_profile_repr_is_structural_only(module_name: str) -> None:
    provider = importlib.import_module(module_name)
    profile = provider.torus_mesh_switch_1()
    rendered = repr(profile)

    assert "torus2d" in rendered
    assert "mesh2d" in rendered
    assert "switch" in rendered
    for parameter_name in (
        "hop_latency",
        "link_bandwidth",
        "switch_center_in_bw",
        "switch_center_out_bw",
    ):
        assert parameter_name not in rendered


def test_dimensionless_profile_scaling_is_relative() -> None:
    latency_scale = 2.0
    bandwidth_scale = 1.5
    baseline = _noc_config_set.make_scaled_profile("torus_mesh_switch_1")
    scaled = _noc_config_set.make_scaled_profile(
        "torus_mesh_switch_1",
        latency_scale=latency_scale,
        l3_bandwidth_scale=1.0,
        l2_bandwidth_scale=bandwidth_scale,
        l1_bandwidth_scale=1.0,
    )

    for actual, reference in zip(scaled.layers, baseline.layers):
        assert actual.hop_latency / reference.hop_latency == pytest.approx(
            latency_scale
        )
    ratios = [
        actual.link_bandwidth / reference.link_bandwidth
        for actual, reference in zip(scaled.layers, baseline.layers)
    ]
    assert ratios == pytest.approx([1.0, bandwidth_scale, 1.0])


def test_custom_profile_module_is_source_only_and_compatibly_reexported() -> None:
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(DEEPSTACK_SRC)!r}); "
        "import mosaic.noc.custom_profile; "
        "assert 'mosaic.noc._model_support' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
    assert _noc_config_set.make_topology is _custom_profile.make_topology
    assert _noc_config_set.make_custom_profile is _custom_profile.make_custom_profile


def test_explicit_layer_latencies_are_absolute_and_do_not_mutate_input() -> None:
    generator = random.Random("deepstack-custom-profile")
    layer_latencies_ns = {
        "L3": generator.uniform(1.0, 20.0),
        "L2": generator.uniform(1.0, 20.0),
        "L1": 0.0,
    }
    original = layer_latencies_ns.copy()
    baseline = _noc_config_set.torus_mesh_switch_1()

    profile = _noc_config_set.make_profile_with_latencies(
        "torus_mesh_switch_1",
        layer_latencies_ns=layer_latencies_ns,
        output_name="custom_latency_profile",
    )

    assert layer_latencies_ns == original
    assert str(profile) == "custom_latency_profile"
    assert len(profile.layers) == len(LAYER_NAMES)
    actual_ns = [layer.hop_latency * 1.0e9 for layer in profile.layers]
    assert actual_ns == pytest.approx(
        [layer_latencies_ns[name] for name in LAYER_NAMES]
    )
    for actual, reference in zip(profile.layers, baseline.layers):
        assert actual.kind == reference.kind
        assert actual.shape == reference.shape
        assert actual.link_bandwidth == reference.link_bandwidth


@pytest.mark.parametrize(
    "layer_latencies_ns",
    [
        {"L3": 1.0, "L2": 2.0},
        {"L3": 1.0, "L2": 2.0, "L1": 3.0, "extra": 4.0},
        {"L3": float("nan"), "L2": 2.0, "L1": 3.0},
        {"L3": float("inf"), "L2": 2.0, "L1": 3.0},
        {"L3": 1.0, "L2": -2.0, "L1": 3.0},
    ],
)
def test_explicit_layer_latencies_reject_invalid_input(
    layer_latencies_ns: dict[str, float],
) -> None:
    original = layer_latencies_ns.copy()
    with pytest.raises((TypeError, ValueError)):
        _noc_config_set.make_profile_with_latencies(
            "torus_mesh_switch_1",
            layer_latencies_ns=layer_latencies_ns,
        )
    assert layer_latencies_ns == original


@pytest.mark.parametrize(("kind", "shape"), TOPOLOGY_DISPATCH_CASES)
def test_make_topology_dispatches_every_kind(
    kind: str,
    shape: tuple[int, int],
) -> None:
    latency_ns = 6.625
    bandwidth_gbytes_per_s = 29.75
    center = (
        {
            "switch_center_in_gbytes_per_s": 61.5,
            "switch_center_out_gbytes_per_s": 63.25,
        }
        if kind == "switch"
        else {}
    )
    topology = _custom_profile.make_topology(
        kind,
        shape,
        hop_latency_ns=latency_ns,
        link_bandwidth_gbytes_per_s=bandwidth_gbytes_per_s,
        **center,
    )

    assert topology.kind.value == kind
    assert topology.shape == shape
    assert topology.hop_latency == pytest.approx(latency_ns * 1.0e-9)
    assert topology.link_bandwidth == pytest.approx(
        bandwidth_gbytes_per_s * 1.0e9
    )
    last_node = shape[0] * shape[1] - 1
    assert topology.route_intra(0, last_node)
    for side in (
        "left_in",
        "left_out",
        "right_in",
        "right_out",
        "up_in",
        "up_out",
        "down_in",
        "down_out",
    ):
        assert topology.ports(side)
    if kind == "switch":
        assert topology.switch_center_in_bw == pytest.approx(
            center["switch_center_in_gbytes_per_s"] * 1.0e9
        )
        assert topology.switch_center_out_bw == pytest.approx(
            center["switch_center_out_gbytes_per_s"] * 1.0e9
        )
    else:
        assert topology.switch_center_in_bw is None
        assert topology.switch_center_out_bw is None


def test_make_custom_profile_preserves_contract_and_input() -> None:
    layer_configs = _toy_layer_configs()
    original = copy.deepcopy(layer_configs)
    profile = _custom_profile.make_custom_profile(
        layers=layer_configs,
        name="synthetic_three_level_profile",
        port_spread="nearest",
    )

    assert layer_configs == original
    assert tuple(layer_configs) != LAYER_NAMES
    assert str(profile) == "synthetic_three_level_profile"
    assert profile.port_spread.value == "nearest"
    assert [layer.kind.value for layer in profile.layers] == [
        str(layer_configs[name]["kind"]) for name in LAYER_NAMES
    ]
    assert [layer.shape for layer in profile.layers] == [
        tuple(layer_configs[name]["shape"]) for name in LAYER_NAMES
    ]
    assert [layer.hop_latency for layer in profile.layers] == pytest.approx(
        [
            float(layer_configs[name]["hop_latency_ns"]) * 1.0e-9
            for name in LAYER_NAMES
        ]
    )
    assert [layer.link_bandwidth for layer in profile.layers] == pytest.approx(
        [
            float(layer_configs[name]["link_bandwidth_gbytes_per_s"]) * 1.0e9
            for name in LAYER_NAMES
        ]
    )
    assert profile.layers[-1].switch_center_in_bw == pytest.approx(
        float(layer_configs["L1"]["switch_center_in_gbytes_per_s"]) * 1.0e9
    )
    assert profile.layers[-1].switch_center_out_bw == pytest.approx(
        float(layer_configs["L1"]["switch_center_out_gbytes_per_s"]) * 1.0e9
    )
    rendered = repr(profile)
    assert "hop_latency=" in rendered
    assert "link_bandwidth=" in rendered


@pytest.mark.parametrize(
    ("kind", "shape"),
    [
        ("switch", (2, 2)),
        ("ring", (2, 4)),
        ("ring", (1, 3)),
        ("chain", (1, 3)),
        ("all_to_all", (1, 3)),
        ("mesh2d", (3, 2)),
        ("torus2d", (2, 3)),
        ("mchain_nring", (3, 4)),
        ("mring_nchain", (4, 3)),
    ],
)
def test_make_topology_rejects_kind_specific_invalid_shapes(
    kind: str,
    shape: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        _custom_profile.make_topology(
            kind,
            shape,
            hop_latency_ns=2.5,
            link_bandwidth_gbytes_per_s=11.5,
        )


@pytest.mark.parametrize(
    "shape",
    ["1x4", (1,), (1, 2, 4), (True, 4), (1.5, 4), (0, 4), (1, -4)],
)
def test_make_topology_rejects_malformed_shape(shape: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _custom_profile.make_topology(
            "switch",
            shape,
            hop_latency_ns=2.5,
            link_bandwidth_gbytes_per_s=11.5,
        )


@pytest.mark.parametrize("latency_ns", [float("nan"), float("inf"), -0.25])
def test_make_topology_rejects_invalid_latency(latency_ns: float) -> None:
    with pytest.raises(ValueError):
        _custom_profile.make_topology(
            "mesh2d",
            (2, 2),
            hop_latency_ns=latency_ns,
            link_bandwidth_gbytes_per_s=11.5,
        )


@pytest.mark.parametrize(
    "bandwidth_gbytes_per_s",
    [0.0, -0.5, float("nan"), float("inf")],
)
def test_make_topology_rejects_invalid_bandwidth(
    bandwidth_gbytes_per_s: float,
) -> None:
    with pytest.raises(ValueError):
        _custom_profile.make_topology(
            "mesh2d",
            (2, 2),
            hop_latency_ns=2.5,
            link_bandwidth_gbytes_per_s=bandwidth_gbytes_per_s,
        )


def test_switch_center_bandwidth_contract_is_strict() -> None:
    common = {
        "hop_latency_ns": 2.5,
        "link_bandwidth_gbytes_per_s": 11.5,
    }
    with pytest.raises(ValueError):
        _custom_profile.make_topology(
            "mesh2d",
            (2, 2),
            switch_center_in_gbytes_per_s=23.0,
            switch_center_out_gbytes_per_s=24.0,
            **common,
        )
    for one_sided in (
        {"switch_center_in_gbytes_per_s": 23.0},
        {"switch_center_out_gbytes_per_s": 24.0},
    ):
        with pytest.raises(ValueError):
            _custom_profile.make_topology(
                "switch",
                (1, 4),
                **one_sided,
                **common,
            )
    with pytest.raises(ValueError):
        _custom_profile.make_topology(
            "switch",
            (1, 4),
            switch_center_in_gbytes_per_s=0.0,
            switch_center_out_gbytes_per_s=24.0,
            **common,
        )


def test_make_custom_profile_rejects_layer_set_mismatch() -> None:
    layer_configs = _toy_layer_configs()
    missing = copy.deepcopy(layer_configs)
    missing.pop("L2")
    extra = copy.deepcopy(layer_configs)
    extra["L0"] = copy.deepcopy(extra["L1"])

    for invalid in (missing, extra):
        original = copy.deepcopy(invalid)
        with pytest.raises(ValueError):
            _custom_profile.make_custom_profile(layers=invalid)
        assert invalid == original


def test_make_custom_profile_rejects_config_field_mismatch() -> None:
    missing = _toy_layer_configs()
    missing["L3"].pop("hop_latency_ns")
    extra = _toy_layer_configs()
    extra["L2"]["unexpected_field"] = "synthetic"

    for invalid in (missing, extra):
        original = copy.deepcopy(invalid)
        with pytest.raises(ValueError):
            _custom_profile.make_custom_profile(layers=invalid)
        assert invalid == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency_scale": 0.0},
        {"latency_scale": -1.0},
        {"l1_bandwidth_scale": 0.0},
        {"l2_bandwidth_scale": float("nan")},
    ],
)
def test_binary_provider_rejects_invalid_scales(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _noc_config_set.make_scaled_profile("torus_mesh_switch_1", **kwargs)


def test_binary_provider_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError):
        _noc_config_set.make_scaled_profile("not_a_profile")


def test_binary_provider_process_pool_smoke() -> None:
    parent = _construct_profile_in_worker()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        child = executor.submit(_construct_profile_in_worker).result(timeout=30)
    assert child == parent


def test_explicit_layer_latencies_spawn_smoke() -> None:
    layer_latencies_ns = {"L3": 8.25, "L2": 4.5, "L1": 1.75}
    original = layer_latencies_ns.copy()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        name, actual_ns = executor.submit(
            _construct_custom_profile_in_worker,
            layer_latencies_ns,
        ).result(timeout=30)
    assert layer_latencies_ns == original
    assert name == "spawn_custom_profile"
    assert actual_ns == pytest.approx(
        tuple(layer_latencies_ns[name] for name in LAYER_NAMES)
    )


def test_custom_mapping_profile_spawn_smoke() -> None:
    layer_configs = _toy_layer_configs()
    original = copy.deepcopy(layer_configs)
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        name, port_spread, summary = executor.submit(
            _construct_mapping_profile_in_worker,
            layer_configs,
        ).result(timeout=30)

    assert layer_configs == original
    assert name == "spawn_mapping_profile"
    assert port_spread == "nearest"
    assert [entry[0] for entry in summary] == [
        str(layer_configs[name]["kind"]) for name in LAYER_NAMES
    ]
    assert [entry[1] for entry in summary] == [
        tuple(layer_configs[name]["shape"]) for name in LAYER_NAMES
    ]
    assert [entry[2] for entry in summary] == pytest.approx(
        [float(layer_configs[name]["hop_latency_ns"]) for name in LAYER_NAMES]
    )
    assert [entry[3] for entry in summary] == pytest.approx(
        [
            float(layer_configs[name]["link_bandwidth_gbytes_per_s"])
            for name in LAYER_NAMES
        ]
    )
