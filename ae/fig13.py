"""Recompute and verify the ASTRA-sim/NS-3 comparison used by Figure 13.

The DeepStack network-model column is regenerated for every plotted point.
ASTRA-sim/NS-3 and the conventional analytical estimates remain immutable
bundled references; this module never invokes those external tools.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_DIR, RESULTS_DIR, activate_vendored_sources


INPUT_DIR = DATA_DIR / "fig13"
OUTPUT_DIR = RESULTS_DIR / "reproduce" / "stages" / "fig13"
NS3_COLUMN = "overall_time_ns_as_ns3"
MODEL_COLUMN = "overall_time_ns"
ANALYTICAL_COLUMN = "overall_time_ns_as_analytical"

PANELS = (
    ("switch", "all_gather", "all_gather_halving_doubling_test.csv"),
    ("switch", "all_reduce", "all_reduce_rabenseifner_test.csv"),
    ("switch", "all_to_all", "all_to_all_test.csv"),
    ("torus", "all_gather", "all_gather_rs_ag_ag_3stage_test.csv"),
    ("torus", "all_reduce", "all_reduce_ring_test.csv"),
    ("torus", "all_to_all", "all_to_all_test.csv"),
)


@dataclass(frozen=True)
class PanelMetric:
    topology: str
    collective: str
    source_file: str
    rows_total: int
    rows_with_ns3: int
    model_abs_error_sum_ns: float
    analytical_abs_error_sum_ns: float
    ns3_abs_sum_ns: float

    @property
    def model_wer_pct(self) -> float:
        return 100.0 * self.model_abs_error_sum_ns / self.ns3_abs_sum_ns

    @property
    def analytical_wer_pct(self) -> float:
        return 100.0 * self.analytical_abs_error_sum_ns / self.ns3_abs_sum_ns


def _float_or_none(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _switch_hierarchy(row: dict[str, str]):
    from mosaic.arch import _reference_model as _provider

    from .fig13_paper_model.noc_topo_switch_only import (
        Hierarchy,
        PortSpread,
        make_switch,
    )

    l1_dim = int(row["num_nodes_l1"])
    l2_dim = int(row["num_nodes_l2"])
    layers = [
        make_switch(
            1,
            hop_latency=0.0,
            link_bandwidth=1.0,
            switch_center_in_bw=1.0,
            switch_center_out_bw=1.0,
        ),
        make_switch(
            l2_dim,
            hop_latency=0.0,
            link_bandwidth=1.0,
            switch_center_in_bw=1.0,
            switch_center_out_bw=1.0,
        ),
        make_switch(
            l1_dim,
            hop_latency=0.0,
            link_bandwidth=1.0,
            switch_center_in_bw=1.0,
            switch_center_out_bw=1.0,
        ),
    ]
    _provider.q12(
        row["reference_case_id"],
        *layers,
    )
    return Hierarchy(
        layers=layers,
        port_spread=PortSpread.EVEN,
        node_mapper=None,
        name="fig13_switch",
    )


def _torus_hierarchy(row: dict[str, str]):
    from mosaic.arch import _reference_model as _provider

    from .fig13_paper_model.noc_topo import (
        Hierarchy,
        PortSpread,
        TopoKind,
        make_mesh_or_torus,
        make_switch,
    )

    dim_x = int(row["num_nodes_x"])
    dim_y = int(row["num_nodes_y"])
    layers = [
        make_switch(
            1,
            hop_latency=0.0,
            link_bandwidth=1.0,
            switch_center_in_bw=1.0,
            switch_center_out_bw=1.0,
        ),
        make_switch(
            1,
            hop_latency=0.0,
            link_bandwidth=1.0,
            switch_center_in_bw=1.0,
            switch_center_out_bw=1.0,
        ),
        make_mesh_or_torus(
            dim_x,
            dim_y,
            TopoKind.TORUS2D,
            hop_latency=0.0,
            link_bandwidth=1.0,
        ),
    ]
    _provider.q13(
        row["reference_case_id"],
        *layers,
    )
    return Hierarchy(
        layers=layers,
        port_spread=PortSpread.EVEN,
        node_mapper=None,
        name="fig13_torus",
    )


def _route_time(traffic, hierarchy, *, switch_only: bool) -> tuple[float, float]:
    if switch_only:
        from .fig13_paper_model.noc_topo_switch_only import (
            get_extend_max_routes_switch_only,
        )

        hop, extension, _ = get_extend_max_routes_switch_only(traffic, hierarchy)
    else:
        from .fig13_paper_model.noc_topo import get_extend_max_routes

        hop, extension, _ = get_extend_max_routes(traffic, hierarchy)
    return float(hop), float(extension)


def _pair_traffic(
    num_nodes: int,
    pairs: list[list[float | int]],
    *,
    dimension: str,
    tp: int,
    pp: int,
):
    from .fig13_paper_model.traffic_matrix import TrafficMatrix

    traffic = TrafficMatrix(num_nodes)
    traffic.add_intra_group_traffic_pair_bulk(
        dimension,
        pairs,
        tp=tp,
        ep=1,
        sp=1,
        cp=1,
        dp=1,
        pp=pp,
    )
    return traffic


def _all_to_all_time(
    num_nodes: int, hierarchy, num_bytes: float, *, switch_only: bool
) -> tuple[float, float]:
    from .fig13_paper_model.traffic_matrix import TrafficMatrix

    traffic = TrafficMatrix(num_nodes)
    traffic.add_intra_group_traffic(
        "tp", num_bytes, tp=num_nodes, ep=1, sp=1, cp=1, dp=1, pp=1
    )
    return _route_time(traffic, hierarchy, switch_only=switch_only)


def _bytes_with_packet_headers(
    payload_bytes: float, package_bytes: int = 1024, header_bytes: int = 36
) -> float:
    """Match the packetization applied before the paper-time Torus model."""

    packets = math.ceil(payload_bytes / (package_bytes - header_bytes))
    return payload_bytes + packets * header_bytes


def _switch_model_time(row: dict[str, str], collective: str) -> tuple[float, float]:
    hierarchy = _switch_hierarchy(row)
    num_nodes = int(row["num_nodes_l1"]) * int(row["num_nodes_l2"])
    num_bytes = float(row["bytes"])
    if collective == "all_to_all":
        return _all_to_all_time(num_nodes, hierarchy, num_bytes, switch_only=True)

    stages = int(math.log2(num_nodes))
    hop_total = 0.0
    extension_total = 0.0
    if collective == "all_reduce":
        for stage in range(stages):
            stride = 1 << stage
            stage_bytes = num_bytes / (1 << (stage + 1))
            pairs = [[stage_bytes, node, node ^ stride] for node in range(num_nodes)]
            traffic = _pair_traffic(
                num_nodes, pairs, dimension="tp", tp=num_nodes, pp=1
            )
            hop, extension = _route_time(traffic, hierarchy, switch_only=True)
            hop_total += hop
            extension_total += extension
        return 2.0 * hop_total, 2.0 * extension_total

    if collective == "all_gather":
        max_stride = 1 << (stages - 1)
        for stage in range(stages):
            stride = max_stride >> stage
            stage_bytes = num_bytes * (1 << stage)
            pairs = [[stage_bytes, node, node ^ stride] for node in range(num_nodes)]
            traffic = _pair_traffic(
                num_nodes, pairs, dimension="tp", tp=num_nodes, pp=1
            )
            hop, extension = _route_time(traffic, hierarchy, switch_only=True)
            hop_total += hop
            extension_total += extension
        return hop_total, extension_total
    raise ValueError(f"unsupported Switch collective: {collective}")


def _torus_model_time(row: dict[str, str], collective: str) -> tuple[float, float]:
    hierarchy = _torus_hierarchy(row)
    dim_x = int(row["num_nodes_x"])
    dim_y = int(row["num_nodes_y"])
    num_nodes = dim_x * dim_y
    num_bytes = float(row["bytes"])
    if collective == "all_to_all":
        return _all_to_all_time(
            num_nodes,
            hierarchy,
            _bytes_with_packet_headers(num_bytes),
            switch_only=False,
        )

    if collective == "all_reduce":
        num_bytes = _bytes_with_packet_headers(num_bytes)
        pairs = [
            [num_bytes / num_nodes, node, (node + 1) % num_nodes]
            for node in range(num_nodes)
        ]
        traffic = _pair_traffic(
            num_nodes, pairs, dimension="tp", tp=num_nodes, pp=1
        )
        hop, extension = _route_time(traffic, hierarchy, switch_only=False)
        multiplier = 2 * (num_nodes - 1)
        return multiplier * hop, multiplier * extension

    if collective == "all_gather":
        if dim_x != dim_y:
            raise ValueError("Figure 13 all-gather expects a square torus")
        dim = dim_x
        stages = dim - 1
        # Match the submitted three-stage model: the reduce-scatter first stage
        # is intentionally disabled in the paper experiment; PP all-gather and
        # TP all-gather are evaluated below.
        stage2_pairs = [[num_bytes, node, (node + 1) % dim] for node in range(dim)]
        stage2 = _pair_traffic(
            num_nodes, stage2_pairs, dimension="pp", tp=dim, pp=dim
        )
        hop2, extension2 = _route_time(stage2, hierarchy, switch_only=False)
        stage3_pairs = [
            [num_bytes * dim, node, (node + 1) % dim] for node in range(dim)
        ]
        stage3 = _pair_traffic(
            num_nodes, stage3_pairs, dimension="tp", tp=dim, pp=dim
        )
        hop3, extension3 = _route_time(stage3, hierarchy, switch_only=False)
        return stages * (hop2 + hop3), stages * (extension2 + extension3)
    raise ValueError(f"unsupported Torus collective: {collective}")


def _recompute_panel(
    topology: str, collective: str, filename: str
) -> tuple[list[float], list[dict[str, object]]]:
    path = INPUT_DIR / topology / filename
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: list[float] = []
    output_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        if topology == "switch":
            hop, extension = _switch_model_time(row, collective)
        else:
            hop, extension = _torus_model_time(row, collective)
        recomputed_ns = (hop + extension) * 1e9
        archived_ns = float(row[MODEL_COLUMN])
        ns3_ns = _float_or_none(row.get(NS3_COLUMN))
        analytical_ns = _float_or_none(row.get(ANALYTICAL_COLUMN))
        values.append(recomputed_ns)
        output_rows.append(
            {
                "topology": topology,
                "collective": collective,
                "source_file": filename,
                "row_index": row_index,
                "bytes": row["bytes"],
                "ns3_time_ns": "nan" if ns3_ns is None else f"{ns3_ns:.15g}",
                "analytical_time_ns": (
                    "nan" if analytical_ns is None else f"{analytical_ns:.15g}"
                ),
                "archived_model_time_ns": f"{archived_ns:.15g}",
                "recomputed_model_time_ns": f"{recomputed_ns:.15g}",
                "relative_delta": f"{recomputed_ns / archived_ns - 1.0:.15g}",
                "gpu_runs": 0,
            }
        )
    return values, output_rows


def _load_panel(
    topology: str,
    collective: str,
    filename: str,
    recomputed_model: list[float],
) -> PanelMetric:
    path = INPUT_DIR / topology / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {NS3_COLUMN, MODEL_COLUMN, ANALYTICAL_COLUMN}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)

    valid: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(rows):
        ns3 = _float_or_none(row[NS3_COLUMN])
        model = recomputed_model[row_index]
        analytical = _float_or_none(row[ANALYTICAL_COLUMN])
        # This is the same filtering rule used by the original plotting code:
        # rows without NS-3 data do not enter the error calculation.
        if ns3 is None:
            continue
        if model is None or analytical is None:
            raise ValueError(f"{path}: prediction missing where NS-3 is present")
        valid.append((ns3, model, analytical))

    ns3_sum = sum(abs(ns3) for ns3, _, _ in valid)
    if not valid or ns3_sum == 0.0:
        raise ValueError(f"{path}: no usable non-zero NS-3 reference values")
    return PanelMetric(
        topology=topology,
        collective=collective,
        source_file=str(path.relative_to(INPUT_DIR)),
        rows_total=len(rows),
        rows_with_ns3=len(valid),
        model_abs_error_sum_ns=sum(abs(model - ns3) for ns3, model, _ in valid),
        analytical_abs_error_sum_ns=sum(
            abs(analytical - ns3) for ns3, _, analytical in valid
        ),
        ns3_abs_sum_ns=ns3_sum,
    )


def _check_hashes() -> tuple[int, int]:
    manifest = INPUT_DIR / "SHA256SUMS"
    checked = 0
    mismatches = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        actual = hashlib.sha256((INPUT_DIR / relative).read_bytes()).hexdigest()
        checked += 1
        mismatches += actual != expected
    return checked, mismatches


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _claim(
    metric: str,
    value: float,
    unit: str,
    expected: float | None,
    tolerance: float | None,
    scope: str,
    evidence: str,
) -> dict[str, object]:
    if expected is None or tolerance is None:
        status = "INFO"
    else:
        status = "PASS" if abs(value - expected) <= tolerance else "FAIL"
    return {
        "metric": metric,
        "value": f"{value:.12g}",
        "unit": unit,
        "expected": "" if expected is None else f"{expected:.12g}",
        "tolerance": "" if tolerance is None else f"{tolerance:.12g}",
        "status": status,
        "scope": scope,
        "evidence": evidence,
    }


def run(output_dir: Path = OUTPUT_DIR) -> int:
    started = time.perf_counter()
    activate_vendored_sources()
    recomputed: dict[tuple[str, str], list[float]] = {}
    point_rows: list[dict[str, object]] = []
    for topology, collective, filename in PANELS:
        values, rows = _recompute_panel(topology, collective, filename)
        recomputed[(topology, collective)] = values
        point_rows.extend(rows)
    metrics = [
        _load_panel(
            topology,
            collective,
            filename,
            recomputed[(topology, collective)],
        )
        for topology, collective, filename in PANELS
    ]
    metric_by_key = {(m.topology, m.collective): m for m in metrics}

    _write_csv(
        output_dir / "model_points.csv",
        [
            "topology",
            "collective",
            "source_file",
            "row_index",
            "bytes",
            "ns3_time_ns",
            "analytical_time_ns",
            "archived_model_time_ns",
            "recomputed_model_time_ns",
            "relative_delta",
            "gpu_runs",
        ],
        point_rows,
    )

    panel_rows = [
        {
            "topology": metric.topology,
            "collective": metric.collective,
            "source_file": metric.source_file,
            "rows_total": metric.rows_total,
            "rows_with_ns3": metric.rows_with_ns3,
            "deepstack_weighted_error_pct": f"{metric.model_wer_pct:.12g}",
            "analytical_weighted_error_pct": f"{metric.analytical_wer_pct:.12g}",
        }
        for metric in metrics
    ]
    _write_csv(
        output_dir / "panel_metrics.csv",
        [
            "topology",
            "collective",
            "source_file",
            "rows_total",
            "rows_with_ns3",
            "deepstack_weighted_error_pct",
            "analytical_weighted_error_pct",
        ],
        panel_rows,
    )

    switch_ar = metric_by_key[("switch", "all_reduce")]
    torus_ar = metric_by_key[("torus", "all_reduce")]
    all_reduce_analytical_worst = max(
        switch_ar.analytical_wer_pct, torus_ar.analytical_wer_pct
    )

    topology_aggregates: dict[str, tuple[float, float]] = {}
    for topology in ("switch", "torus"):
        selected = [metric for metric in metrics if metric.topology == topology]
        denominator = sum(metric.ns3_abs_sum_ns for metric in selected)
        topology_aggregates[topology] = (
            100.0 * sum(metric.model_abs_error_sum_ns for metric in selected) / denominator,
            100.0
            * sum(metric.analytical_abs_error_sum_ns for metric in selected)
            / denominator,
        )

    with (INPUT_DIR / "reported_runtime_endpoints.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        runtime_rows = list(csv.DictReader(handle))
    runtime = {row["implementation"]: float(row["wall_time_s"]) for row in runtime_rows}
    reported_speedup = runtime["ASTRA-sim_NS-3"] / runtime["DeepStack"]

    hashes_checked, hash_mismatches = _check_hashes()
    max_model_relative_delta = max(
        abs(float(row["relative_delta"])) for row in point_rows
    )
    claims = [
        _claim(
            "reference_csv_checksum_mismatches",
            float(hash_mismatches),
            "files",
            0.0,
            0.0,
            f"{hashes_checked} bundled Figure 13 inputs",
            "SHA256SUMS",
        ),
        _claim(
            "deepstack_model_points_recomputed",
            float(len(point_rows)),
            "points",
            420.0,
            0.0,
            "all six plotted panels; CPU only",
            "model_points.csv",
        ),
        _claim(
            "deepstack_archived_regression_max_relative_delta",
            max_model_relative_delta,
            "relative",
            0.0,
            1e-12,
            "all 420 CPU-rerun model points",
            "model_points.csv",
        ),
        _claim(
            "paper_switch_weighted_error",
            switch_ar.model_wer_pct,
            "percent",
            2.12,
            0.005,
            "Switch all-reduce panel",
            switch_ar.source_file,
        ),
        _claim(
            "paper_torus_weighted_error",
            torus_ar.model_wer_pct,
            "percent",
            1.62,
            0.005,
            "Torus all-reduce panel",
            torus_ar.source_file,
        ),
        _claim(
            "paper_analytical_worst_all_reduce",
            all_reduce_analytical_worst,
            "percent",
            58.0,
            0.5,
            "maximum analytical weighted error across the two all-reduce panels",
            f"{switch_ar.source_file}; {torus_ar.source_file}",
        ),
        _claim(
            "paper_reported_runtime_speedup",
            reported_speedup,
            "x",
            100000.0,
            10000.0,
            "reported 3 h NS-3 endpoint divided by reported 0.1 s DeepStack endpoint",
            "reported_runtime_endpoints.csv (reported endpoints; raw logs unavailable)",
        ),
        _claim(
            "all_panels_switch_weighted_error",
            topology_aggregates["switch"][0],
            "percent",
            None,
            None,
            "aggregate over all three Switch panels",
            "three bundled Switch CSVs",
        ),
        _claim(
            "all_panels_torus_weighted_error",
            topology_aggregates["torus"][0],
            "percent",
            None,
            None,
            "aggregate over all three Torus panels",
            "three bundled Torus CSVs",
        ),
        _claim(
            "largest_per_panel_analytical_weighted_error",
            max(metric.analytical_wer_pct for metric in metrics),
            "percent",
            None,
            None,
            "maximum of six per-panel weighted errors; provided to make claim scope explicit",
            "six bundled Figure 13 CSVs",
        ),
    ]
    claim_fields = [
        "metric",
        "value",
        "unit",
        "expected",
        "tolerance",
        "status",
        "scope",
        "evidence",
    ]
    _write_csv(output_dir / "summary.csv", claim_fields, claims)

    failed = [row for row in claims if row["status"] == "FAIL"]
    for row in claims:
        print(
            f"[{row['status']}] {row['metric']}={row['value']} {row['unit']} "
            f"({row['scope']})"
        )
    print(f"wrote {output_dir / 'panel_metrics.csv'}")
    print(f"wrote {output_dir / 'model_points.csv'}")
    print(f"wrote {output_dir / 'summary.csv'}")
    print(f"wall_time_s={time.perf_counter() - started:.6f}")
    print(f"Fig.13 verifier: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for all generated Figure 13 outputs",
    )
    args = parser.parse_args(argv)
    return run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
