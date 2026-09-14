"""Validate constructive bank-scheduling witnesses on synthetic inputs."""

from __future__ import annotations

import argparse

import numpy as np

from bank_oracle.engine import service_sectors
from bank_oracle.layout import BankSwizzle
from bank_oracle.spec import DramBankSpec


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _sector_permutation(spec: DramBankSpec) -> BankSwizzle:
    if (
        _is_power_of_two(spec.bank_count)
        and _is_power_of_two(spec.sectors_per_row)
    ):
        width = min(
            spec.bank_count.bit_length() - 1,
            spec.sectors_per_row.bit_length() - 1,
        )
        return BankSwizzle(kind="xor", sector_bits=width)
    return BankSwizzle(kind="cyclic", cyclic_sector_alpha=1)


def _row_permutation(spec: DramBankSpec) -> BankSwizzle:
    if (
        _is_power_of_two(spec.bank_count)
        and _is_power_of_two(spec.sectors_per_row)
    ):
        return BankSwizzle(
            kind="xor",
            row_bits=spec.bank_count.bit_length() - 1,
        )
    return BankSwizzle(kind="cyclic", cyclic_row_beta=1)


def validate_synthetic(spec: DramBankSpec) -> None:
    """Prove two explicit schedules attain their configured lower bounds."""

    spread_count = min(spec.bank_count, spec.sectors_per_row)
    sector_trace = np.arange(spread_count, dtype=np.int64)
    sector_linear = service_sectors(sector_trace, spec)
    sector_xor = service_sectors(
        sector_trace,
        spec,
        swizzle=_sector_permutation(spec),
    )

    row_trace = np.concatenate(
        [
            np.arange(
                row * spec.bank_count * spec.sectors_per_row,
                row * spec.bank_count * spec.sectors_per_row
                + spec.sectors_per_row,
                dtype=np.int64,
            )
            for row in range(spec.bank_count)
        ]
    )
    row_linear = service_sectors(row_trace, spec)
    row_xor = service_sectors(
        row_trace,
        spec,
        swizzle=_row_permutation(spec),
    )

    assert sector_xor.cycles == sector_xor.sector_lower_bound_cycles
    assert row_xor.cycles == row_xor.sector_lower_bound_cycles
    assert row_xor.cycles == spec.full_row_cycles

    print("synthetic configuration")
    print(
        f"layers={spec.total_layers}, connected={spec.connected_layers}, "
        f"banks/layer={spec.banks_per_layer}, "
        f"sectors/row={spec.sectors_per_row}"
    )
    print("trace                 linear      swizzled      lower-bound")
    print(
        f"contiguous sectors {sector_linear.cycles:11.1f} "
        f"{sector_xor.cycles:13.1f} "
        f"{sector_xor.sector_lower_bound_cycles:16.1f}"
    )
    print(
        f"colliding rows     {row_linear.cycles:11.1f} "
        f"{row_xor.cycles:13.1f} "
        f"{row_xor.sector_lower_bound_cycles:16.1f}"
    )


def validate_shared_ports(spec: DramBankSpec) -> None:
    """Check that a partial topology remains below its analytic cap."""

    connected = max(1, spec.total_layers // 2)
    partial = DramBankSpec(
        total_layers=spec.total_layers,
        connected_layers=connected,
        banks_per_layer=spec.banks_per_layer,
        row_bytes=spec.row_bytes,
        sector_bytes=spec.sector_bytes,
        sector_cycles=spec.sector_cycles,
        recharge_cycles=spec.recharge_cycles,
        data_rate_hz=spec.data_rate_hz,
        round_trip_latency_cycles=spec.round_trip_latency_cycles,
        latency_clock_hz=spec.latency_clock_hz,
    )
    rows_per_bank = 9
    sectors = np.arange(
        partial.bank_count * partial.sectors_per_row * rows_per_bank,
        dtype=np.int64,
    )
    result = service_sectors(sectors, partial)
    assert result.overall_efficiency <= (
        partial.connectivity_efficiency + 1e-12
    )
    assert result.sector_lower_bound_cycles <= result.cycles
    print(
        "partial topology: "
        f"physical_banks={partial.bank_count}, ports={partial.port_count}, "
        f"cap={partial.connectivity_efficiency:.4f}, "
        f"achieved={result.overall_efficiency:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row-sectors",
        type=int,
        default=None,
        help="override sectors per row for the synthetic validation",
    )
    args = parser.parse_args()
    spec = DramBankSpec()
    if args.row_sectors is not None:
        spec = DramBankSpec(
            row_bytes=args.row_sectors * spec.sector_bytes,
            sector_bytes=spec.sector_bytes,
        )
    validate_synthetic(spec)
    validate_shared_ports(spec)


if __name__ == "__main__":
    main()
