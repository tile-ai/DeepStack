"""User-configurable parameters for the standalone DRAM-bank oracle.

The values in :class:`DramBankSpec` are deliberately synthetic.  They keep the
examples and tests independent of any target-specific calibration.  A caller
can replace every physical field explicitly, including topology, row geometry,
service timing, and both clock domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DramBankSpec:
    """Immutable physical snapshot used by the fine model.

    ``total_layers`` counts physical row-buffer layers and
    ``connected_layers`` counts independently transferring ports per bank
    column.  Keeping them separate lets the scheduler model several physical
    banks sharing fewer transfer ports.

    The defaults form a documented toy system; they are not measurements and
    are not used by the reference-result reproduction path.
    """

    total_layers: int = 8
    connected_layers: int = 8
    banks_per_layer: int = 4
    row_bytes: int = 4 * 1024
    sector_bytes: int = 128
    sector_cycles: int = 3
    recharge_cycles: int = 7
    data_rate_hz: float = 1.0e9
    round_trip_latency_cycles: float = 20.0
    latency_clock_hz: float = 2.0e9

    def __post_init__(self) -> None:
        if self.total_layers <= 0:
            raise ValueError("total_layers must be positive")
        if not 0 < self.connected_layers <= self.total_layers:
            raise ValueError("connected_layers must be in [1, total_layers]")
        if self.banks_per_layer <= 0:
            raise ValueError("banks_per_layer must be positive")
        if self.row_bytes <= 0 or self.sector_bytes <= 0:
            raise ValueError("row_bytes and sector_bytes must be positive")
        if self.row_bytes % self.sector_bytes:
            raise ValueError("row_bytes must be an integer multiple of sector_bytes")
        if not 1 <= self.row_bytes // self.sector_bytes <= 64:
            raise ValueError("sectors_per_row must be in [1, 64]")
        if self.sector_cycles <= 0 or self.recharge_cycles < 0:
            raise ValueError("invalid sector/recharge cycle count")
        if self.data_rate_hz <= 0 or self.latency_clock_hz <= 0:
            raise ValueError("clock rates must be positive")
        if self.round_trip_latency_cycles < 0:
            raise ValueError("round_trip_latency_cycles must be non-negative")

    @property
    def physical_bank_count(self) -> int:
        return self.total_layers * self.banks_per_layer

    @property
    def port_count(self) -> int:
        return self.connected_layers * self.banks_per_layer

    @property
    def bank_count(self) -> int:
        """Compatibility alias for the number of physical banks.

        Existing layout/swizzle code uses ``bank_count`` as the size of the
        reversible bank field.  It must therefore denote physical banks, not
        the smaller number of transfer ports in a partially connected stack.
        """

        return self.physical_bank_count

    @property
    def sectors_per_row(self) -> int:
        return self.row_bytes // self.sector_bytes

    @property
    def row_read_cycles(self) -> int:
        return self.sectors_per_row * self.sector_cycles

    @property
    def full_row_cycles(self) -> int:
        return self.row_read_cycles + self.recharge_cycles

    @property
    def bytes_per_bank_cycle(self) -> float:
        return self.sector_bytes / self.sector_cycles

    @property
    def raw_peak_bandwidth_bytes_s(self) -> float:
        return self.port_count * self.bytes_per_bank_cycle * self.data_rate_hz

    @property
    def connectivity_efficiency(self) -> float:
        """Analytic steady-state cap for the caller-supplied topology.

        With enough backing layers per connected layer, another bank can cover
        recharge.  A fully connected topology instead exposes the
        caller-supplied row-read/full-row ratio.  Intermediate connection
        counts use a deterministic linear sensitivity between those limits;
        explicit scheduling still captures finite-wave and layout effects.
        """

        half = self.total_layers / 2.0
        if self.connected_layers <= half:
            return 1.0
        direct_efficiency = self.row_read_cycles / self.full_row_cycles
        if self.connected_layers >= self.total_layers:
            return direct_efficiency
        interpolation = (self.connected_layers - half) / (
            self.total_layers - half
        )
        return 1.0 + interpolation * (direct_efficiency - 1.0)

    @property
    def cycle_seconds(self) -> float:
        return 1.0 / self.data_rate_hz

    @property
    def round_trip_latency_seconds(self) -> float:
        return self.round_trip_latency_cycles / self.latency_clock_hz

    @property
    def round_trip_latency_data_cycles(self) -> float:
        return self.round_trip_latency_seconds / self.cycle_seconds

    @property
    def full_bank_wave_bytes(self) -> int:
        """All physical banks each contribute one complete row."""

        return self.bank_count * self.row_bytes

    def cycles_to_seconds(self, cycles: float) -> float:
        return cycles * self.cycle_seconds

    def seconds_to_cycles(self, seconds: float) -> float:
        return seconds / self.cycle_seconds

    @classmethod
    def from_arch(
        cls,
        arch: Any,
        *,
        banks_per_layer: Optional[int] = None,
        total_layers: Optional[int] = None,
        connected_layers: Optional[int] = None,
        row_bytes: Optional[int] = None,
        sector_bytes: Optional[int] = None,
        sector_cycles: Optional[int] = None,
        recharge_cycles: Optional[int] = None,
        data_rate_hz: Optional[float] = None,
        round_trip_latency_cycles: Optional[float] = None,
        latency_clock_hz: Optional[float] = None,
    ) -> "DramBankSpec":
        """Copy matching physical fields from ``arch`` plus explicit overrides.

        This adapter never infers a transfer rate from an unrelated clock and
        never consumes a pre-scaled bandwidth field.  Missing fields retain the
        documented synthetic defaults.  For calibrated studies, pass every
        physical value explicitly or expose the matching ``dram_*`` attribute
        on the architecture object.
        """

        defaults = cls()

        def choose(explicit: Any, attribute: str, fallback: Any) -> Any:
            if explicit is not None:
                return explicit
            return getattr(arch, attribute, fallback)

        m = int(choose(
            total_layers,
            "dram_layers_per_cluster",
            defaults.total_layers,
        ))
        n = int(choose(
            connected_layers,
            "dram_active_layers",
            min(defaults.connected_layers, m),
        ))
        return cls(
            total_layers=m,
            connected_layers=n,
            banks_per_layer=int(choose(
                banks_per_layer,
                "dram_banks_per_layer",
                defaults.banks_per_layer,
            )),
            row_bytes=int(choose(
                row_bytes,
                "dram_row_bytes",
                defaults.row_bytes,
            )),
            sector_bytes=int(choose(
                sector_bytes,
                "dram_sector_bytes",
                defaults.sector_bytes,
            )),
            sector_cycles=int(choose(
                sector_cycles,
                "dram_sector_cycles",
                defaults.sector_cycles,
            )),
            recharge_cycles=int(choose(
                recharge_cycles,
                "dram_recharge_cycles",
                defaults.recharge_cycles,
            )),
            data_rate_hz=float(choose(
                data_rate_hz,
                "dram_data_rate_hz",
                defaults.data_rate_hz,
            )),
            round_trip_latency_cycles=float(choose(
                round_trip_latency_cycles,
                "dram_round_trip_latency_cycles",
                defaults.round_trip_latency_cycles,
            )),
            latency_clock_hz=float(choose(
                latency_clock_hz,
                "dram_latency_clock_hz",
                defaults.latency_clock_hz,
            )),
        )
