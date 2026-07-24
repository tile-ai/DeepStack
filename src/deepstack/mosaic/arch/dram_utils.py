"""Generic caller-configured DRAM connectivity helpers."""

from .custom_profile import dram_connectivity_efficiency


def dram_bw_efficiency(
    total_layers: int,
    active_layers: int,
    *,
    fully_connected_efficiency: float,
) -> float:
    """Compatibility name for the source-visible generic efficiency model."""

    return dram_connectivity_efficiency(
        total_layers,
        active_layers,
        fully_connected_efficiency=fully_connected_efficiency,
    )
