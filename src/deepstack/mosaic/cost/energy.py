"""Source-visible energy equations with protected release calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from ..noc.energy_config import NocEnergyConfig
from ..noc.noc_energy import (
    build_extended_energy_matrix,
    build_extended_energy_matrix_switch_only,
)
from ..noc.noc_topo import (
    Hierarchy,
    TrafficMatrix,
    build_extended_traffic_matrix,
)


if TYPE_CHECKING:
    from tilesight.arch.arch_base import Arch


__all__ = [
    "ChipEnergyConfig",
    "EnergyModel",
    "build_extended_energy_matrix",
    "build_extended_energy_matrix_switch_only",
]


def _nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ChipEnergyConfig:
    """Caller-owned chip energy coefficients.

    Memory-hierarchy fields use pJ/bit, compute fields use pJ/op, and
    ``static_power_w`` uses watts.  Every field is required: constructing this
    object never consults the bundled release calibration.
    """

    register_read_pj_per_bit: float
    register_write_pj_per_bit: float
    shared_memory_read_pj_per_bit: float
    shared_memory_write_pj_per_bit: float
    l2_read_pj_per_bit: float
    l2_write_pj_per_bit: float
    dram_read_pj_per_bit: float
    dram_write_pj_per_bit: float
    sfu_pj_per_op: float
    cuda_core_pj_per_op: float
    tensor_core_pj_per_op: float
    static_power_w: float

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field,
                _nonnegative(getattr(self, field), field),
            )


class EnergyModel:
    """Evaluate chip and NoC energy with release or caller-owned inputs."""

    def __init__(
        self,
        arch: "Arch",
        chip_energy_config: ChipEnergyConfig | None = None,
        noc_energy_config: NocEnergyConfig | None = None,
    ):
        self.arch = arch
        inherited_chip_config = getattr(arch, "chip_energy_config", None)
        self.chip_energy_config = (
            chip_energy_config
            if chip_energy_config is not None
            else inherited_chip_config
        )
        if self.chip_energy_config is not None and not isinstance(
            self.chip_energy_config,
            ChipEnergyConfig,
        ):
            raise TypeError(
                "chip_energy_config must be a ChipEnergyConfig or None"
            )
        if noc_energy_config is not None and not isinstance(
            noc_energy_config,
            NocEnergyConfig,
        ):
            raise TypeError(
                "noc_energy_config must be a NocEnergyConfig or None"
            )
        self.noc_energy_config = noc_energy_config
        self._use_reference_chip_model = self.chip_energy_config is None

    def get_noc_energy(
        self,
        tm: "TrafficMatrix",
        h: Hierarchy,
    ) -> float:
        energy_matrix = build_extended_energy_matrix(
            h,
            self.noc_energy_config,
        )
        traffic, _ = build_extended_traffic_matrix(tm, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            pj_matrix = np.where(
                energy_matrix > 0.0,
                traffic * 8 * energy_matrix,
                0,
            )
        return float(np.sum(pj_matrix))

    def _memory_energy(
        self,
        level: int,
        rw: tuple[float, float],
        read_pj_per_bit: float,
        write_pj_per_bit: float,
    ) -> float:
        read_bytes, write_bytes = rw
        if self._use_reference_chip_model:
            from ..noc import _model_support

            return float(
                _model_support.p95(
                    level,
                    float(read_bytes),
                    float(write_bytes),
                )
            )
        return (
            8 * read_bytes * read_pj_per_bit
            + 8 * write_bytes * write_pj_per_bit
        )

    def get_reg_energy(self, rw: tuple[float, float]) -> float:
        config = self.chip_energy_config
        return self._memory_energy(
            0,
            rw,
            0.0 if config is None else config.register_read_pj_per_bit,
            0.0 if config is None else config.register_write_pj_per_bit,
        )

    def get_smem_energy(self, rw: tuple[float, float]) -> float:
        config = self.chip_energy_config
        return self._memory_energy(
            1,
            rw,
            0.0 if config is None else config.shared_memory_read_pj_per_bit,
            0.0 if config is None else config.shared_memory_write_pj_per_bit,
        )

    def get_l2_energy(self, rw: tuple[float, float]) -> float:
        config = self.chip_energy_config
        return self._memory_energy(
            2,
            rw,
            0.0 if config is None else config.l2_read_pj_per_bit,
            0.0 if config is None else config.l2_write_pj_per_bit,
        )

    def get_dram_energy(self, rw: tuple[float, float]) -> float:
        config = self.chip_energy_config
        return self._memory_energy(
            3,
            rw,
            0.0 if config is None else config.dram_read_pj_per_bit,
            0.0 if config is None else config.dram_write_pj_per_bit,
        )

    def _compute_energy(
        self,
        unit: int,
        operations: float,
        pj_per_op: float,
    ) -> float:
        if self._use_reference_chip_model:
            from ..noc import _model_support

            return float(
                _model_support.p94(
                    unit,
                    float(operations),
                )
            )
        return operations * pj_per_op

    def get_sfu_energy(self, operations: float) -> float:
        config = self.chip_energy_config
        return self._compute_energy(
            0,
            operations,
            0.0 if config is None else config.sfu_pj_per_op,
        )

    def get_cuda_core_energy(self, operations: float) -> float:
        config = self.chip_energy_config
        return self._compute_energy(
            1,
            operations,
            0.0 if config is None else config.cuda_core_pj_per_op,
        )

    def get_tensor_core_energy(self, operations: float) -> float:
        config = self.chip_energy_config
        return self._compute_energy(
            2,
            operations,
            0.0 if config is None else config.tensor_core_pj_per_op,
        )

    def get_memory_rw_energy(
        self,
        reg_rw: tuple[float, float],
        smem_rw: tuple[float, float],
        l2_rw: tuple[float, float],
        dram_rw: tuple[float, float],
    ) -> float:
        return (
            self.get_reg_energy(reg_rw)
            + self.get_smem_energy(smem_rw)
            + self.get_l2_energy(l2_rw)
            + self.get_dram_energy(dram_rw)
        )

    def get_compute_energy(
        self,
        sfu_ops: float,
        cuda_ops: float,
        tensor_ops: float,
    ) -> float:
        return (
            self.get_sfu_energy(sfu_ops)
            + self.get_cuda_core_energy(cuda_ops)
            + self.get_tensor_core_energy(tensor_ops)
        )

    def get_energy(
        self,
        tm: "TrafficMatrix",
        h: Hierarchy,
        reg_rw: tuple[float, float] = (0, 0),
        smem_rw: tuple[float, float] = (0, 0),
        l2_rw: tuple[float, float] = (0, 0),
        dram_rw: tuple[float, float] = (0, 0),
        sfu_ops: float = 0,
        cuda_ops: float = 0,
        tensor_ops: float = 0,
    ) -> float:
        return (
            self.get_noc_energy(tm, h)
            + self.get_memory_rw_energy(
                reg_rw,
                smem_rw,
                l2_rw,
                dram_rw,
            )
            + self.get_compute_energy(
                sfu_ops,
                cuda_ops,
                tensor_ops,
            )
        )

    def get_static_power(self) -> float:
        if self._use_reference_chip_model:
            from ..noc import _model_support

            return float(_model_support.p93())
        assert self.chip_energy_config is not None
        return self.chip_energy_config.static_power_w
