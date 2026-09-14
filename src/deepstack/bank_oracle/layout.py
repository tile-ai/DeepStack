"""Tensor storage layouts and reversible physical-bank swizzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .spec import DramBankSpec


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, alignment: int) -> int:
    return _ceil_div(x, alignment) * alignment


@dataclass(frozen=True)
class BankSwizzle:
    """A static, reversible permutation of the physical bank field.

    ``xor`` mirrors the triangular bit-XOR family used by CuTe layouts.  It
    keeps physical ``row`` and ``sector`` unchanged and XORs selected source
    bits into bank bits.  For fixed (row, sector), this is a permutation over
    banks and therefore cannot alias addresses.

    ``cyclic`` is the corresponding modular permutation for non-power-of-two
    bank counts.
    """

    kind: str = "none"  # none | xor | cyclic
    phase: int = 0

    sector_source_shift: int = 0
    sector_bits: int = 0
    sector_bank_shift: int = 0

    row_source_shift: int = 0
    row_bits: int = 0
    row_bank_shift: int = 0

    cyclic_sector_alpha: int = 0
    cyclic_row_beta: int = 0

    def validate(self, spec: DramBankSpec) -> None:
        if self.kind not in {"none", "xor", "cyclic"}:
            raise ValueError("swizzle kind must be none, xor, or cyclic")
        if not 0 <= self.phase < spec.bank_count:
            raise ValueError("phase must be in [0, bank_count)")
        if self.kind == "xor":
            if not _is_power_of_two(spec.bank_count):
                raise ValueError("xor swizzle requires a power-of-two bank count")
            bank_bits = spec.bank_count.bit_length() - 1
            sector_width = spec.sectors_per_row.bit_length() - 1
            if not _is_power_of_two(spec.sectors_per_row):
                raise ValueError("xor swizzle requires power-of-two sectors_per_row")
            values = (
                self.sector_source_shift,
                self.sector_bits,
                self.sector_bank_shift,
                self.row_source_shift,
                self.row_bits,
                self.row_bank_shift,
            )
            if any(value < 0 for value in values):
                raise ValueError("xor bit parameters must be non-negative")
            if self.sector_source_shift + self.sector_bits > sector_width:
                raise ValueError("sector source bits exceed sector field")
            if self.sector_bank_shift + self.sector_bits > bank_bits:
                raise ValueError("sector hash exceeds bank field")
            if self.row_bank_shift + self.row_bits > bank_bits:
                raise ValueError("row hash exceeds bank field")

    def map_bank(
        self,
        bank: np.ndarray,
        row: np.ndarray,
        sector: np.ndarray,
        spec: DramBankSpec,
    ) -> np.ndarray:
        self.validate(spec)
        bank = np.asarray(bank, dtype=np.int64)
        row = np.asarray(row, dtype=np.int64)
        sector = np.asarray(sector, dtype=np.int64)
        if self.kind == "none":
            return (bank + self.phase) % spec.bank_count
        if self.kind == "xor":
            sector_mask = (1 << self.sector_bits) - 1
            row_mask = (1 << self.row_bits) - 1
            sector_hash = (
                ((sector >> self.sector_source_shift) & sector_mask)
                << self.sector_bank_shift
            )
            row_hash = (
                ((row >> self.row_source_shift) & row_mask)
                << self.row_bank_shift
            )
            return (bank ^ sector_hash ^ row_hash ^ self.phase) & (spec.bank_count - 1)
        return (
            bank
            + self.cyclic_sector_alpha * sector
            + self.cyclic_row_beta * row
            + self.phase
        ) % spec.bank_count

    def unmap_bank(
        self,
        mapped_bank: np.ndarray,
        row: np.ndarray,
        sector: np.ndarray,
        spec: DramBankSpec,
    ) -> np.ndarray:
        """Inverse bank permutation for validation and debugging."""

        self.validate(spec)
        mapped_bank = np.asarray(mapped_bank, dtype=np.int64)
        row = np.asarray(row, dtype=np.int64)
        sector = np.asarray(sector, dtype=np.int64)
        if self.kind == "none":
            return (mapped_bank - self.phase) % spec.bank_count
        if self.kind == "xor":
            # XOR is its own inverse because row and sector are unchanged.
            sector_mask = (1 << self.sector_bits) - 1
            row_mask = (1 << self.row_bits) - 1
            sector_hash = (
                ((sector >> self.sector_source_shift) & sector_mask)
                << self.sector_bank_shift
            )
            row_hash = (
                ((row >> self.row_source_shift) & row_mask)
                << self.row_bank_shift
            )
            return (mapped_bank ^ sector_hash ^ row_hash ^ self.phase) & (spec.bank_count - 1)
        return (
            mapped_bank
            - self.cyclic_sector_alpha * sector
            - self.cyclic_row_beta * row
            - self.phase
        ) % spec.bank_count


@dataclass(frozen=True)
class MatrixAccess:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    @property
    def rows(self) -> int:
        return max(self.row_stop - self.row_start, 0)

    @property
    def cols(self) -> int:
        return max(self.col_stop - self.col_start, 0)


@dataclass(frozen=True)
class TensorLayout:
    """Logical 2-D tensor storage plus an optional physical bank swizzle."""

    kind: str = "linear"  # linear | pitch_pad | tile_major
    order: str = "row_major"  # row_major | column_major
    leading_dim: Optional[int] = None
    base_offset_bytes: int = 0
    pitch_pad_sectors: int = 0
    pitch_unit_bytes: int = 128
    tile_shape: Optional[Tuple[int, int]] = None
    tile_order: str = "row_major"
    swizzle: BankSwizzle = BankSwizzle()
    name: str = ""

    def validate(self, shape: Tuple[int, int], dtype_bytes: int) -> None:
        rows, cols = shape
        if rows <= 0 or cols <= 0 or dtype_bytes <= 0:
            raise ValueError("shape and dtype_bytes must be positive")
        if self.kind not in {"linear", "pitch_pad", "tile_major"}:
            raise ValueError("layout kind must be linear, pitch_pad, or tile_major")
        if self.order not in {"row_major", "column_major"}:
            raise ValueError("order must be row_major or column_major")
        if self.tile_order not in {"row_major", "column_major"}:
            raise ValueError("tile_order must be row_major or column_major")
        if (
            self.base_offset_bytes < 0
            or self.pitch_pad_sectors < 0
            or self.pitch_unit_bytes <= 0
        ):
            raise ValueError(
                "base offset/pitch padding must be non-negative and "
                "pitch_unit_bytes must be positive"
            )
        inner = cols if self.order == "row_major" else rows
        if self.leading_dim is not None and self.leading_dim < inner:
            raise ValueError("leading_dim is smaller than the logical inner extent")
        if self.kind == "tile_major":
            if self.tile_shape is None or min(self.tile_shape) <= 0:
                raise ValueError("tile_major requires a positive tile_shape")

    def storage_nbytes(self, shape: Tuple[int, int], dtype_bytes: int) -> int:
        self.validate(shape, dtype_bytes)
        rows, cols = shape
        if self.kind == "tile_major":
            assert self.tile_shape is not None
            tile_rows, tile_cols = self.tile_shape
            return (
                _ceil_div(rows, tile_rows)
                * _ceil_div(cols, tile_cols)
                * tile_rows
                * tile_cols
                * dtype_bytes
            )
        inner = cols if self.order == "row_major" else rows
        outer = rows if self.order == "row_major" else cols
        ld = self.leading_dim if self.leading_dim is not None else inner
        pitch = ld * dtype_bytes
        if self.kind == "pitch_pad":
            pitch = (
                _align_up(pitch, self.pitch_unit_bytes)
                + self.pitch_pad_sectors * self.pitch_unit_bytes
            )
        return outer * pitch

    def _linear_intervals(
        self,
        shape: Tuple[int, int],
        access: MatrixAccess,
        dtype_bytes: int,
    ) -> List[Tuple[int, int]]:
        rows, cols = shape
        inner = cols if self.order == "row_major" else rows
        ld = self.leading_dim if self.leading_dim is not None else inner
        pitch = ld * dtype_bytes
        if self.kind == "pitch_pad":
            pitch = (
                _align_up(pitch, self.pitch_unit_bytes)
                + self.pitch_pad_sectors * self.pitch_unit_bytes
            )
        intervals: List[Tuple[int, int]] = []
        if self.order == "row_major":
            for row in range(access.row_start, access.row_stop):
                start = self.base_offset_bytes + row * pitch + access.col_start * dtype_bytes
                stop = self.base_offset_bytes + row * pitch + access.col_stop * dtype_bytes
                intervals.append((start, stop))
        else:
            for col in range(access.col_start, access.col_stop):
                start = self.base_offset_bytes + col * pitch + access.row_start * dtype_bytes
                stop = self.base_offset_bytes + col * pitch + access.row_stop * dtype_bytes
                intervals.append((start, stop))
        return intervals

    def _tile_intervals(
        self,
        shape: Tuple[int, int],
        access: MatrixAccess,
        dtype_bytes: int,
    ) -> List[Tuple[int, int]]:
        assert self.tile_shape is not None
        rows, cols = shape
        tile_rows, tile_cols = self.tile_shape
        grid_rows = _ceil_div(rows, tile_rows)
        grid_cols = _ceil_div(cols, tile_cols)

        def tile_id(tile_row: int, tile_col: int) -> int:
            if self.tile_order == "row_major":
                return tile_row * grid_cols + tile_col
            return tile_col * grid_rows + tile_row

        tile_elems = tile_rows * tile_cols
        intervals: List[Tuple[int, int]] = []
        if self.order == "row_major":
            first_tile_col = access.col_start // tile_cols
            last_tile_col = (access.col_stop - 1) // tile_cols
            for row in range(access.row_start, access.row_stop):
                tile_row, inner_row = divmod(row, tile_rows)
                for tile_col in range(first_tile_col, last_tile_col + 1):
                    c0 = max(access.col_start, tile_col * tile_cols)
                    c1 = min(access.col_stop, (tile_col + 1) * tile_cols)
                    inner_col = c0 - tile_col * tile_cols
                    elem = (
                        tile_id(tile_row, tile_col) * tile_elems
                        + inner_row * tile_cols
                        + inner_col
                    )
                    start = self.base_offset_bytes + elem * dtype_bytes
                    intervals.append((start, start + (c1 - c0) * dtype_bytes))
        else:
            first_tile_row = access.row_start // tile_rows
            last_tile_row = (access.row_stop - 1) // tile_rows
            for col in range(access.col_start, access.col_stop):
                tile_col, inner_col = divmod(col, tile_cols)
                for tile_row in range(first_tile_row, last_tile_row + 1):
                    r0 = max(access.row_start, tile_row * tile_rows)
                    r1 = min(access.row_stop, (tile_row + 1) * tile_rows)
                    inner_row = r0 - tile_row * tile_rows
                    elem = (
                        tile_id(tile_row, tile_col) * tile_elems
                        + inner_col * tile_rows
                        + inner_row
                    )
                    start = self.base_offset_bytes + elem * dtype_bytes
                    intervals.append((start, start + (r1 - r0) * dtype_bytes))
        return intervals

    def byte_intervals(
        self,
        shape: Tuple[int, int],
        access: MatrixAccess,
        dtype_bytes: int,
    ) -> List[Tuple[int, int]]:
        self.validate(shape, dtype_bytes)
        rows, cols = shape
        if not (
            0 <= access.row_start <= access.row_stop <= rows
            and 0 <= access.col_start <= access.col_stop <= cols
        ):
            raise ValueError("access rectangle is outside the tensor")
        if access.rows == 0 or access.cols == 0:
            return []
        if self.kind == "tile_major":
            return self._tile_intervals(shape, access, dtype_bytes)
        return self._linear_intervals(shape, access, dtype_bytes)

    def touched_sectors(
        self,
        shape: Tuple[int, int],
        access: MatrixAccess,
        dtype_bytes: int,
        sector_bytes: int = 128,
    ) -> np.ndarray:
        """Return sorted unique logical sector IDs touched by the rectangle."""

        chunks: List[np.ndarray] = []
        for start, stop in self.byte_intervals(shape, access, dtype_bytes):
            if stop <= start:
                continue
            first = start // sector_bytes
            last = (stop - 1) // sector_bytes
            chunks.append(np.arange(first, last + 1, dtype=np.int64))
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(chunks))
