import numpy as np
import pytest

from bank_oracle.layout import MatrixAccess, TensorLayout


def _sectors_from_scalar_elements(
    layout,
    shape,
    access,
    dtype_bytes,
    sector_bytes=128,
):
    # Independent scalar reference for linear layouts.
    rows, cols = shape
    inner = cols if layout.order == "row_major" else rows
    ld = layout.leading_dim if layout.leading_dim is not None else inner
    pitch = ld * dtype_bytes
    if layout.kind == "pitch_pad":
        unit = layout.pitch_unit_bytes
        pitch = (
            (pitch + unit - 1) // unit
        ) * unit + layout.pitch_pad_sectors * unit
    sectors = set()
    for i in range(access.row_start, access.row_stop):
        for j in range(access.col_start, access.col_stop):
            if layout.order == "row_major":
                address = layout.base_offset_bytes + i * pitch + j * dtype_bytes
            else:
                address = layout.base_offset_bytes + j * pitch + i * dtype_bytes
            sectors.add(address // sector_bytes)
    return np.asarray(sorted(sectors), dtype=np.int64)


@pytest.mark.parametrize("order", ["row_major", "column_major"])
@pytest.mark.parametrize("kind", ["linear", "pitch_pad"])
def test_linear_and_padded_rectangles_match_scalar_reference(order, kind):
    layout = TensorLayout(
        kind=kind,
        order=order,
        leading_dim=41 if order == "row_major" else 37,
        base_offset_bytes=17,
        pitch_pad_sectors=3,
    )
    shape = (37, 41)
    access = MatrixAccess(3, 34, 5, 39)
    actual = layout.touched_sectors(shape, access, 2)
    expected = _sectors_from_scalar_elements(layout, shape, access, 2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("order", ["row_major", "column_major"])
@pytest.mark.parametrize("tile_order", ["row_major", "column_major"])
def test_tile_major_is_alias_free_on_ragged_shape(order, tile_order):
    shape = (7, 11)
    layout = TensorLayout(
        kind="tile_major",
        order=order,
        tile_order=tile_order,
        tile_shape=(4, 5),
    )
    addresses = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            interval = layout.byte_intervals(
                shape, MatrixAccess(i, i + 1, j, j + 1), 2
            )
            assert len(interval) == 1
            addresses.append(interval[0][0])
    assert len(set(addresses)) == shape[0] * shape[1]
    assert max(addresses) < layout.storage_nbytes(shape, 2)


def test_pitch_pad_changes_stride_without_aliasing():
    shape = (32, 32)
    linear = TensorLayout(kind="linear")
    padded = TensorLayout(kind="pitch_pad", pitch_pad_sectors=1)
    whole = MatrixAccess(0, 32, 0, 32)
    linear_sectors = linear.touched_sectors(shape, whole, 2)
    padded_sectors = padded.touched_sectors(shape, whole, 2)
    assert padded.storage_nbytes(shape, 2) > linear.storage_nbytes(shape, 2)
    assert not np.array_equal(linear_sectors, padded_sectors)


def test_pitch_padding_unit_is_caller_configurable():
    layout = TensorLayout(
        kind="pitch_pad",
        leading_dim=37,
        pitch_pad_sectors=2,
        pitch_unit_bytes=96,
    )
    shape = (9, 37)
    access = MatrixAccess(0, 9, 0, 37)
    actual = layout.touched_sectors(
        shape,
        access,
        dtype_bytes=2,
        sector_bytes=96,
    )
    expected = _sectors_from_scalar_elements(
        layout,
        shape,
        access,
        dtype_bytes=2,
        sector_bytes=96,
    )
    np.testing.assert_array_equal(actual, expected)
    assert layout.storage_nbytes(shape, 2) == 9 * (96 + 2 * 96)
