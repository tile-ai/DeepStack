import numpy as np

def extract_blocks(gridM, gridN, Stride_M, Stride_N, raster_axis="legacy"):
    """Generate the (m, n) CTA iteration order under a 2D block raster.

    The `raster_axis` argument selects the scan pattern. All modes
    share the same `Stride_M × Stride_N` sub-block geometry; only
    the inner-loop order and outer (sub-block advance) direction
    differ. The legacy default reproduces the original sequence
    byte-for-byte; the two cutlass modes correspond exactly to
    cutlass `RasterOrder::AlongM` and `RasterOrder::AlongN` with
    `swizzle_size = Stride_N` (along_m) or `Stride_M` (along_n).

    Modes
    -----
    'legacy' (default; row_panel-only callers):
        Within each sub-block iterate `for m: for n:` (N fast at
        the micro level). Sub-blocks then slide along M (m_start
        first). This is a 2D-block raster that is NEITHER cutlass
        AlongM nor AlongN — it mixes M-fast-macro with N-fast-micro
        and is preserved for backward compatibility.

    'along_m' (cutlass `RasterOrder::AlongM`):
        M is fast at BOTH levels. Each sub-block scans `for n: for m:`
        (M fast at micro), and sub-blocks slide along M first
        (m_start advances). Stride_N is the strip width; Stride_M is
        ignored for non-strip configurations because cutlass AlongM
        sweeps full gridM per strip — pass Stride_M = gridM to
        match exactly, or any Stride_M to get a "broken-up" variant.

    'along_n' (cutlass `RasterOrder::AlongN`):
        N is fast at BOTH levels. Each sub-block scans `for m: for n:`
        (N fast at micro), and sub-blocks slide along N first
        (n_start advances). Stride_M is the strip height.

    Args:
        gridM, gridN: number of CTA tiles in the M and N dimensions.
        Stride_M, Stride_N: sub-block dimensions.
        raster_axis: 'legacy' (default), 'along_m', or 'along_n'.

    Returns:
        coords: (gridM*gridN, 2) int array of (m, n) tile indices.
    """
    coords = np.zeros((gridM * gridN, 2), dtype=int)
    count = 0

    m_start = 1
    n_start = 1
    if raster_axis == "legacy":
        # Original sequence: inner = for m: for n (N-fast
        # micro), outer = m_start first (M-fast macro).
        while m_start <= gridM and n_start <= gridN:
            m_end = min(m_start + Stride_M - 1, gridM)
            n_end = min(n_start + Stride_N - 1, gridN)
            for m in range(m_start, m_end + 1):
                for n in range(n_start, n_end + 1):
                    coords[count] = [m, n]
                    count += 1
            m_start = m_end + 1
            if m_start > gridM:
                m_start = 1
                n_start = n_end + 1
    elif raster_axis == "along_m":
        # cutlass AlongM: M-fast everywhere.
        # Inner: for n: for m (M-fast micro).
        # Outer: m_start first (M-fast macro).
        while m_start <= gridM and n_start <= gridN:
            m_end = min(m_start + Stride_M - 1, gridM)
            n_end = min(n_start + Stride_N - 1, gridN)
            for n in range(n_start, n_end + 1):
                for m in range(m_start, m_end + 1):
                    coords[count] = [m, n]
                    count += 1
            m_start = m_end + 1
            if m_start > gridM:
                m_start = 1
                n_start = n_end + 1
    elif raster_axis == "along_n":
        # cutlass AlongN: N-fast everywhere.
        # Inner: for m: for n (N-fast micro).
        # Outer: n_start first (N-fast macro).
        while m_start <= gridM and n_start <= gridN:
            m_end = min(m_start + Stride_M - 1, gridM)
            n_end = min(n_start + Stride_N - 1, gridN)
            for m in range(m_start, m_end + 1):
                for n in range(n_start, n_end + 1):
                    coords[count] = [m, n]
                    count += 1
            n_start = n_end + 1
            if n_start > gridN:
                n_start = 1
                m_start = m_end + 1
    else:
        raise ValueError(
            f"unknown raster_axis {raster_axis!r}; expected "
            f"'legacy', 'along_m', or 'along_n'"
        )

    return coords[:count]

def extract_blocks_triton_swizzle_column_major(gridM, gridN, Group_M):
    # 初始化坐标矩阵，Python中没有直接等价于Matlab中的zeros函数，使用numpy的zeros
    coords = np.zeros((gridM * gridN, 2), dtype=int)
    count = 0
    
    m_start = 1
    n_start = 1
    
    while m_start <= gridM and n_start <= gridN:
        m_end = min(m_start + Group_M - 1, gridM)
        n_end = gridN
        
        
        for n in range(n_start, n_end + 1):
            for m in range(m_start, m_end + 1):
                coords[count] = [m, n]
                count += 1
                # if (count<=4000):
                #     print(m, n)
                
        
        m_start = m_end + 1
    
    # 调整coords数组的大小为实际使用的大小
    coords = coords[:count]
    
    return coords

if __name__ == '__main__':
    # gridM = 5
    # gridN = 5
    # Stride_M = 2
    # Stride_N = 2
    # coords = extract_blocks(gridM, gridN, Stride_M, Stride_N)
    # print(coords)

    gridM = 9
    gridN = 9
    Group_M=2
    coords = extract_blocks_triton_swizzle_column_major(gridM, gridN, Group_M)
    print(coords)


