"""Run a small synthetic GEMV/FIFO-L2 diagnostic."""

from __future__ import annotations

from bank_oracle.gemm import (
    GemmBankOptions,
    GemmProblem,
    GemmTiling,
    make_layout_set,
    model_gemm_bank,
)
from bank_oracle.l2_cache import L2CacheSpec
from bank_oracle.layout import BankSwizzle
from bank_oracle.pipeline import model_gemm_pipeline
from bank_oracle.spec import DramBankSpec


def _model(l2_bytes: int | None):
    spec = DramBankSpec()
    problem = GemmProblem(1, 384, 512)
    tiling = GemmTiling(
        1,
        64,
        32,
        stage=3,
        ctas_per_wave=4,
        row_panel=2,
    )
    bank_bits = spec.bank_count.bit_length() - 1
    sector_bits = min(
        4,
        bank_bits,
        spec.sectors_per_row.bit_length() - 1,
    )
    layouts = make_layout_set(
        problem,
        tiling,
        spec,
        preset="tile_major",
        swizzle=BankSwizzle(
            kind="xor",
            sector_bits=sector_bits,
            sector_bank_shift=bank_bits - sector_bits,
        ),
        name="synthetic_gemv_xor",
    )
    options = GemmBankOptions(
        max_spatial_samples=None,
        max_k_samples=None,
        l2_cache=(
            None
            if l2_bytes is None
            else L2CacheSpec(
                capacity_bytes=l2_bytes,
                line_bytes=spec.sector_bytes,
            )
        ),
    )
    bank = model_gemm_bank(problem, tiling, layouts, spec, options)
    pipeline = model_gemm_pipeline(
        bank,
        compute_cycles_per_k=1200,
        other_memory_cycles_per_k=0,
        store_other_memory_cycles=0,
    )
    return bank, pipeline


def main() -> None:
    uncached, uncached_pipeline = _model(None)
    cached, cached_pipeline = _model(256 * 1024)

    assert cached.l2_result is not None
    assert cached.actual_read_bytes <= uncached.actual_read_bytes
    assert cached_pipeline.total_seconds <= uncached_pipeline.total_seconds
    assert 0.0 <= cached.l2_result.a.line_hit_rate <= 1.0
    assert 0.0 <= cached.l2_result.b.line_hit_rate <= 1.0

    print("synthetic GEMV/FIFO-L2 diagnostic")
    print("mode       read_bytes    total_cycles    A-line-hit    B-line-hit")
    print(
        f"uncached {uncached.actual_read_bytes:12.0f} "
        f"{uncached_pipeline.total_cycles:15.1f} "
        f"{0.0:12.4f} {0.0:12.4f}"
    )
    print(
        f"cached   {cached.actual_read_bytes:12.0f} "
        f"{cached_pipeline.total_cycles:15.1f} "
        f"{cached.l2_result.a.line_hit_rate:12.4f} "
        f"{cached.l2_result.b.line_hit_rate:12.4f}"
    )


if __name__ == "__main__":
    main()
