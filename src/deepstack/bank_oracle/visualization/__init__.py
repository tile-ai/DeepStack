"""Interactive, self-contained bank-wave diagnostics."""

from .bank_wave import (
    RequestGroup,
    build_bank_wave_snapshot,
    build_demo_dashboard,
    build_gemm_wave_snapshot,
    build_logical_matrix_maps,
    render_dashboard_html,
    write_dashboard_html,
)

__all__ = [
    "RequestGroup",
    "build_bank_wave_snapshot",
    "build_demo_dashboard",
    "build_gemm_wave_snapshot",
    "build_logical_matrix_maps",
    "render_dashboard_html",
    "write_dashboard_html",
]
