from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from mosaic.noc.traffic_matrix import TrafficMatrix
from mosaic.noc.route_stats import build_route_stats_from_extended
from mosaic.noc.noc_topo import get_total_traffic_bytes

if TYPE_CHECKING:
    import numpy as np
    from mosaic.noc.route_stats import RouteStats
    from mosaic.noc.noc_topo import Hierarchy
    from mosaic.noc.energy_config import NocEnergyConfig
    from mosaic.cost.energy import ChipEnergyConfig
    # Arch comes from tilesight; the IDE may not resolve its path. Used only as a string annotation, without runtime evaluation.
    from tilesight.arch.arch_base import Arch  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


@dataclass
class OpPerfStats:
    """Hierarchical, accumulated performance statistics for one operator invocation.

    Example:
        stats = OpPerfStats(op_name="mlp")
        stats.append_hete_list([data_a, data_b], n=1)
        stats.append_traffic(ext_traffic_tp, n=1)
        stats.append_hete_list([data_c], n=3)
        stats.append_traffic(ext_traffic_ep, n=3)
        stats.finalize(total_time_s=e2e_time, h=hierarchy)
        stats.dump_log()

    Each segment may have its own repetition count n. Finalize once the total
    end-to-end time is known.
    """

    op_name: str = ""
    dump_perf_log: bool = True       # Automatically call dump_log() after finalize completes.

    # ── Timing ──────────────────────────────────────────────────────────────
    compute_time_s: float = 0.0       # Σ(n_i × t_i), the sum of effective compute times across segments.
    comm_time_s: float = 0.0          # Σ(n_i × comm_i), the sum of collective communication times.
    e2e_time_s: float = 0.0           # Total e2e time (passed to finalize).
    comm_overlap_pct: float = 0.0     # Percentage of communication overlapped by compute % (=1-(e2e-compute)/comm).

    # ── Hierarchical Memory Traffic ──────────────────────────────────────────
    # From smem_fusion_post_data (tilesight hete_smem_fusion output).
    dram_util_pct: float = 0.0        # DRAM bandwidth utilization %.
    l2_util_pct: float = 0.0          # L2 bandwidth utilization %.
    l2_hit_rate_pct: float = 0.0      # L2 hit rate %.
    smem_l1_util_pct: float = 0.0     # Smem/L1 utilization %.
    smem_bytes_per_tb: float = 0.0    # smem (shared memory) footprint per thread block (bytes).
    reg_bytes_per_thread: float = 0.0 # Register footprint per thread (bytes).

    # ── Compute Utilization ──────────────────────────────────────────────────
    tensor_util_pct: float = 0.0      # Tensor Core utilization %.
    cuda_util_pct: float = 0.0        # CUDA Core (FP32) utilization %.
    sfu_util_pct: float = 0.0         # SFU utilization %.

    # ── Hierarchical NoC Traffic ─────────────────────────────────────────────
    noc_total_bytes: float = 0.0          # Total logical traffic (tm.totals), in bytes.
    noc_total_energy_j: float = 0.0       # NoC energy across all links, in J.
    noc_max_util: float = 0.0             # Maximum link utilization (averaged over e2e time).
    noc_mean_util: float = 0.0            # Mean utilization across nonzero links (averaged over e2e time).
    noc_bottleneck_src: int = -1
    noc_bottleneck_dst: int = -1
    noc_bottleneck_bytes: float = 0.0     # Bytes carried by the bottleneck link.
    noc_bottleneck_bw: float = 0.0        # Bottleneck link bandwidth (bytes/s).
    noc_hop_time_s: float = 0.0           # Σ(n_i × hop_i), accumulated switch hop latency (s).
    noc_link_time_s: float = 0.0          # Σ(n_i × link_i), accumulated link transfer time (s).
    tm: Optional[TrafficMatrix] = field(default=None, repr=False)
    route_stats: Optional["RouteStats"] = field(default=None, repr=False)

    # ── Chip Energy (memory hierarchy + compute) × num_devices ───────────────
    # Reverse calculation: actual_bytes = util × t_eff × arch.bandwidth.
    #           actual_ops  = util × t_eff × arch.peak_flops
    # DRAM/L2 read bytes come directly from hete row[7]/row[8] (exact tilesight values); write = total − read.
    # Smem/Reg: 50% read / 50% write (tilesight does not separate reads and writes).
    # Multiply all results by num_devices (obtained from h.num_devices or an explicit argument during finalize).
    chip_dram_energy_j:    float = 0.0    # DRAM access energy × devices (J).
    chip_l2_energy_j:      float = 0.0    # L2 access energy × devices (J).
    chip_smem_energy_j:    float = 0.0    # Smem (shared memory) access energy × devices (J).
    chip_reg_energy_j:     float = 0.0    # Reg access energy × devices (J).
    chip_memory_energy_j:  float = 0.0    # Sum of energy across all memory levels × devices (J).
    chip_tensor_energy_j:  float = 0.0    # Tensor Core compute energy × devices (J).
    chip_cuda_energy_j:    float = 0.0    # CUDA Core compute energy × devices (J).
    chip_sfu_energy_j:     float = 0.0    # SFU compute energy × devices (J).
    chip_compute_energy_j: float = 0.0    # Sum of energy across all compute units × devices (J).
    chip_static_energy_j:  float = 0.0    # Static leakage energy = static_power × e2e_time × devices (J).
    chip_total_energy_j:   float = 0.0    # Total chip energy (memory + compute + static) × devices (J).
    chip_num_devices:      int   = 1      # Number of devices used in energy calculations (for conversion to per-device values).

    # ── Internal accumulators (excluded from the __init__ signature) ────────────────────────────────
    # _hete_rows  : list of (smem_fusion_post_data, n: float)
    _hete_rows: list = field(default_factory=list, init=False, repr=False)
    # _traffic_mats : list of scaled np.ndarray (already multiplied by n in append_traffic).
    _traffic_mats: list = field(default_factory=list, init=False, repr=False)
    # _comm_time_accum : Σ(n_i × comm_i)
    _comm_time_accum: float = field(default=0.0, init=False, repr=False)
    # _hop_time_accum / _link_time_accum : Σ(n_i × hop_i/link_i)
    _hop_time_accum:  float = field(default=0.0, init=False, repr=False)
    _link_time_accum: float = field(default=0.0, init=False, repr=False)

    # ────────────────────────────────────────────────────────────────────────
    # Append API
    # ────────────────────────────────────────────────────────────────────────

    def append_hete(self, smem_fusion_post_data, n: float = 1.0) -> None:
        """Append one smem_fusion_post_data row for a segment executed n times."""
        self._hete_rows.append((smem_fusion_post_data, float(n)))

    def append_hete_list(self, smem_fusion_list: list, n: float = 1.0) -> None:
        """Append multiple smem_fusion_post_data rows with a shared repetition count n.
        Effective time is row[0]*n; utilization contributes effective_time*util_fraction.
        IO bytes contribute row[7/8]*n. Footprints use the maximum and do not scale with n.
        """
        for row in smem_fusion_list:
            self._hete_rows.append((row, float(n)))

    def append_traffic(
        self,
        traffic: "np.ndarray",
        hop_time_s: float = 0.0,
        link_time_s: float = 0.0,
        comm_time_s: float | None = None,
        n: float = 1.0,
    ) -> None:
        """Append expanded traffic and communication timing for n executions.
        traffic is multiplied by n before storage. link_time_s is transfer time from the
        bandwidth bottleneck; hop_time_s is accumulated switch-hop latency. comm_time_s
        defaults to link_time_s + hop_time_s. All times are in seconds.
        """
        eff_comm = (link_time_s + hop_time_s) if comm_time_s is None else comm_time_s
        self._traffic_mats.append(traffic * n if n != 1.0 else traffic.copy())
        self._comm_time_accum += eff_comm    * n
        self._hop_time_accum  += hop_time_s  * n
        self._link_time_accum += link_time_s * n

    def append_comm_time(
        self,
        link_time_s: float = 0.0,
        hop_time_s: float = 0.0,
        comm_time_s: float | None = None,
        n: float = 1.0,
    ) -> None:
        """Append communication timing without a traffic matrix, for example PP send/recv.
        link_time_s and hop_time_s are in seconds. comm_time_s defaults to their sum.
        n is the execution count.
        """
        eff_comm = (link_time_s + hop_time_s) if comm_time_s is None else comm_time_s
        self._comm_time_accum += eff_comm    * n
        self._hop_time_accum  += hop_time_s  * n
        self._link_time_accum += link_time_s * n

    def absorb(self, other: "OpPerfStats") -> None:
        """Merge another OpPerfStats object's internal accumulators into this object.
        This can inject KV-independent base statistics into per-KV statistics without
        accessing private fields. Only _hete_rows, _traffic_mats, and timing accumulators
        are merged; finalized result fields are not copied.
        """
        self._hete_rows.extend(other._hete_rows)
        self._traffic_mats.extend(other._traffic_mats)
        self._comm_time_accum += other._comm_time_accum
        self._hop_time_accum  += other._hop_time_accum
        self._link_time_accum += other._link_time_accum

    # ────────────────────────────────────────────────────────────────────────
    # Finalize
    # ────────────────────────────────────────────────────────────────────────

    def finalize(
        self,
        total_time_s: float,
        h: "Hierarchy | None" = None,
        energy_config: "NocEnergyConfig | None" = None,
        arch: "Arch | None" = None,
        num_devices: int = 1,
        chip_energy_config: "ChipEnergyConfig | None" = None,
    ) -> None:
        """Finalize all metrics using end-to-end total_time_s as the denominator.

        Utilization is sum(n_i*t_i*util_i)/total_time_s. L2 hit rate is
        1 - sum(n_i*ddr_read_io_i)/sum(n_i*l2_read_io_i). Footprints use the maximum;
        compute_time_s is sum(n_i*t_i). Traffic matrices are already scaled by n when
        appended, so they are summed directly. NoC utilization is traffic/time/bandwidth.
        Communication overlap is max(0, 1 - (e2e-compute)/comm)*100.

        Chip accounting reconstructs bytes and operations from utilization and effective
        time. DRAM/L2 reads use TileSight row[7]/row[8]; writes are total minus reads.
        Shared-memory/register traffic assumes equal reads and writes. Scale energy by
        h.num_devices when h is supplied, otherwise by the explicit num_devices.

        h supplies the NoC hierarchy; None skips NoC calculations. energy_config may
        override h.energy_config. arch supplies chip memory/compute parameters; None
        skips chip accounting. chip_energy_config may override arch.chip_energy_config.
        """
        T = float(total_time_s)
        self.e2e_time_s      = T
        self.comm_time_s     = self._comm_time_accum
        self.noc_hop_time_s  = self._hop_time_accum
        self.noc_link_time_s = self._link_time_accum

        # ── hete rows ───────────────────────────────────────────────────────
        if self._hete_rows:
            compute_time_sum   = 0.0
            ddr_util_sum       = 0.0
            l2_util_sum        = 0.0
            smem_l1_util_sum   = 0.0
            tensor_util_sum    = 0.0
            cuda_util_sum      = 0.0
            sfu_util_sum       = 0.0
            ddr_read_io_sum    = 0.0
            l2_read_io_sum     = 0.0
            smem_footprint_max = 0.0
            reg_footprint_max  = 0.0

            for row, n in self._hete_rows:
                t_eff = float(row[0]) * n
                compute_time_sum   += t_eff
                ddr_util_sum       += t_eff * float(row[1])
                l2_util_sum        += t_eff * float(row[3])
                smem_l1_util_sum   += t_eff * float(row[5])
                tensor_util_sum    += t_eff * float(row[9])
                cuda_util_sum      += t_eff * float(row[10])
                sfu_util_sum       += t_eff * float(row[11])
                ddr_read_io_sum    += float(row[7]) * n
                l2_read_io_sum     += float(row[8]) * n
                smem_footprint_max  = max(smem_footprint_max, float(row[4]))
                reg_footprint_max   = max(reg_footprint_max,  float(row[6]))

            self.compute_time_s       = compute_time_sum
            self.dram_util_pct        = ddr_util_sum     / T * 100
            self.l2_util_pct          = l2_util_sum      / T * 100
            self.smem_l1_util_pct     = smem_l1_util_sum / T * 100
            self.tensor_util_pct      = tensor_util_sum  / T * 100
            self.cuda_util_pct        = cuda_util_sum    / T * 100
            self.sfu_util_pct         = sfu_util_sum     / T * 100
            self.l2_hit_rate_pct      = (
                1.0 - max(ddr_read_io_sum, 1.0) / max(l2_read_io_sum, 1.0)
            ) * 100
            self.smem_bytes_per_tb    = smem_footprint_max
            self.reg_bytes_per_thread = reg_footprint_max

        # ── traffic mats → RouteStats ────────────────────────────────────────
        if h is not None:
            rs = build_route_stats_from_extended(self._traffic_mats, h, T, energy_config)
            self.route_stats          = rs
            self.noc_total_energy_j   = rs.total_noc_energy_pj * 1e-12
            self.noc_bottleneck_src   = rs.max_link_src
            self.noc_bottleneck_dst   = rs.max_link_dst
            self.noc_bottleneck_bytes = rs.max_link_bytes
            self.noc_bottleneck_bw    = rs.max_link_bw
            nz = rs.util[rs.util > 0]
            self.noc_max_util  = float(nz.max())  if nz.size > 0 else 0.0
            self.noc_mean_util = float(nz.mean()) if nz.size > 0 else 0.0
        elif self._traffic_mats:
            log.warning("OpPerfStats.finalize: traffic_mats present but h=None; NoC stats skipped.")

        if self._traffic_mats:
            self.noc_total_bytes = get_total_traffic_bytes(self._traffic_mats)

        # comm overlap: fraction of comm hidden by compute
        # = 1 - (e2e - compute) / comm; clamped to [0, 1]
        if self._comm_time_accum > 0:
            self.comm_overlap_pct = max(0.0, min(1.0,
                1.0 - (T - self.compute_time_s) / self._comm_time_accum
            )) * 100
        else:
            self.comm_overlap_pct = 100.0

        # ── chip energy ─────────────────────────────────────────────────────
        if arch is not None and self._hete_rows:
            nd = h.num_devices if h is not None else int(num_devices)
            self.chip_num_devices = nd
            self._calc_chip_energy(
                arch,
                nd,
                chip_energy_config=chip_energy_config,
            )

        if self.dump_perf_log:
            self.dump_log()

    # ────────────────────────────────────────────────────────────────────────
    # Chip energy accounting
    # ────────────────────────────────────────────────────────────────────────

    def _calc_chip_energy(
        self,
        arch: "Arch",
        num_devices: int,
        *,
        chip_energy_config: "ChipEnergyConfig | None" = None,
    ) -> None:
        """Reconstruct bytes and operations from _hete_rows and evaluate chip energy.
        For each segment and GPU, bytes = utilization * effective_time * bandwidth,
        and operations = utilization * effective_time * peak_flops.
        DRAM/L2 reads use row[7/8]*n; writes are max(0, reconstructed total - reads).
        Shared-memory traffic is smem_util*t_eff*smem_bandwidth, split equally into reads
        and writes. Register traffic uses tensor/CUDA/SFU operation counts and operand
        widths, also split equally. Tensor, CUDA, and SFU counts use their respective
        peak throughput. Multiply all resulting energies by num_devices.
        """
        from mosaic.cost.energy import EnergyModel
        em = EnergyModel(
            arch,
            chip_energy_config=chip_energy_config,
        )

        ddr_read  = ddr_write  = 0.0
        l2_read   = l2_write   = 0.0
        smem_io   = 0.0
        tensor_ops = cuda_ops = sfu_ops = 0.0

        for row, n in self._hete_rows:
            t_eff        = float(row[0]) * n
            ddr_util     = float(row[1])
            l2_util      = float(row[3])
            smem_util    = float(row[5])
            ddr_read_io  = float(row[7]) * n   # Exact read bytes from tilesight.
            l2_read_io   = float(row[8]) * n   # Exact read bytes from tilesight.
            t_util       = float(row[9])
            c_util       = float(row[10])
            s_util       = float(row[11])

            ddr_total     = ddr_util  * t_eff * arch.ddr_bandwidth
            ddr_read     += ddr_read_io
            ddr_write    += max(0.0, ddr_total - ddr_read_io)

            l2_total      = l2_util   * t_eff * arch.l2_bandwidth
            l2_read      += l2_read_io
            l2_write     += max(0.0, l2_total - l2_read_io)

            smem_io      += smem_util * t_eff * arch.smem_bandwidth

            tensor_ops   += t_util * t_eff * arch.fp16_tensor_flops
            cuda_ops     += c_util * t_eff * arch.fp32_cuda_core_flops
            sfu_ops      += s_util * t_eff * arch.sfu_flops

        # reg traffic: each op reads/writes 3/2 register operands.
        # reg_io = smem_io + tensor_ops * 2 * 2 + cuda_ops * 2 * 4 + sfu_ops * 2 * 4
        tc_m, tc_n, tc_k = arch.get_tensor_core_minimum_ptx()
        tc_regio_to_flops_ratio = (2*tc_m*tc_n+2*tc_m*tc_k+4*tc_n*tc_k)/(2*tc_m*tc_n*tc_k)
        # print(f"tc_regio_to_flops_ratio: {tc_regio_to_flops_ratio}")
        reg_io = smem_io + tensor_ops * tc_regio_to_flops_ratio + cuda_ops * 2 * 4 + sfu_ops * 2 * 4

        pj2j = 1e-12
        nd   = float(num_devices)

        dram_e   = em.get_dram_energy  ((ddr_read,      ddr_write))      * pj2j
        l2_e     = em.get_l2_energy    ((l2_read,        l2_write))       * pj2j
        smem_e   = em.get_smem_energy  ((smem_io * 0.5,  smem_io * 0.5)) * pj2j
        reg_e    = em.get_reg_energy   ((reg_io  * 0.5,  reg_io  * 0.5)) * pj2j
        tensor_e = em.get_tensor_core_energy(tensor_ops)                  * pj2j
        cuda_e   = em.get_cuda_core_energy  (cuda_ops)                    * pj2j
        sfu_e    = em.get_sfu_energy        (sfu_ops)                     * pj2j
        # Static power: static_power (W) × e2e_time (s), dependent only on elapsed time, regardless of the computation.
        static_e = em.get_static_power() * self.e2e_time_s

        self.chip_dram_energy_j    = dram_e   * nd
        self.chip_l2_energy_j      = l2_e     * nd
        self.chip_smem_energy_j    = smem_e   * nd
        self.chip_reg_energy_j     = reg_e    * nd
        self.chip_memory_energy_j  = (dram_e + l2_e + smem_e + reg_e) * nd
        self.chip_tensor_energy_j  = tensor_e * nd
        self.chip_cuda_energy_j    = cuda_e   * nd
        self.chip_sfu_energy_j     = sfu_e    * nd
        self.chip_compute_energy_j = (tensor_e + cuda_e + sfu_e) * nd
        self.chip_static_energy_j  = static_e * nd
        self.chip_total_energy_j   = self.chip_memory_energy_j + self.chip_compute_energy_j + self.chip_static_energy_j

    # ────────────────────────────────────────────────────────────────────────
    # Dump API
    # ────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a plain Python dictionary suitable for json.dump."""
        T = self.e2e_time_s if self.e2e_time_s > 0 else 1.0
        nd = float(self.chip_num_devices) if self.chip_num_devices > 0 else 1.0
        compute_pct = self.compute_time_s / T * 100
        comm_pct = self.comm_time_s / T * 100
        hop_pct = (self.noc_hop_time_s / self.comm_time_s * 100) if self.comm_time_s > 0 else 0.0
        link_pct = (self.noc_link_time_s / self.comm_time_s * 100) if self.comm_time_s > 0 else 0.0
        noc_power_w = self.noc_total_energy_j / T
        total_chip_noc = self.chip_total_energy_j + self.noc_total_energy_j
        noc_pct_vs_chip_noc = (self.noc_total_energy_j / total_chip_noc * 100) if total_chip_noc > 0 else 0.0
        chip_pct_vs_chip_noc = 100.0 - noc_pct_vs_chip_noc if total_chip_noc > 0 else 0.0

        def _w(j: float) -> float:
            return j / T

        def _pd(j: float) -> float:
            return j / nd

        def _pdw(j: float) -> float:
            return j / nd / T

        return {
            "op_name": self.op_name,
            "timing": {
                "compute_time_s":   self.compute_time_s,
                "comm_time_s":      self.comm_time_s,
                "e2e_time_s":       self.e2e_time_s,
                "comm_overlap_pct": self.comm_overlap_pct,
                "compute_pct": compute_pct,
                "comm_pct": comm_pct,
            },
            "memory_hierarchy": {
                "dram_util_pct":        self.dram_util_pct,
                "l2_util_pct":          self.l2_util_pct,
                "l2_hit_rate_pct":      self.l2_hit_rate_pct,
                "smem_l1_util_pct":     self.smem_l1_util_pct,
                "smem_bytes_per_tb":    self.smem_bytes_per_tb,
                "reg_bytes_per_thread": self.reg_bytes_per_thread,
            },
            "compute_util": {
                "tensor_util_pct": self.tensor_util_pct,
                "cuda_util_pct":   self.cuda_util_pct,
                "sfu_util_pct":    self.sfu_util_pct,
            },
            "noc": {
                "total_bytes":    self.noc_total_bytes,
                "total_energy_j": self.noc_total_energy_j,
                "max_util":       self.noc_max_util,
                "mean_util":      self.noc_mean_util,
                "comm_time_s":    self.comm_time_s,
                "hop_time_s":     self.noc_hop_time_s,
                "link_time_s":    self.noc_link_time_s,
                "comm_breakdown_pct": {
                    "hop_pct": hop_pct,
                    "link_pct": link_pct,
                },
                "power_w": noc_power_w,
                "energy_share_vs_chip_plus_noc_pct": {
                    "noc_pct": noc_pct_vs_chip_noc,
                    "chip_pct": chip_pct_vs_chip_noc,
                },
                "bottleneck": {
                    "src":   self.noc_bottleneck_src,
                    "dst":   self.noc_bottleneck_dst,
                    "bytes": self.noc_bottleneck_bytes,
                    "bw":    self.noc_bottleneck_bw,
                },
            },
            "chip_energy": {
                "num_devices": self.chip_num_devices,
                "total": {"j": self.chip_total_energy_j, "w": _w(self.chip_total_energy_j)},
                "memory": {"j": self.chip_memory_energy_j, "w": _w(self.chip_memory_energy_j)},
                "compute": {"j": self.chip_compute_energy_j, "w": _w(self.chip_compute_energy_j)},
                "static": {"j": self.chip_static_energy_j, "w": _w(self.chip_static_energy_j)},
                "per_device": {
                    "total": {"j": _pd(self.chip_total_energy_j), "w": _pdw(self.chip_total_energy_j)},
                    "memory": {"j": _pd(self.chip_memory_energy_j), "w": _pdw(self.chip_memory_energy_j)},
                    "compute": {"j": _pd(self.chip_compute_energy_j), "w": _pdw(self.chip_compute_energy_j)},
                    "static": {"j": _pd(self.chip_static_energy_j), "w": _pdw(self.chip_static_energy_j)},
                },
                "memory_breakdown": {
                    "total": {
                        "dram": {"j": self.chip_dram_energy_j, "w": _w(self.chip_dram_energy_j)},
                        "l2": {"j": self.chip_l2_energy_j, "w": _w(self.chip_l2_energy_j)},
                        "smem": {"j": self.chip_smem_energy_j, "w": _w(self.chip_smem_energy_j)},
                        "reg": {"j": self.chip_reg_energy_j, "w": _w(self.chip_reg_energy_j)},
                    },
                    "per_device": {
                        "dram": {"j": _pd(self.chip_dram_energy_j), "w": _pdw(self.chip_dram_energy_j)},
                        "l2": {"j": _pd(self.chip_l2_energy_j), "w": _pdw(self.chip_l2_energy_j)},
                        "smem": {"j": _pd(self.chip_smem_energy_j), "w": _pdw(self.chip_smem_energy_j)},
                        "reg": {"j": _pd(self.chip_reg_energy_j), "w": _pdw(self.chip_reg_energy_j)},
                    },
                },
                "compute_breakdown": {
                    "total": {
                        "tensor": {"j": self.chip_tensor_energy_j, "w": _w(self.chip_tensor_energy_j)},
                        "cuda": {"j": self.chip_cuda_energy_j, "w": _w(self.chip_cuda_energy_j)},
                        "sfu": {"j": self.chip_sfu_energy_j, "w": _w(self.chip_sfu_energy_j)},
                    },
                    "per_device": {
                        "tensor": {"j": _pd(self.chip_tensor_energy_j), "w": _pdw(self.chip_tensor_energy_j)},
                        "cuda": {"j": _pd(self.chip_cuda_energy_j), "w": _pdw(self.chip_cuda_energy_j)},
                        "sfu": {"j": _pd(self.chip_sfu_energy_j), "w": _pdw(self.chip_sfu_energy_j)},
                    },
                },
            },
        }

    def _build_detail_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"=== OpPerfStats [{self.op_name}] ===")
        T_pct = self.e2e_time_s if self.e2e_time_s > 0 else 1.0
        compute_pct = self.compute_time_s / T_pct * 100
        comm_pct = self.comm_time_s / T_pct * 100
        lines.append(
            "  [Timing]  compute=%.3e s (%.1f%%)   comm=%.3e s (%.1f%%)   e2e=%.3e s   comm_overlap=%.1f%%"
            % (self.compute_time_s, compute_pct, self.comm_time_s, comm_pct, self.e2e_time_s, self.comm_overlap_pct)
        )
        lines.append("  [Memory Hierarchy]")
        lines.append(
            "    DRAM util=%.1f%%   L2 util=%.1f%%   L2 hit=%.1f%%"
            % (self.dram_util_pct, self.l2_util_pct, self.l2_hit_rate_pct)
        )
        lines.append(
            "    SMEM/L1 util=%.1f%%   smem_per_tb=%.0f B   reg_per_thread=%.0f B"
            % (self.smem_l1_util_pct, self.smem_bytes_per_tb, self.reg_bytes_per_thread)
        )
        lines.append("  [Compute Utilization]")
        lines.append(
            "    tensor=%.1f%%   cuda=%.1f%%   sfu=%.1f%%"
            % (self.tensor_util_pct, self.cuda_util_pct, self.sfu_util_pct)
        )
        lines.append("  [NoC Traffic]")
        lines.append(
            "    total_bytes=%.3e   energy=%.3e J   max_util=%.3f   mean_util=%.3f"
            % (self.noc_total_bytes, self.noc_total_energy_j, self.noc_max_util, self.noc_mean_util)
        )
        if self.comm_time_s > 0:
            hop_pct = self.noc_hop_time_s / self.comm_time_s * 100
            link_pct = self.noc_link_time_s / self.comm_time_s * 100
            lines.append(
                "    comm: total=%.3e s   hop=%.3e s (%.1f%%)   link=%.3e s (%.1f%%)"
                % (self.comm_time_s, self.noc_hop_time_s, hop_pct, self.noc_link_time_s, link_pct)
            )
        if self.noc_bottleneck_src >= 0:
            lines.append(
                "    bottleneck [%d->%d] %.3e B / %.3e B/s"
                % (self.noc_bottleneck_src, self.noc_bottleneck_dst, self.noc_bottleneck_bytes, self.noc_bottleneck_bw)
            )
        if self.noc_total_energy_j > 0:
            T_noc = self.e2e_time_s if self.e2e_time_s > 0 else 1.0
            lines.append("  [NoC Energy]  total=%.3e J (%.1f W)" % (self.noc_total_energy_j, self.noc_total_energy_j / T_noc))
            if self.chip_total_energy_j > 0:
                noc_pct = self.noc_total_energy_j / (self.chip_total_energy_j + self.noc_total_energy_j) * 100
                lines.append("    vs chip+noc: noc=%.1f%%   chip=%.1f%%" % (noc_pct, 100.0 - noc_pct))
        if self.chip_total_energy_j > 0:
            T = self.e2e_time_s if self.e2e_time_s > 0 else 1.0
            nd = float(self.chip_num_devices) if self.chip_num_devices > 0 else 1.0
            def _w(j: float) -> float: return j / T
            def _pd(j: float) -> float: return j / nd
            def _pdw(j: float) -> float: return j / nd / T
            lines.append(
                "  [Chip Energy ×%d]  total=%.3e J (%.1f W)   memory=%.3e J (%.1f W)   compute=%.3e J (%.1f W)   static=%.3e J (%.1f W)"
                % (self.chip_num_devices, self.chip_total_energy_j, _w(self.chip_total_energy_j),
                   self.chip_memory_energy_j, _w(self.chip_memory_energy_j),
                   self.chip_compute_energy_j, _w(self.chip_compute_energy_j),
                   self.chip_static_energy_j, _w(self.chip_static_energy_j))
            )
            lines.append(
                "    per-device:    total=%.3e J (%.1f W)   memory=%.3e J (%.1f W)   compute=%.3e J (%.1f W)   static=%.3e J (%.1f W)"
                % (_pd(self.chip_total_energy_j), _pdw(self.chip_total_energy_j),
                   _pd(self.chip_memory_energy_j), _pdw(self.chip_memory_energy_j),
                   _pd(self.chip_compute_energy_j), _pdw(self.chip_compute_energy_j),
                   _pd(self.chip_static_energy_j), _pdw(self.chip_static_energy_j))
            )
            lines.append(
                "    mem per device: DRAM=%.3e J (%.1f W)   L2=%.3e J (%.1f W)   smem=%.3e J (%.1f W)   reg=%.3e J (%.1f W)"
                % (_pd(self.chip_dram_energy_j), _pdw(self.chip_dram_energy_j),
                   _pd(self.chip_l2_energy_j), _pdw(self.chip_l2_energy_j),
                   _pd(self.chip_smem_energy_j), _pdw(self.chip_smem_energy_j),
                   _pd(self.chip_reg_energy_j), _pdw(self.chip_reg_energy_j))
            )
            lines.append(
                "    compute per device: tensor=%.3e J (%.1f W)   cuda=%.3e J (%.1f W)   sfu=%.3e J (%.1f W)"
                % (_pd(self.chip_tensor_energy_j), _pdw(self.chip_tensor_energy_j),
                   _pd(self.chip_cuda_energy_j), _pdw(self.chip_cuda_energy_j),
                   _pd(self.chip_sfu_energy_j), _pdw(self.chip_sfu_energy_j))
            )
            lines.append(
                "    memory:  DRAM=%.3e J (%.1f W)   L2=%.3e J (%.1f W)   smem=%.3e J (%.1f W)   reg=%.3e J (%.1f W)"
                % (self.chip_dram_energy_j, _w(self.chip_dram_energy_j),
                   self.chip_l2_energy_j, _w(self.chip_l2_energy_j),
                   self.chip_smem_energy_j, _w(self.chip_smem_energy_j),
                   self.chip_reg_energy_j, _w(self.chip_reg_energy_j))
            )
            lines.append(
                "    compute: tensor=%.3e J (%.1f W)   cuda=%.3e J (%.1f W)   sfu=%.3e J (%.1f W)"
                % (self.chip_tensor_energy_j, _w(self.chip_tensor_energy_j),
                   self.chip_cuda_energy_j, _w(self.chip_cuda_energy_j),
                   self.chip_sfu_energy_j, _w(self.chip_sfu_energy_j))
            )
        return lines

    @staticmethod
    def _render_table(headers: list[str], rows: list[list[str]], markdown: bool) -> str:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        if markdown:
            head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
            sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
            body = [
                "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
                for row in rows
            ]
            return "\n".join([head, sep] + body)

        border = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        body = [
            "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
            for row in rows
        ]
        return "\n".join([border, head, border] + body + [border])

    def render_human(self, markdown: bool = False, title_level: int = 2) -> str:
        """Render human-readable tabular text with the same information as dump_log."""
        T = self.e2e_time_s if self.e2e_time_s > 0 else 1.0
        nd = float(self.chip_num_devices) if self.chip_num_devices > 0 else 1.0
        compute_pct = self.compute_time_s / T * 100
        comm_pct = self.comm_time_s / T * 100
        hop_pct = (self.noc_hop_time_s / self.comm_time_s * 100) if self.comm_time_s > 0 else 0.0
        link_pct = (self.noc_link_time_s / self.comm_time_s * 100) if self.comm_time_s > 0 else 0.0
        noc_power_w = self.noc_total_energy_j / T
        total_chip_noc = self.chip_total_energy_j + self.noc_total_energy_j
        noc_pct = (self.noc_total_energy_j / total_chip_noc * 100) if total_chip_noc > 0 else 0.0
        chip_pct = 100.0 - noc_pct if total_chip_noc > 0 else 0.0

        def _w(j: float) -> float:
            return j / T

        def _pd(j: float) -> float:
            return j / nd

        def _pdw(j: float) -> float:
            return j / nd / T

        blocks: list[str] = []
        if markdown:
            level = max(1, int(title_level))
            blocks.append(f"{'#' * level} OpPerfStats [{self.op_name}]")
        else:
            blocks.append(f"=== OpPerfStats [{self.op_name}] ===")

        blocks.append("[Timing]")
        blocks.append(self._render_table(
            ["metric", "value"],
            [
                ["compute", f"{self.compute_time_s:.3e} s ({compute_pct:.1f}%)"],
                ["comm", f"{self.comm_time_s:.3e} s ({comm_pct:.1f}%)"],
                ["e2e", f"{self.e2e_time_s:.3e} s"],
                ["comm_overlap", f"{self.comm_overlap_pct:.1f}%"],
            ],
            markdown,
        ))

        blocks.append("[Memory Hierarchy]")
        blocks.append(self._render_table(
            ["metric", "value"],
            [
                ["DRAM util", f"{self.dram_util_pct:.1f}%"],
                ["L2 util", f"{self.l2_util_pct:.1f}%"],
                ["L2 hit", f"{self.l2_hit_rate_pct:.1f}%"],
                ["SMEM/L1 util", f"{self.smem_l1_util_pct:.1f}%"],
                ["smem_per_tb", f"{self.smem_bytes_per_tb:.0f} B"],
                ["reg_per_thread", f"{self.reg_bytes_per_thread:.0f} B"],
            ],
            markdown,
        ))

        blocks.append("[Compute Utilization]")
        blocks.append(self._render_table(
            ["metric", "value"],
            [
                ["tensor", f"{self.tensor_util_pct:.1f}%"],
                ["cuda", f"{self.cuda_util_pct:.1f}%"],
                ["sfu", f"{self.sfu_util_pct:.1f}%"],
            ],
            markdown,
        ))

        blocks.append("[NoC Traffic]")
        noc_rows = [
            ["total_bytes", f"{self.noc_total_bytes:.3e} B"],
            ["energy", f"{self.noc_total_energy_j:.3e} J"],
            ["max_util", f"{self.noc_max_util:.3f}"],
            ["mean_util", f"{self.noc_mean_util:.3f}"],
        ]
        if self.comm_time_s > 0:
            noc_rows.extend([
                ["comm_total", f"{self.comm_time_s:.3e} s"],
                ["hop", f"{self.noc_hop_time_s:.3e} s ({hop_pct:.1f}%)"],
                ["link", f"{self.noc_link_time_s:.3e} s ({link_pct:.1f}%)"],
            ])
        if self.noc_bottleneck_src >= 0:
            noc_rows.append(["bottleneck", f"[{self.noc_bottleneck_src}->{self.noc_bottleneck_dst}] {self.noc_bottleneck_bytes:.3e} B / {self.noc_bottleneck_bw:.3e} B/s"])
        blocks.append(self._render_table(["metric", "value"], noc_rows, markdown))

        if self.noc_total_energy_j > 0:
            blocks.append("[NoC Energy]")
            blocks.append(self._render_table(
                ["metric", "value"],
                [
                    ["total", f"{self.noc_total_energy_j:.3e} J ({noc_power_w:.1f} W)"],
                    ["vs chip+noc", f"noc={noc_pct:.1f}%   chip={chip_pct:.1f}%"],
                ],
                markdown,
            ))

        if self.chip_total_energy_j > 0:
            blocks.append(f"[Chip Energy x{self.chip_num_devices}]")
            blocks.append(self._render_table(
                ["scope", "total", "memory", "compute", "static"],
                [
                    [
                        "all devices",
                        f"{self.chip_total_energy_j:.3e} J ({_w(self.chip_total_energy_j):.1f} W)",
                        f"{self.chip_memory_energy_j:.3e} J ({_w(self.chip_memory_energy_j):.1f} W)",
                        f"{self.chip_compute_energy_j:.3e} J ({_w(self.chip_compute_energy_j):.1f} W)",
                        f"{self.chip_static_energy_j:.3e} J ({_w(self.chip_static_energy_j):.1f} W)",
                    ],
                    [
                        "per-device",
                        f"{_pd(self.chip_total_energy_j):.3e} J ({_pdw(self.chip_total_energy_j):.1f} W)",
                        f"{_pd(self.chip_memory_energy_j):.3e} J ({_pdw(self.chip_memory_energy_j):.1f} W)",
                        f"{_pd(self.chip_compute_energy_j):.3e} J ({_pdw(self.chip_compute_energy_j):.1f} W)",
                        f"{_pd(self.chip_static_energy_j):.3e} J ({_pdw(self.chip_static_energy_j):.1f} W)",
                    ],
                ],
                markdown,
            ))
            blocks.append(self._render_table(
                ["memory part", "all devices", "per-device"],
                [
                    ["DRAM", f"{self.chip_dram_energy_j:.3e} J ({_w(self.chip_dram_energy_j):.1f} W)", f"{_pd(self.chip_dram_energy_j):.3e} J ({_pdw(self.chip_dram_energy_j):.1f} W)"],
                    ["L2", f"{self.chip_l2_energy_j:.3e} J ({_w(self.chip_l2_energy_j):.1f} W)", f"{_pd(self.chip_l2_energy_j):.3e} J ({_pdw(self.chip_l2_energy_j):.1f} W)"],
                    ["smem", f"{self.chip_smem_energy_j:.3e} J ({_w(self.chip_smem_energy_j):.1f} W)", f"{_pd(self.chip_smem_energy_j):.3e} J ({_pdw(self.chip_smem_energy_j):.1f} W)"],
                    ["reg", f"{self.chip_reg_energy_j:.3e} J ({_w(self.chip_reg_energy_j):.1f} W)", f"{_pd(self.chip_reg_energy_j):.3e} J ({_pdw(self.chip_reg_energy_j):.1f} W)"],
                ],
                markdown,
            ))
            blocks.append(self._render_table(
                ["compute part", "all devices", "per-device"],
                [
                    ["tensor", f"{self.chip_tensor_energy_j:.3e} J ({_w(self.chip_tensor_energy_j):.1f} W)", f"{_pd(self.chip_tensor_energy_j):.3e} J ({_pdw(self.chip_tensor_energy_j):.1f} W)"],
                    ["cuda", f"{self.chip_cuda_energy_j:.3e} J ({_w(self.chip_cuda_energy_j):.1f} W)", f"{_pd(self.chip_cuda_energy_j):.3e} J ({_pdw(self.chip_cuda_energy_j):.1f} W)"],
                    ["sfu", f"{self.chip_sfu_energy_j:.3e} J ({_w(self.chip_sfu_energy_j):.1f} W)", f"{_pd(self.chip_sfu_energy_j):.3e} J ({_pdw(self.chip_sfu_energy_j):.1f} W)"],
                ],
                markdown,
            ))

        return "\n\n".join(blocks)

    def dump_log(self) -> None:
        """Log all hierarchical statistics using log.info."""
        for line in self._build_detail_lines():
            log.info("%s", line)

    def dump_json(self, path: str) -> None:
        """Serialize the statistics to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("OpPerfStats [%s] saved to %s", self.op_name, path)

    def dump_human(self, path: str, markdown: bool = False) -> None:
        """Write human-readable text or Markdown output."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render_human(markdown=markdown))
            f.write("\n")
        log.info("OpPerfStats [%s] human-readable dump saved to %s", self.op_name, path)
