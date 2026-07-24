from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mosaic.noc.route_stats import build_route_stats_from_extended

if TYPE_CHECKING:
    from mosaic.noc.noc_topo import Hierarchy
    from mosaic.noc.energy_config import NocEnergyConfig
    from mosaic.cost.op_perf_stats import OpPerfStats

log = logging.getLogger(__name__)


@dataclass
class ModelPerfStats:
    """
    多个 OpPerfStats 的加权聚合，代表一个完整模型/端到端配置的性能统计。

    用法：
        model_stats = ModelPerfStats(name="llama3-70b")
        model_stats.add(mlp_stats,  n=32)   # 32 层 MLP
        model_stats.add(attn_stats, n=32)   # 32 层 Attention
        model_stats.add(pp_stats,   n=1)    # 1 次 PP 开销

        model_stats.finalize(h=hierarchy)   # 传 h 则重新从聚合 traffic 精确计算 NoC 指标
        model_stats.dump_log()

    加权规则：
      总时间          : Σ(n_i × e2e_i)
      util / overlap  : 时间加权平均 Σ(n_i × e2e_i × metric_i) / Σ(n_i × e2e_i)
      energy / bytes  : 线性求和 Σ(n_i × metric_i)
      footprints      : max（不随 n 变化）
      NoC bottleneck  : 聚合 n_i × traffic_mats_i 后以 T_total 为分母重新计算
                        （需要传 h；否则退化为时间加权近似值）
    """

    name: str = ""
    dump_perf_log: bool = False

    # ── Aggregated Timing ────────────────────────────────────────────────────
    total_time_s: float = 0.0         # Σ(n_i × e2e_i)
    compute_time_s: float = 0.0       # Σ(n_i × compute_i)
    comm_time_s: float = 0.0          # Σ(n_i × comm_i)
    comm_overlap_pct: float = 0.0     # 时间加权平均

    # ── Aggregated Memory Hierarchy ──────────────────────────────────────────
    dram_util_pct: float = 0.0
    l2_util_pct: float = 0.0
    l2_hit_rate_pct: float = 0.0
    smem_l1_util_pct: float = 0.0
    smem_bytes_per_tb: float = 0.0        # max（最大 smem 占用）
    reg_bytes_per_thread: float = 0.0     # max（最大 reg 占用）

    # ── Aggregated Compute Utilization ───────────────────────────────────────
    tensor_util_pct: float = 0.0
    cuda_util_pct: float = 0.0
    sfu_util_pct: float = 0.0

    # ── Aggregated NoC ───────────────────────────────────────────────────────
    noc_total_bytes: float = 0.0
    noc_total_energy_j: float = 0.0
    noc_max_util: float = 0.0
    noc_mean_util: float = 0.0
    noc_bottleneck_src: int = -1
    noc_bottleneck_dst: int = -1
    noc_bottleneck_bytes: float = 0.0
    noc_bottleneck_bw: float = 0.0
    noc_hop_time_s: float = 0.0           # Σ(n_i × hop_i)
    noc_link_time_s: float = 0.0          # Σ(n_i × link_i)

    # ── Aggregated Chip Energy — Σ(n_i × chip_*_energy_j) ───────────────────
    chip_dram_energy_j:    float = 0.0
    chip_l2_energy_j:      float = 0.0
    chip_smem_energy_j:    float = 0.0
    chip_reg_energy_j:     float = 0.0
    chip_memory_energy_j:  float = 0.0
    chip_tensor_energy_j:  float = 0.0
    chip_cuda_energy_j:    float = 0.0
    chip_sfu_energy_j:     float = 0.0
    chip_compute_energy_j: float = 0.0
    chip_static_energy_j:  float = 0.0
    chip_total_energy_j:   float = 0.0
    chip_num_devices:      int   = 1      # 从 entries 推断的设备数

    # ── 内部条目列表（不出现在 __init__ 签名中）──────────────────────────────
    # _entries: list of (n: float, OpPerfStats)
    _entries: list = field(default_factory=list, init=False, repr=False)

    # ────────────────────────────────────────────────────────────────────────
    # Add API
    # ────────────────────────────────────────────────────────────────────────

    def add(self, op_stats: "OpPerfStats", n: float = 1.0) -> None:
        """
        注册一个算子/stage 的统计, n 为在完整模型中出现的次数。

        op_stats 必须已经调用过 finalize()。
        n 可以是整数或浮点数（例如 PP 时某些 stage 权重可能不是整数）。
        """
        self._entries.append((float(n), op_stats))

    # ────────────────────────────────────────────────────────────────────────
    # Finalize
    # ────────────────────────────────────────────────────────────────────────

    def finalize(
        self,
        h: "Hierarchy | None" = None,
        energy_config: "NocEnergyConfig | None" = None,
    ) -> None:
        """
        聚合所有注册的 OpPerfStats,计算模型级性能统计。

        h             : 传入时从聚合 traffic 矩阵精确计算 NoC 利用率/能耗/瓶颈。
                        不传则 NoC util/bottleneck 退化为时间加权近似；
                        energy 仍为 Σ(n_i x energy_i)（精确）。
        energy_config : 可选，覆盖 h.energy_config。
        """
        if not self._entries:
            log.warning("ModelPerfStats.finalize: no entries registered, nothing to do.")
            return

        T = sum(n * s.e2e_time_s for n, s in self._entries)
        self.total_time_s    = T
        self.compute_time_s  = sum(n * s.compute_time_s  for n, s in self._entries)
        self.comm_time_s     = sum(n * s.comm_time_s     for n, s in self._entries)
        self.noc_hop_time_s  = sum(n * s.noc_hop_time_s  for n, s in self._entries)
        self.noc_link_time_s = sum(n * s.noc_link_time_s for n, s in self._entries)
        self.noc_total_bytes = sum(n * s.noc_total_bytes for n, s in self._entries)

        # footprints: max（最坏情况）
        self.smem_bytes_per_tb    = max((s.smem_bytes_per_tb    for _, s in self._entries), default=0.0)
        self.reg_bytes_per_thread = max((s.reg_bytes_per_thread for _, s in self._entries), default=0.0)

        # comm_overlap: 模型级（以总 comm/compute/e2e 重新计算，与 OpPerfStats 语义一致）
        if self.comm_time_s > 0:
            self.comm_overlap_pct = max(0.0, min(1.0,
                1.0 - (T - self.compute_time_s) / self.comm_time_s
            )) * 100
        else:
            self.comm_overlap_pct = 100.0

        # 时间加权平均 util
        if T > 0:
            def _tw(attr: str) -> float:
                return sum(n * s.e2e_time_s * getattr(s, attr)
                           for n, s in self._entries) / T

            self.dram_util_pct    = _tw("dram_util_pct")
            self.l2_util_pct      = _tw("l2_util_pct")
            self.l2_hit_rate_pct  = _tw("l2_hit_rate_pct")
            self.smem_l1_util_pct = _tw("smem_l1_util_pct")
            self.tensor_util_pct  = _tw("tensor_util_pct")
            self.cuda_util_pct    = _tw("cuda_util_pct")
            self.sfu_util_pct     = _tw("sfu_util_pct")
            # NoC util 先用时间加权近似；如果有 h 下面会精确覆盖
            self.noc_max_util     = _tw("noc_max_util")
            self.noc_mean_util    = _tw("noc_mean_util")

        # NoC：从聚合 traffic 矩阵精确计算（需要 h）
        if h is not None:
            agg_mats = []
            for n_i, s in self._entries:
                for mat in s._traffic_mats:
                    agg_mats.append(mat * n_i if n_i != 1.0 else mat.copy())

            if agg_mats:
                rs = build_route_stats_from_extended(agg_mats, h, T, energy_config)
                self.noc_total_energy_j   = rs.total_noc_energy_pj * 1e-12
                self.noc_bottleneck_src   = rs.max_link_src
                self.noc_bottleneck_dst   = rs.max_link_dst
                self.noc_bottleneck_bytes = rs.max_link_bytes
                self.noc_bottleneck_bw    = rs.max_link_bw
                nz = rs.util[rs.util > 0]
                self.noc_max_util  = float(nz.max())  if nz.size > 0 else 0.0
                self.noc_mean_util = float(nz.mean()) if nz.size > 0 else 0.0
            else:
                # 无 traffic 矩阵，energy 退化为求和
                self.noc_total_energy_j = sum(n * s.noc_total_energy_j
                                              for n, s in self._entries)
        else:
            # 无 h：energy 线性求和（精确）；util/bottleneck 保持时间加权近似
            self.noc_total_energy_j = sum(n * s.noc_total_energy_j
                                          for n, s in self._entries)
            if self._entries:
                # bottleneck: 取 noc_max_util 最大的那个 op 的 bottleneck
                best = max(self._entries, key=lambda x: x[1].noc_max_util)
                bs = best[1]
                self.noc_bottleneck_src   = bs.noc_bottleneck_src
                self.noc_bottleneck_dst   = bs.noc_bottleneck_dst
                self.noc_bottleneck_bytes = bs.noc_bottleneck_bytes
                self.noc_bottleneck_bw    = bs.noc_bottleneck_bw

        # chip energy: 线性求和 Σ(n_i × chip_*_j)
        def _esum(attr: str) -> float:
            return sum(n * getattr(s, attr) for n, s in self._entries)

        self.chip_dram_energy_j    = _esum("chip_dram_energy_j")
        self.chip_l2_energy_j      = _esum("chip_l2_energy_j")
        self.chip_smem_energy_j    = _esum("chip_smem_energy_j")
        self.chip_reg_energy_j     = _esum("chip_reg_energy_j")
        self.chip_memory_energy_j  = _esum("chip_memory_energy_j")
        self.chip_tensor_energy_j  = _esum("chip_tensor_energy_j")
        self.chip_cuda_energy_j    = _esum("chip_cuda_energy_j")
        self.chip_sfu_energy_j     = _esum("chip_sfu_energy_j")
        self.chip_compute_energy_j = _esum("chip_compute_energy_j")
        self.chip_static_energy_j  = _esum("chip_static_energy_j")
        self.chip_total_energy_j   = _esum("chip_total_energy_j")
        # 取所有 entry 中 chip_num_devices 的最大值（应该都相同）
        self.chip_num_devices = max((s.chip_num_devices for _, s in self._entries), default=1)

        if self.dump_perf_log:
            self.dump_log()

    # ────────────────────────────────────────────────────────────────────────
    # Query API
    # ────────────────────────────────────────────────────────────────────────

    def breakdown(self) -> list[tuple[float, "OpPerfStats"]]:
        """返回 (n, op_stats) 列表，按 n x e2e_time 降序排列（最耗时的 op 在前）。"""
        return sorted(self._entries, key=lambda x: x[0] * x[1].e2e_time_s, reverse=True)

    def breakdown_by_name(self) -> list[dict]:
        """
        按 op_name 聚类，返回每个 op_name 的汇总信息，按总耗时降序排列。

        每条记录包含：
          op_name      : str
          n_total      : float  — Σ n_i(该 op_name 的所有 entry 的出现次数之和）
          time_total_s : float  — Σ(n_i x e2e_i)
          time_pct     : float  — 占模型总时间的百分比
          compute_s    : float  — Σ(n_i x compute_i)
          comm_s       : float  — Σ(n_i x comm_i)
          noc_energy_j : float  — Σ(n_i x noc_energy_i)
        """
        from collections import defaultdict
        groups: dict = defaultdict(lambda: {
            "n_total": 0.0, "time_total_s": 0.0,
            "compute_s": 0.0, "comm_s": 0.0, "noc_energy_j": 0.0,
            "chip_total_energy_j": 0.0, "chip_memory_energy_j": 0.0,
            "chip_compute_energy_j": 0.0, "chip_static_energy_j": 0.0,
            "chip_dram_energy_j": 0.0, "chip_l2_energy_j": 0.0,
            "chip_smem_energy_j": 0.0, "chip_reg_energy_j": 0.0,
            "chip_tensor_energy_j": 0.0, "chip_cuda_energy_j": 0.0,
            "chip_sfu_energy_j": 0.0,
        })
        for n, s in self._entries:
            g = groups[s.op_name]
            g["n_total"]               += n
            g["time_total_s"]          += n * s.e2e_time_s
            g["compute_s"]             += n * s.compute_time_s
            g["comm_s"]                += n * s.comm_time_s
            g["noc_energy_j"]          += n * s.noc_total_energy_j
            g["chip_total_energy_j"]   += n * s.chip_total_energy_j
            g["chip_memory_energy_j"]  += n * s.chip_memory_energy_j
            g["chip_compute_energy_j"] += n * s.chip_compute_energy_j
            g["chip_static_energy_j"]  += n * s.chip_static_energy_j
            g["chip_dram_energy_j"]    += n * s.chip_dram_energy_j
            g["chip_l2_energy_j"]      += n * s.chip_l2_energy_j
            g["chip_smem_energy_j"]    += n * s.chip_smem_energy_j
            g["chip_reg_energy_j"]     += n * s.chip_reg_energy_j
            g["chip_tensor_energy_j"]  += n * s.chip_tensor_energy_j
            g["chip_cuda_energy_j"]    += n * s.chip_cuda_energy_j
            g["chip_sfu_energy_j"]     += n * s.chip_sfu_energy_j

        T = self.total_time_s if self.total_time_s > 0 else 1.0
        result = []
        for op_name, g in groups.items():
            result.append({
                "op_name":               op_name,
                "n_total":               g["n_total"],
                "time_total_s":          g["time_total_s"],
                "time_pct":              g["time_total_s"] / T * 100,
                "compute_s":             g["compute_s"],
                "comm_s":                g["comm_s"],
                "noc_energy_j":          g["noc_energy_j"],
                "chip_total_energy_j":   g["chip_total_energy_j"],
                "chip_memory_energy_j":  g["chip_memory_energy_j"],
                "chip_compute_energy_j": g["chip_compute_energy_j"],
                "chip_static_energy_j":  g["chip_static_energy_j"],
                "chip_dram_energy_j":    g["chip_dram_energy_j"],
                "chip_l2_energy_j":      g["chip_l2_energy_j"],
                "chip_smem_energy_j":    g["chip_smem_energy_j"],
                "chip_reg_energy_j":     g["chip_reg_energy_j"],
                "chip_tensor_energy_j":  g["chip_tensor_energy_j"],
                "chip_cuda_energy_j":    g["chip_cuda_energy_j"],
                "chip_sfu_energy_j":     g["chip_sfu_energy_j"],
            })
        result.sort(key=lambda x: x["time_total_s"], reverse=True)
        return result

    # ────────────────────────────────────────────────────────────────────────
    # Dump API
    # ────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """序列化为纯 Python dict(可直接传给 json.dump)"""
        T = self.total_time_s if self.total_time_s > 0 else 1.0
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
            "name": self.name,
            "timing": {
                "total_time_s":    self.total_time_s,
                "compute_time_s":  self.compute_time_s,
                "comm_time_s":     self.comm_time_s,
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
            "breakdown": [
                {"n": n, **s.to_dict()}
                for n, s in self.breakdown()
            ],
            "op_entries_in_order": [
                {"entry_idx": idx, "n": n, "op_stats": s.to_dict()}
                for idx, (n, s) in enumerate(self._entries)
            ],
            "breakdown_by_name": self.breakdown_by_name(),
        }

    def _build_detail_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"=== ModelPerfStats [{self.name}] ===")
        T_pct = self.total_time_s if self.total_time_s > 0 else 1.0
        compute_pct = self.compute_time_s / T_pct * 100
        comm_pct = self.comm_time_s / T_pct * 100
        lines.append(
            "  [Total Timing]  total=%.3e s   compute=%.3e s (%.1f%%)   comm=%.3e s (%.1f%%)   comm_overlap=%.1f%%"
            % (self.total_time_s, self.compute_time_s, compute_pct, self.comm_time_s, comm_pct, self.comm_overlap_pct)
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
            T_noc = self.total_time_s if self.total_time_s > 0 else 1.0
            lines.append("  [NoC Energy]  total=%.3e J (%.1f W)" % (self.noc_total_energy_j, self.noc_total_energy_j / T_noc))
            if self.chip_total_energy_j > 0:
                noc_pct = self.noc_total_energy_j / (self.chip_total_energy_j + self.noc_total_energy_j) * 100
                lines.append("    vs chip+noc: noc=%.1f%%   chip=%.1f%%" % (noc_pct, 100.0 - noc_pct))
        if self.chip_total_energy_j > 0:
            T = self.total_time_s if self.total_time_s > 0 else 1.0
            nd = float(self.chip_num_devices) if self.chip_num_devices > 0 else 1.0
            def _w(j: float) -> float: return j / T
            def _pd(j: float) -> float: return j / nd
            def _pdw(j: float) -> float: return j / nd / T
            lines.append(
                "  [Chip Energy ×%d]  total=%.3e J (%.1f W)   memory=%.3e J (%.1f W)   compute=%.3e J (%.1f W)   static=%.3e J (%.1f W)"
                % (self.chip_num_devices,
                   self.chip_total_energy_j, _w(self.chip_total_energy_j),
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
                "    memory:  DRAM=%.3e J (%.1f W)   L2=%.3e J (%.1f W)   smem=%.3e J (%.1f W)   reg=%.3e J (%.1f W)"
                % (self.chip_dram_energy_j, _w(self.chip_dram_energy_j),
                   self.chip_l2_energy_j, _w(self.chip_l2_energy_j),
                   self.chip_smem_energy_j, _w(self.chip_smem_energy_j),
                   self.chip_reg_energy_j, _w(self.chip_reg_energy_j))
            )
            lines.append(
                "    mem/dev: DRAM=%.3e J (%.1f W)   L2=%.3e J (%.1f W)   smem=%.3e J (%.1f W)   reg=%.3e J (%.1f W)"
                % (_pd(self.chip_dram_energy_j), _pdw(self.chip_dram_energy_j),
                   _pd(self.chip_l2_energy_j), _pdw(self.chip_l2_energy_j),
                   _pd(self.chip_smem_energy_j), _pdw(self.chip_smem_energy_j),
                   _pd(self.chip_reg_energy_j), _pdw(self.chip_reg_energy_j))
            )
            lines.append(
                "    compute: tensor=%.3e J (%.1f W)   cuda=%.3e J (%.1f W)   sfu=%.3e J (%.1f W)"
                % (self.chip_tensor_energy_j, _w(self.chip_tensor_energy_j),
                   self.chip_cuda_energy_j, _w(self.chip_cuda_energy_j),
                   self.chip_sfu_energy_j, _w(self.chip_sfu_energy_j))
            )
            lines.append(
                "    cmp/dev: tensor=%.3e J (%.1f W)   cuda=%.3e J (%.1f W)   sfu=%.3e J (%.1f W)"
                % (_pd(self.chip_tensor_energy_j), _pdw(self.chip_tensor_energy_j),
                   _pd(self.chip_cuda_energy_j), _pdw(self.chip_cuda_energy_j),
                   _pd(self.chip_sfu_energy_j), _pdw(self.chip_sfu_energy_j))
            )
        if self._entries and self.total_time_s > 0:
            T = self.total_time_s if self.total_time_s > 0 else 1.0
            lines.append("  [By-Op-Name Summary] (sorted by total time)")
            for g in self.breakdown_by_name():
                lines.append(
                    "    %-32s  n_total=%-5g  total=%.3e s  (%5.1f%%)  compute=%.3e s  comm=%.3e s  noc_energy=%.3e J"
                    % (g["op_name"], g["n_total"], g["time_total_s"], g["time_pct"], g["compute_s"], g["comm_s"], g["noc_energy_j"])
                )
                if g["chip_total_energy_j"] > 0:
                    def _gw(j: float) -> float: return j / T
                    lines.append(
                        "      chip: total=%.3e J (%.1f W)  mem=%.3e J [DRAM=%.3e L2=%.3e smem=%.3e reg=%.3e]  compute=%.3e J [tensor=%.3e cuda=%.3e sfu=%.3e]  static=%.3e J"
                        % (g["chip_total_energy_j"], _gw(g["chip_total_energy_j"]),
                           g["chip_memory_energy_j"], g["chip_dram_energy_j"], g["chip_l2_energy_j"], g["chip_smem_energy_j"], g["chip_reg_energy_j"],
                           g["chip_compute_energy_j"], g["chip_tensor_energy_j"], g["chip_cuda_energy_j"], g["chip_sfu_energy_j"], g["chip_static_energy_j"])
                    )
        if self._entries and self.total_time_s > 0:
            lines.append("  [Per-Entry Breakdown] (n x e2e_time, descending)")
            for n, s in self.breakdown():
                op_total = n * s.e2e_time_s
                pct = op_total / self.total_time_s * 100
                lines.append("    %-32s  n=%-4g  e2e=%.3e s  total=%.3e s  (%5.1f%%)" % (s.op_name, n, s.e2e_time_s, op_total, pct))
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

    def render_human(self, markdown: bool = False) -> str:
        """生成人类可读文本，信息对齐 dump_log（表格版）。"""
        T = self.total_time_s if self.total_time_s > 0 else 1.0
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
        blocks.append(f"## ModelPerfStats [{self.name}]" if markdown else f"=== ModelPerfStats [{self.name}] ===")

        blocks.append("[Total Timing]")
        blocks.append(self._render_table(
            ["metric", "value"],
            [
                ["total", f"{self.total_time_s:.3e} s"],
                ["compute", f"{self.compute_time_s:.3e} s ({compute_pct:.1f}%)"],
                ["comm", f"{self.comm_time_s:.3e} s ({comm_pct:.1f}%)"],
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

        by_name = self.breakdown_by_name()
        if by_name:
            blocks.append("[By-Op-Name Summary]")
            rows = []
            for g in by_name:
                rows.append([
                    g["op_name"],
                    f"{g['n_total']}",
                    f"{g['time_total_s']:.3e} s ({g['time_pct']:.1f}%)",
                    f"{g['compute_s']:.3e} s",
                    f"{g['comm_s']:.3e} s",
                    f"{g['noc_energy_j']:.3e} J",
                    f"{g['chip_total_energy_j']:.3e} J",
                ])
            blocks.append(self._render_table(
                ["op_name", "n_total", "total", "compute", "comm", "noc_energy", "chip_total"],
                rows,
                markdown,
            ))

        if self._entries and self.total_time_s > 0:
            blocks.append("[Per-Entry Breakdown]")
            rows = []
            for n, s in self.breakdown():
                op_total = n * s.e2e_time_s
                pct = op_total / self.total_time_s * 100
                rows.append([s.op_name, f"{n}", f"{s.e2e_time_s:.3e} s", f"{op_total:.3e} s ({pct:.1f}%)"])
            blocks.append(self._render_table(["op_name", "n", "e2e", "total"], rows, markdown))

        if self._entries:
            blocks.append("[Per-Op Detailed Dump]")
            ordered_entries = self.breakdown()
            for idx, (n, s) in enumerate(ordered_entries):
                if markdown:
                    blocks.append(f"### Entry {idx + 1}: `{s.op_name}` (n={n})")
                    blocks.append(s.render_human(markdown=True, title_level=4))
                else:
                    blocks.append(f"Entry {idx + 1}: {s.op_name} (n={n})")
                    blocks.append(s.render_human(markdown=False))

        return "\n\n".join(blocks)

    def dump_log(self) -> None:
        """用 log.info 分层打印聚合统计 + 各 op 贡献分解。"""
        for line in self._build_detail_lines():
            log.info("%s", line)

    def dump_json(self, path: str) -> None:
        """序列化写入 JSON 文件。"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("ModelPerfStats [%s] saved to %s", self.name, path)

    def dump_human(self, path: str, markdown: bool = False) -> None:
        """写入人类可读格式（txt/md）。"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render_human(markdown=markdown))
            f.write("\n")
        log.info("ModelPerfStats [%s] human-readable dump saved to %s", self.name, path)
