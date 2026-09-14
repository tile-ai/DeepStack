# traffic_matrix.py
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, Optional
from functools import lru_cache
import cProfile
import logging
import os
import json
log = logging.getLogger(__name__)

class TrafficMatrix:
    """Dense directed traffic matrix:
      - counts[s, d] stores the cumulative bytes/packets from s to d (uint64).
      - N is fixed at 256 (configurable).
    """
    __slots__ = ("N", "counts")

    def __init__(self, N: int = 256):
        self.N = N
        self.counts = np.zeros((N, N), dtype=np.uint64, order="C")

    def add(self, src: int, dst: int, value: int) -> None:
        """Single-entry update: add value to src->dst."""
        self.counts[src, dst] += np.uint64(value)

    def add_many(self, srcs, dsts, values) -> None:
        s = np.asarray(list(srcs), dtype=np.uint32)
        d = np.asarray(list(dsts), dtype=np.uint32)
        v = np.asarray(list(values), dtype=np.uint64)
        idx = s * np.uint32(self.N) + d
        flat = self.counts.ravel()
        np.add.at(flat, idx.astype(np.intp, copy=False), v)

    def out_bytes(self, node: int) -> int:
        """Sum outgoing traffic from node."""
        return int(self.counts[node, :].sum(dtype=np.uint64))

    def in_bytes(self, node: int) -> int:
        """Sum incoming traffic to node."""
        return int(self.counts[:, node].sum(dtype=np.uint64))

    def totals(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Return (network-wide total, per-node outgoing traffic array, per-node incoming traffic array)."""
        row_sum = self.counts.sum(axis=1, dtype=np.uint64)
        col_sum = self.counts.sum(axis=0, dtype=np.uint64)
        total = int(row_sum.sum(dtype=np.uint64))
        return total, row_sum.astype(np.uint64, copy=False), col_sum.astype(np.uint64, copy=False)

    def topk_pairs(self, k: int = 10) -> list[tuple[int, int, int]]:
        """Return the k largest traffic entries as (src, dst, value) tuples.
        Suitable for k << N^2.
        """
        flat = self.counts.ravel()
        if k >= flat.size:
            k = flat.size
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        N = self.N
        result = []
        for t in idx:
            val = int(flat[t])
            if val == 0:
                break
            s, d = divmod(int(t), N)
            result.append((s, d, val))
        return result

    def reset(self) -> None:
        """Zero the matrix."""
        self.counts.fill(0)

    def save(self, path: str) -> None:
        """Save as .npy (fastest and lossless)."""
        np.save(path, self.counts, allow_pickle=False)

    @classmethod
    def load(cls, path: str) -> "TrafficMatrix":
        arr = np.load(path, allow_pickle=False)
        tm = cls(N=arr.shape[0])
        if arr.shape != (tm.N, tm.N) or arr.dtype != np.uint64:
            raise ValueError("文件格式不匹配（需要 (N,N) 且 dtype=uint64）")
        tm.counts[:] = arr
        return tm

    def save_heatmap(
        self,
        path: str,
        cmap: str = "viridis",
        normalize: bool = False,
        log_scale: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: Tuple[float, float] = (6.0, 5.0),
        dpi: int = 300,
    ) -> None:
        """Export the current traffic matrix as a heatmap image."""

        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
        except Exception as e:
            raise ImportError(
                "需要安装 matplotlib 才能导出热力图：pip install matplotlib"
            ) from e

        data = self.counts.astype(np.float64, copy=True)

        if normalize and not log_scale:
            max_val = float(data.max())
            if max_val > 0.0:
                data /= max_val

        norm = None
        if log_scale:
            positive = data[data > 0.0]
            if positive.size > 0:
                auto_vmin = float(positive.min()) if vmin is None else float(vmin)
                auto_vmax = float(positive.max()) if vmax is None else float(vmax)
                if auto_vmax <= auto_vmin:
                    auto_vmax = auto_vmin * 10.0
                if auto_vmin <= 0.0:
                    auto_vmin = min(1.0, auto_vmax / 10.0)
                norm = LogNorm(vmin=auto_vmin, vmax=auto_vmax)
            else:
                norm = None

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        im = ax.imshow(
            data,
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            norm=norm,
            vmin=None if log_scale else vmin,
            vmax=None if log_scale else vmax,
        )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Traffic (bytes)", fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.get_offset_text().set_fontsize(16)

        ax.set_xlabel("Destination", fontsize=16)
        ax.set_ylabel("Source", fontsize=16)
        ax.set_title(f"Traffic Matrix Heatmap (N={self.N})", fontsize=16)

        ax.tick_params(axis="both", which="major", labelsize=16)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def export_json(self, path: str) -> None:
        """Append the current matrix as a JSON record to the file at `path`.

        Rules:
          - If the file does not exist, create it with the first record and time_stamp = 0.
          - Otherwise, read the last record's time_stamp and increment it by 1 for the new record.
          - Include all current (src, dst, data_traffic_bytes) entries, excluding
            diagonal entries (src == dst) and entries with data_traffic_bytes <= 0.
        """
        N = int(self.N)

        offdiag = ~np.eye(N, dtype=bool)
        positive = self.counts > 0
        mask = offdiag & positive
        src_idx, dst_idx = np.nonzero(mask)
        values = self.counts[src_idx, dst_idx].astype(np.uint64, copy=False)

        entries = [
            {
                "src": int(s),
                "dst": int(d),
                "data_traffic_bytes": int(v),
            }
            for s, d, v in zip(src_idx.tolist(), dst_idx.tolist(), values.tolist())
        ]

        data: list[dict] = []
        last_ts = -1
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                    if data:
                        last_ts = int(data[-1].get("time_stamp", -1))
            except Exception:
                data = []
                last_ts = -1

        new_ts = last_ts + 1
        record = {
            "time_stamp": new_ts,
            "entries": entries,
        }

        data.append(record)

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_parallelism_groups(
        N: int, tp: int, ep: int, sp: int, cp: int, dp: int, pp: int
    ) -> dict[str, list[list[int]]]:
        """Cached core computation for grouping rules.
        Dimensions run from innermost to outermost: TP -> EP -> SP -> CP -> DP -> PP.
        """
        dims = [tp, ep, sp, cp, dp, pp]
        names = ["tp", "ep", "sp", "cp", "dp", "pp"]

        total = 1
        for x in dims:
            total *= x
        if total != N:
            raise ValueError(f"并行维度乘积 {total} 不等于矩阵大小 {N}")

        prefix = [1] * (len(dims) + 1)   # prefix[i] = prod(dims[:i])
        for i in range(1, len(dims) + 1):
            prefix[i] = prefix[i - 1] * dims[i - 1]
        suffix = [1] * (len(dims) + 1)   # suffix[i] = prod(dims[i:])
        for i in range(len(dims) - 1, -1, -1):
            suffix[i] = suffix[i + 1] * dims[i]

        result: dict[str, list[list[int]]] = {}

        for level, (gsize, name) in enumerate(zip(dims, names)):
            if gsize <= 0:
                raise ValueError(f"{name} 必须为正整数")
            stride = prefix[level]
            outer_blocks = suffix[level + 1]
            if N % gsize != 0:
                raise ValueError(f"N 必须能被 {name} 整除（N={N}, {name}={gsize}）")

            groups: list[list[int]] = []
            for off in range(stride):                           # 0..stride-1
                for ob in range(outer_blocks):                 # 0..outer_blocks-1
                    start = ob * (gsize * stride) + off
                    group = [start + k * stride for k in range(gsize)]
                    groups.append(group)

            result[f"{name}_groups"] = groups

        return result

    def group_by_parallelism(
        self, tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2
    ) -> dict[str, list[list[int]]]:
        """Group by parallelism, with dimensions ordered from innermost to outermost:
        TP -> EP -> SP -> CP -> DP -> PP.

        - Group size = current dimension size (xp).
        - Number of groups = N / xp.
        - Stride = product of all preceding inner dimension sizes, excluding the current one.

        Example: N=256, (tp,ep,sp,cp,dp,pp)=(8,1,8,1,2,2)
        - TP: size 8, 32 groups, stride 1 -> contiguous blocks [0..7], [8..15], ...
        - SP: size 8, 32 groups, stride 8 -> [0,8,16,24,32,40,48,56], [1,9,17,...], ...
        - DP: size 2, 128 groups, stride 64 -> [0,64], [1,65], ...
        - PP: size 2, 128 groups, stride 128 -> [0,128], [1,129], ...
        EP and CP follow the same rules, generating singleton groups even when their size is 1.

        Uses an LRU cache to accelerate repeated calls.
        """
        return self._compute_parallelism_groups(self.N, tp, ep, sp, cp, dp, pp)

    def visualize_parallelism_groups(
        self,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
        save_path: str = "parallelism_groups.png"
    ) -> None:
        """Visualize grouping across all six dimensions, with group size, group count, and stride in the titles."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            raise ImportError("需要安装 matplotlib：pip install matplotlib") from e

        groups = self.group_by_parallelism(tp, ep, sp, cp, dp, pp)
        dims = [("tp", tp), ("ep", ep), ("sp", sp), ("cp", cp), ("dp", dp), ("pp", pp)]
        dim_sizes = [tp, ep, sp, cp, dp, pp]
        stride = [1]
        for x in dim_sizes[:-1]:
            stride.append(stride[-1] * x)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()

        for i, ((name, size), ax) in enumerate(zip(dims, axes)):
            g = groups[f"{name}_groups"]
            node_colors = np.zeros(self.N, dtype=int)
            for gid, arr in enumerate(g):
                for n in arr:
                    node_colors[n] = gid % 20
            ax.bar(range(self.N), [1] * self.N, color=plt.cm.tab20(node_colors))
            ax.set_title(f"{name.upper()}  size={size}, groups={len(g)}, stride={stride[i]}")
            ax.set_xlabel("Node ID")
            ax.set_yticks([])
            ax.set_xlim(-0.5, self.N - 0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def add_intra_group_traffic(
        self,
        which: str,
        bytes_per_pair: int,
        *,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
        include_self: bool = False,
    ) -> None:
        """Add bytes_per_pair bytes of directed traffic i->j to every pair (i, j)
        in every group of parallel dimension 'which' ('tp'|'ep'|'sp'|'cp'|'dp'|'pp').
        Self-pairs are excluded by default (i != j).

        This is equivalent to each node reading bytes_per_pair bytes from every other
        node in its group, so both i->j and j->i receive the same increment
        (symmetric off-diagonal increments).

        Args:
            which: Dimension name, such as "sp".
            bytes_per_pair: Increment for each pair (i, j), in bytes.
            include_self: If True, also increment diagonal entries (i == j).
            Other parameters: Dimension sizes passed to group_by_parallelism.
        """
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        key = f"{which}_groups"
        if key not in groups:
            raise ValueError(f"未找到分组 {key}")

        inc = np.uint64(bytes_per_pair)

        for g in groups[key]:
            m = len(g)
            if m == 0:
                continue
            idx = np.ix_(g, g)
            self.counts[idx] += inc
            if not include_self:
                self.counts[g, g] -= inc

    def add_intra_group_traffic_bulk(
        self,
        plan: dict[str, int],
        *,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
        include_self: bool = False,
    ) -> None:
        """Add fully connected intragroup traffic for multiple dimensions in one batch.
        Example: plan={'sp': 1024, 'dp': 2048}

        Other parameters are the same as for add_intra_group_traffic.
        """
        for which, bytes_per_pair in plan.items():
            self.add_intra_group_traffic(
                which,
                bytes_per_pair,
                tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp,
                include_self=include_self,
            )

    def add_intra_group_traffic_pair(
        self,
        which: str,
        bytes_per_pair: int,
        *,
        src: int,
        dst: int,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
    ) -> None:
        """Add a unicast transfer of bytes_per_pair bytes from group position src to
        group position dst in every group of parallel dimension 'which'
        ('tp'|'ep'|'sp'|'cp'|'dp'|'pp').

        For example, which="sp", sp=8, src=0, dst=1 adds directed traffic
        0->8, 1->9, ... across SP groups such as [0,8,16,24,32,40,48,56].

        Asserts 0 <= src < gsize and 0 <= dst < gsize, where gsize is the
        size of the selected dimension.
        """
        # tm.add_intra_group_traffic_pair("dp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        dim_size_map = {"tp": tp, "ep": ep, "sp": sp, "cp": cp, "dp": dp, "pp": pp}
        gsize = int(dim_size_map[which])
        assert 0 <= int(src) < gsize, f"src 索引越界：0 <= src < {gsize} (src={src})"
        assert 0 <= int(dst) < gsize, f"dst 索引越界：0 <= dst < {gsize} (dst={dst})"

        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        key = f"{which}_groups"
        if key not in groups:
            raise ValueError(f"未找到分组 {key}")

        s_list: list[int] = []
        d_list: list[int] = []
        for g in groups[key]:
            s_list.append(g[src])
            d_list.append(g[dst])

        if not s_list:
            return
        v_list = [int(bytes_per_pair)] * len(s_list)
        self.add_many(s_list, d_list, v_list)

    def add_intra_group_traffic_pair_bulk(
        self,
        which: str,
        triples: Iterable[Iterable[int]],
        *,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
    ) -> None:
        """Batch-add unicast transfers specified by [bytes, src, dst] triples
        within every group of parallel dimension which.

        Example:
            add_intra_group_traffic_pair_bulk("dp", [[2048,0,1],[2048,1,0]], tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
        """
        # tm.add_intra_group_traffic_pair_bulk("dp",
        # [[1024, 0, 1], [1024, 1, 0]],
        # tp=8, ep=1, sp=8, cp=1, dp=2, pp=2,
        # )

        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        dim_size_map = {"tp": tp, "ep": ep, "sp": sp, "cp": cp, "dp": dp, "pp": pp}
        gsize = int(dim_size_map[which])

        norm: list[tuple[int, int, int]] = []
        for t in triples:
            if not (isinstance(t, (list, tuple)) and len(t) == 3):
                raise ValueError("每个元素需为 [bytes, src, dst]")
            b, s, d = int(t[0]), int(t[1]), int(t[2])
            assert 0 <= s < gsize and 0 <= d < gsize, f"src/dst 越界（组大小={gsize}）：{t}"
            norm.append((b, s, d))

        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        key = f"{which}_groups"
        if key not in groups:
            raise ValueError(f"未找到分组 {key}")

        srcs: list[int] = []
        dsts: list[int] = []
        vals: list[int] = []
        for g in groups[key]:
            for (b, s, d) in norm:
                srcs.append(g[s])
                dsts.append(g[d])
                vals.append(b)

        if srcs:
            self.add_many(srcs, dsts, vals)

    def add_intra_group_traffic_pair_at_group(
        self,
        which: str,
        bytes_per_pair: int,
        *,
        group_index: int,
        src: int,
        dst: int,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
    ) -> None:
        """Add a unicast transfer of bytes_per_pair bytes from group position src
        to group position dst only in group group_index of dimension which.

        Example: N=256, (tp,ep,sp,cp,dp,pp)=(2,8,4,1,2,2), for dimension EP:
          - Group 0 = [0, 2, 4, 6, 8, 10, 12, 14]
          - Group 1 = [16, 18, 20, 22, 24, 26, 28, 30]
        With group_index=1, src=0, dst=2, add bytes_per_pair to global edge 16 -> 20.

        Asserts 0 <= src,dst < gsize and 0 <= group_index < num_groups.
        """
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        dim_size_map = {"tp": tp, "ep": ep, "sp": sp, "cp": cp, "dp": dp, "pp": pp}
        gsize = int(dim_size_map[which])
        assert 0 <= int(src) < gsize, f"src 索引越界：0 <= src < {gsize} (src={src})"
        assert 0 <= int(dst) < gsize, f"dst 索引越界：0 <= dst < {gsize} (dst={dst})"

        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        key = f"{which}_groups"
        if key not in groups:
            raise ValueError(f"未找到分组 {key}")
        glist = groups[key]

        if not (0 <= int(group_index) < len(glist)):
            raise IndexError(
                f"group_index 越界：0 <= group_index < {len(glist)} (group_index={group_index})"
            )

        g = glist[int(group_index)]
        s = g[int(src)]
        d = g[int(dst)]
        self.add(int(s), int(d), int(bytes_per_pair))

    def add_intra_group_traffic_pair_at_group_bulk(
        self,
        which: str,
        triples: Iterable[Iterable[int]],
        *,
        group_index: int,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
    ) -> None:
        """Batch-add unicast transfers specified by [bytes, src, dst] triples only
        in group group_index of dimension which. Unlike
        add_intra_group_traffic_pair_bulk, this affects a single group.

        Asserts 0 <= src,dst < gsize and 0 <= group_index < num_groups.
        """
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        dim_size_map = {"tp": tp, "ep": ep, "sp": sp, "cp": cp, "dp": dp, "pp": pp}
        gsize = int(dim_size_map[which])

        norm: list[tuple[int, int, int]] = []
        for t in triples:
            if not (isinstance(t, (list, tuple)) and len(t) == 3):
                raise ValueError("每个元素需为 [bytes, src, dst]")
            b, s, d = int(t[0]), int(t[1]), int(t[2])
            assert 0 <= s < gsize and 0 <= d < gsize, f"src/dst 越界（组大小={gsize}）：{t}"
            norm.append((b, s, d))

        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        key = f"{which}_groups"
        if key not in groups:
            raise ValueError(f"未找到分组 {key}")
        glist = groups[key]

        if not (0 <= int(group_index) < len(glist)):
            raise IndexError(
                f"group_index 越界：0 <= group_index < {len(glist)} (group_index={group_index})"
            )

        g = glist[int(group_index)]
        srcs: list[int] = []
        dsts: list[int] = []
        vals: list[int] = []
        for (b, s, d) in norm:
            srcs.append(g[s])
            dsts.append(g[d])
            vals.append(b)

        if srcs:
            self.add_many(srcs, dsts, vals)

    def intragroup_traffic_stats(
        self,
        *,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
        include_self: bool = False,
    ) -> dict[str, dict[str, int | float]]:
        """Compute intragroup traffic totals and averages for each parallel dimension
        (tp/ep/sp/cp/dp/pp).

        For each group G of a dimension, sum the counts[G, G] submatrix, excluding
        the diagonal (i != j) by default. The dimension total is the sum over all groups.

        Returns the following for each dimension:
            group_size, num_groups, stride,
            total_intragroup, avg_per_group, avg_per_node_out

        Definitions:
            total_intragroup: Sum of all intragroup i->j traffic (i != j) for the dimension.
            avg_per_group: total / num_groups.
            avg_per_node_out: total / N (mean intragroup outgoing traffic per node).

        If include_self=True, diagonal counts are included.
        """
        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        dims = [("tp", tp), ("ep", ep), ("sp", sp), ("cp", cp), ("dp", dp), ("pp", pp)]

        dim_sizes = [tp, ep, sp, cp, dp, pp]
        strides = []
        s = 1
        for i in range(len(dim_sizes)):
            strides.append(s)
            s *= dim_sizes[i]

        out: dict[str, dict[str, int | float]] = {}
        N = self.N
        for i, (name, size) in enumerate(dims):
            key = f"{name}_groups"
            glist = groups[key]
            if not glist:
                out[name] = {
                    "group_size": size, "num_groups": 0, "stride": strides[i],
                    "total_intragroup": 0, "avg_per_group": 0.0, "avg_per_node_out": 0.0,
                }
                continue

            total = np.uint64(0)
            for g in glist:
                sub = self.counts[np.ix_(g, g)]
                subtotal = sub.sum(dtype=np.uint64)
                if not include_self:
                    subtotal -= np.diag(sub).sum(dtype=np.uint64)
                total += np.uint64(subtotal)

            num_groups = len(glist)
            avg_per_group = float(total) / float(num_groups) if num_groups > 0 else 0.0
            avg_per_node = float(total) / float(N) if N > 0 else 0.0

            out[name] = {
                "group_size": size,
                "num_groups": num_groups,
                "stride": strides[i],
                "total_intragroup": int(total),
                "avg_per_group": avg_per_group,
                "avg_per_node_out": avg_per_node,
            }
        return out

    def _saturating_add_arrays(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add two uint64 arrays with saturation, clamping entries above 2^64-1 to that maximum."""
        assert a.dtype == np.uint64 and b.dtype == np.uint64 and a.shape == b.shape
        res = a + b
        overflow = res < a
        if np.any(overflow):
            res = res.copy()
            res[overflow] = np.uint64(0xFFFFFFFFFFFFFFFF)
        return res

    def __add__(self, other: "TrafficMatrix") -> "TrafficMatrix":
        """Return a new TrafficMatrix containing the elementwise saturating sum."""
        if not isinstance(other, TrafficMatrix) or other.N != self.N:
            raise ValueError("只能与同尺寸的 TrafficMatrix 相加")
        out = TrafficMatrix(self.N)
        out.counts = self._saturating_add_arrays(self.counts, other.counts)
        return out

    def __iadd__(self, other: "TrafficMatrix"):
        """Add in place with saturation and return self."""
        if not isinstance(other, TrafficMatrix) or other.N != self.N:
            raise ValueError("只能与同尺寸的 TrafficMatrix 相加")
        self.counts[:] = self._saturating_add_arrays(self.counts, other.counts)
        return self

    def add_matrix(self, other: "TrafficMatrix", *, saturating: bool = True, in_place: bool = True) -> "TrafficMatrix":
        """Explicit addition interface:
          - saturating=True uses saturating addition; False uses ordinary uint64 wraparound.
          - in_place=True adds in place; False returns a new object.
        """
        if not isinstance(other, TrafficMatrix) or other.N != self.N:
            raise ValueError("只能与同尺寸的 TrafficMatrix 相加")
        if saturating:
            if in_place:
                self.counts[:] = self._saturating_add_arrays(self.counts, other.counts)
                return self
            else:
                out = TrafficMatrix(self.N)
                out.counts = self._saturating_add_arrays(self.counts, other.counts)
                return out
        else:
            if in_place:
                self.counts += other.counts
                return self
            else:
                out = TrafficMatrix(self.N)
                out.counts = (self.counts + other.counts).astype(np.uint64, copy=True)
                return out




def test_TrafficMatrix_only():
    tm = TrafficMatrix(256)

    tm.add_intra_group_traffic("dp", 1024, tp=8, ep=1, sp=4, cp=1, dp=4, pp=2)

    # tm.add_intra_group_traffic_bulk(
    #     tp=8, ep=1, sp=8, cp=1, dp=2, pp=2
    # )

    
    total, out_arr, in_arr = tm.totals()
    log.info("total: %s", total)
    log.info("out(1): %s in(2): %s", tm.out_bytes(1), tm.in_bytes(2))
    log.info("top5: %s", tm.topk_pairs(5))
    tm.save_heatmap("traffic_matrix.png")
    


    groups = tm.group_by_parallelism(tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    log.info("并行分组示例 (TP=8, EP=1, SP=8, CP=1, DP=2, PP=2):")
    log.info("TP groups (前3组): %s", groups['tp_groups'][:3])
    log.info("EP groups (前3组): %s", groups['ep_groups'][:3])
    log.info("SP groups (前3组): %s", groups['sp_groups'][:3])
    log.info("CP groups (前3组): %s", groups['cp_groups'][:3])
    log.info("DP groups (前3组): %s", groups['dp_groups'][:3])
    log.info("PP groups (前3组): %s", groups['pp_groups'][:3])
    
    # tm.visualize_parallelism_groups(tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)

    tm2 = TrafficMatrix(256)
    # tm2.add_intra_group_traffic("dp", 16*1024, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    tm2.add_intra_group_traffic("dp", 2**35, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # tm2.save_heatmap("tm2_traffic_matrix_dp.png")

    stats = tm.intragroup_traffic_stats(tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    log.info("SP 组统计： %s", stats["sp"])

    tm_sum = tm + tm2

    # tm_sum.save_heatmap("tm_sum_traffic_matrix_sum.png")
    log.info("top5: %s", tm_sum.topk_pairs(5))

def test_TrafficMatrix_ep_all_to_all():
    tm = TrafficMatrix(256)

    # tm.add_intra_group_traffic("dp", 1024, tp=2, ep=8, sp=4, cp=1, dp=2, pp=2)
    tm.add_intra_group_traffic_pair_at_group_bulk(
    "ep",
    triples=[
        [2048, 0, 2],
        [1024, 1, 3],
    ],
    group_index=1,
    tp=2, ep=8, sp=4, cp=1, dp=2, pp=2,
    )

    # tm.add_intra_group_traffic_bulk(
    #     tp=8, ep=1, sp=8, cp=1, dp=2, pp=2
    # )

    total, out_arr, in_arr = tm.totals()
    log.info("total: %s", total)
    log.info("out(1): %s in(2): %s", tm.out_bytes(1), tm.in_bytes(2))
    log.info("top5: %s", tm.topk_pairs(5))
    tm.save_heatmap("traffic_matrix.png")
    


    groups = tm.group_by_parallelism(tp=2, ep=8, sp=4, cp=1, dp=2, pp=2)
    log.info("并行分组示例 (TP=2, EP=8, SP=4, CP=1, DP=2, PP=2):")
    log.info("TP groups (前3组): %s", groups['tp_groups'][:3])
    log.info("EP groups (前3组): %s", groups['ep_groups'][:3])
    # log.info("EP groups: %s", groups['ep_groups'])
    log.info("SP groups (前3组): %s", groups['sp_groups'][:3])
    log.info("CP groups (前3组): %s", groups['cp_groups'][:3])
    log.info("DP groups (前3组): %s", groups['dp_groups'][:3])
    log.info("PP groups (前3组): %s", groups['pp_groups'][:3])


    log.info("top5: %s", tm.topk_pairs(5))

def test_traffic_matrix_intra_group_pair_unit():
    """Verify add_intra_group_traffic_pair for N=16, tp=4, sp=4:
      - which="sp", bytes=100, src=0, dst=1
      - Expect four edges: 0->4, 1->5, 2->6, 3->7, each with value 100.
      - All other entries must be zero.
    """
    tm = TrafficMatrix(256)
    # tm.add_intra_group_traffic_pair("sp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # tm.add_intra_group_traffic_pair("tp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # tm.add_intra_group_traffic_pair("dp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    tm.add_intra_group_traffic_pair("dp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    tm.add_intra_group_traffic_pair("dp", 100, src=1, dst=0, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # expected = {(0, 4): 100, (1, 5): 100, (2, 6): 100, (3, 7): 100}
    # for (s, d), v in expected.items():
    # total, _, _ = tm.totals()
    # log.info("test_intra_group_pair_unit passed: total=%s", total)

    # tm.save_heatmap("tm_traffic_matrix_intra_group_pair_unit.png")


def test_traffic_matrix_intra_group_pair_bulk_unit():
    """Verify add_intra_group_traffic_pair_bulk for N=16, tp=4, sp=4:
      - sp: bytes=200, src=2, dst=3 -> 8->12, 9->13, 10->14, 11->15
      - tp: bytes=50, src=1, dst=0 -> 1->0, 5->4, 9->8, 13->12
      - The total must be 4*200 + 4*50 = 1000.
    """
    tm = TrafficMatrix(256)
    tm.add_intra_group_traffic_pair_bulk(
        "dp",
        [[1024, 0, 1], [1024, 1, 0]],
        tp=8, ep=1, sp=8, cp=1, dp=2, pp=2,
    )
    # expected = {
    #     (8, 12): 200, (9, 13): 200, (10, 14): 200, (11, 15): 200,
    #     (1, 0): 50, (5, 4): 50, (9, 8): 50, (13, 12): 50,
    # }
    # for (s, d), v in expected.items():
    # total, _, _ = tm.totals()
    # log.info("test_intra_group_pair_bulk_unit passed: total=%s", total)
    # tm.save_heatmap("tm_traffic_matrix_intra_group_pair_bulk_unit.png")



if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    )
    # cProfile.run("test_example_and_print()")
    # test_TrafficMatrix_only()
    test_TrafficMatrix_ep_all_to_all()
    # test_traffic_matrix_intra_group_pair_unit()
    # test_traffic_matrix_intra_group_pair_bulk_unit()
    # test_example_and_print()


    # py-spy record -o profile.svg -- python your_script.py
    # pyinstrument -r html -o pyinstrument.html traffic_matrix.py > test_pyinstrument.log
