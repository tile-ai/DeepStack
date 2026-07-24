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
    """
    稠密有向流量矩阵：
      - counts[s, d] 存 s->d 的累计字节/包数（uint64）
      - N 固定为 256（可改）
    """
    __slots__ = ("N", "counts")

    def __init__(self, N: int = 256):
        self.N = N
        self.counts = np.zeros((N, N), dtype=np.uint64, order="C")  # row-major，cache 友好

    # ========= 写入 =========
    def add(self, src: int, dst: int, value: int) -> None:
        """单条更新：src->dst 增加 value。"""
        self.counts[src, dst] += np.uint64(value)

    def add_many(self, srcs, dsts, values) -> None:
        s = np.asarray(list(srcs), dtype=np.uint32)
        d = np.asarray(list(dsts), dtype=np.uint32)
        v = np.asarray(list(values), dtype=np.uint64)
        idx = s * np.uint32(self.N) + d
        flat = self.counts.ravel()
        np.add.at(flat, idx.astype(np.intp, copy=False), v)  # 整型路径，无精度损失

    # ========= 查询/统计 =========
    def out_bytes(self, node: int) -> int:
        """节点 node 的出流量和。"""
        return int(self.counts[node, :].sum(dtype=np.uint64))

    def in_bytes(self, node: int) -> int:
        """节点 node 的入流量和。"""
        return int(self.counts[:, node].sum(dtype=np.uint64))

    def totals(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        返回 (全网总和, 每节点出流数组, 每节点入流数组)
        """
        row_sum = self.counts.sum(axis=1, dtype=np.uint64)
        col_sum = self.counts.sum(axis=0, dtype=np.uint64)
        total = int(row_sum.sum(dtype=np.uint64))
        return total, row_sum.astype(np.uint64, copy=False), col_sum.astype(np.uint64, copy=False)

    def topk_pairs(self, k: int = 10) -> list[tuple[int, int, int]]:
        """
        返回流量最大的前 k 个 (src, dst, value)。
        适合 k << N^2 的场景。
        """
        flat = self.counts.ravel()
        if k >= flat.size:
            k = flat.size
        # argpartition 找到前 k 大的下标，再按值排序
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
        """清零矩阵。"""
        self.counts.fill(0)

    # ========= 持久化 =========
    def save(self, path: str) -> None:
        """保存为 .npy（最快/无损）"""
        np.save(path, self.counts, allow_pickle=False)

    @classmethod
    def load(cls, path: str) -> "TrafficMatrix":
        arr = np.load(path, allow_pickle=False)
        tm = cls(N=arr.shape[0])
        if arr.shape != (tm.N, tm.N) or arr.dtype != np.uint64:
            raise ValueError("文件格式不匹配（需要 (N,N) 且 dtype=uint64）")
        tm.counts[:] = arr
        return tm

    # ========= 可视化 =========
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
        """
        将当前流量矩阵导出为热力图图片。

        参数：
          - path: 输出文件路径（如 .png, .pdf）
          - cmap: 颜色映射（如 "viridis", "magma", "inferno" 等）
          - normalize: 是否按最大值归一化到 [0,1]（与 log_scale 互不依赖）
          - log_scale: 是否使用对数色标（适合长尾分布；0 值会显示为背景色）
          - vmin/vmax: 手动指定颜色范围（float）。未指定时将自动推断。
          - figsize: 图像尺寸（英寸）
          - dpi: 输出分辨率
        """

        try:
            import matplotlib.pyplot as plt  # 惰性导入，避免硬依赖
            from matplotlib.colors import LogNorm
        except Exception as e:  # noqa: BLE001 - 明确提示依赖
            raise ImportError(
                "需要安装 matplotlib 才能导出热力图：pip install matplotlib"
            ) from e

        data = self.counts.astype(np.float64, copy=True)

        # 归一化（不影响 log_scale 的 vmin/vmax 推断）
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
                # 避免 vmin==vmax 或 vmin<=0 导致异常
                if auto_vmax <= auto_vmin:
                    auto_vmax = auto_vmin * 10.0
                if auto_vmin <= 0.0:
                    auto_vmin = min(1.0, auto_vmax / 10.0)
                norm = LogNorm(vmin=auto_vmin, vmax=auto_vmax)
            else:
                norm = None  # 全 0，使用线性色标即可

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
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("bytes")
        ax.set_xlabel("dst")
        ax.set_ylabel("src")
        ax.set_title(f"Traffic Matrix Heatmap (N={self.N})")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def export_json(self, path: str) -> None:
        """
        将当前矩阵以追加式 JSON 导出到文件 `path`。

        规则：
          - 若文件不存在：创建并写入首条记录，time_stamp = 0；
          - 若文件已存在：读取最后一条记录的 time_stamp，新的记录取其 +1；
          - 记录内容包含本次所有 (src, dst, data_traffic_bytes) 条目；
            排除对角线 (src == dst)，且仅写入 data_traffic_bytes > 0 的条目。
        """
        N = int(self.N)

        # 选取需要导出的索引对：非对角线且流量 > 0
        offdiag = ~np.eye(N, dtype=bool)
        positive = self.counts > 0
        mask = offdiag & positive
        src_idx, dst_idx = np.nonzero(mask)
        values = self.counts[src_idx, dst_idx].astype(np.uint64, copy=False)

        # 构造条目列表（转为 Python int 以便 JSON 序列化）
        entries = [
            {
                "src": int(s),
                "dst": int(d),
                "data_traffic_bytes": int(v),
            }
            for s, d, v in zip(src_idx.tolist(), dst_idx.tolist(), values.tolist())
        ]

        # 读取已有 JSON，决定新的 time_stamp
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
                # 若不是列表或为空则按新文件处理
            except Exception:
                # 解析失败则重置为新文件写入
                data = []
                last_ts = -1

        new_ts = last_ts + 1
        record = {
            "time_stamp": new_ts,
            "entries": entries,
        }

        data.append(record)

        # 确保目录存在
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 写回（整体重写，简单可靠）
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_parallelism_groups(
        N: int, tp: int, ep: int, sp: int, cp: int, dp: int, pp: int
    ) -> dict[str, list[list[int]]]:
        """
        分组规则的核心计算逻辑（带缓存）。
        按维度从内到外：TP -> EP -> SP -> CP -> DP -> PP
        """
        dims = [tp, ep, sp, cp, dp, pp]
        names = ["tp", "ep", "sp", "cp", "dp", "pp"]

        total = 1
        for x in dims:
            total *= x
        if total != N:
            raise ValueError(f"并行维度乘积 {total} 不等于矩阵大小 {N}")

        # 预计算前缀乘积（用于步长）与后缀乘积（用于外层块数）
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
            stride = prefix[level]              # 步长 = 之前所有维度累乘
            outer_blocks = suffix[level + 1]    # 外层块数 = 之后所有维度累乘
            if N % gsize != 0:
                raise ValueError(f"N 必须能被 {name} 整除（N={N}, {name}={gsize}）")

            groups: list[list[int]] = []
            # 总组数应为 N / gsize = stride * outer_blocks
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
        """
        分组规则（按维度从内到外：TP -> EP -> SP -> CP -> DP -> PP）：
        - 组大小 = 当前维度大小（xp）
        - 组数量 = N / xp
        - 步长 = 之前所有内层维度大小的累乘（不含当前维度）
        例：N=256, (tp,ep,sp,cp,dp,pp)=(8,1,8,1,2,2)
        - TP: 大小8, 组数32, 步长1  -> 连续块 [0..7], [8..15], ...
        - SP: 大小8, 组数32, 步长8  -> [0,8,16,24,32,40,48,56], [1,9,17,...], ...
        - DP: 大小2, 组数128,步长64 -> [0,64], [1,65], ...
        - PP: 大小2, 组数128,步长128-> [0,128], [1,129], ...
        EP/CP 同理（即使=1也按规则生成单元素组）。
        
        注：此函数使用了 LRU 缓存来提高重复调用的性能。
        """
        return self._compute_parallelism_groups(self.N, tp, ep, sp, cp, dp, pp)

    def visualize_parallelism_groups(
        self,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
        save_path: str = "parallelism_groups.png"
    ) -> None:
        """
        可视化 6 个维度的分组；标题展示：组大小/组数/步长。
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            raise ImportError("需要安装 matplotlib：pip install matplotlib") from e

        groups = self.group_by_parallelism(tp, ep, sp, cp, dp, pp)
        dims = [("tp", tp), ("ep", ep), ("sp", sp), ("cp", cp), ("dp", dp), ("pp", pp)]
        # 计算各维度步长（前缀乘积）
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
                    node_colors[n] = gid % 20  # 20色循环
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
        """
        在指定并行维度 'which'（'tp'|'ep'|'sp'|'cp'|'dp'|'pp'）的每个分组内，
        为所有 (i, j), i!=j（默认）一次性增加 bytes_per_pair 字节的有向流量 i->j。
        这等价于“组内每个节点各自从其余节点读取 bytes_per_pair 字节”，
        因而最终 i->j 与 j->i 都会各增加同样的数值（对称的非对角增量）。

        参数：
          - which: 维度名，例如 "sp"
          - bytes_per_pair: 每对 (i, j) 的增量（字节）
          - include_self: 若为 True，则连对角 (i==j) 也会增加
          - 其余参数：传给 group_by_parallelism 的各维度大小
        """
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        # 取各维度分组
        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        key = f"{which}_groups"
        if key not in groups:
            raise ValueError(f"未找到分组 {key}")

        inc = np.uint64(bytes_per_pair)

        # 对每个组：一次性把子矩阵的所有元素加 inc，然后（默认）把对角减回 inc
        for g in groups[key]:
            m = len(g)
            if m == 0:
                continue
            # 子矩阵切片（行列同为该组）
            idx = np.ix_(g, g)
            self.counts[idx] += inc  # 组内所有元素 +inc
            if not include_self:
                # 把对角线减回 inc（避免 uint64 下先减导致下溢，必须先加后减）
                self.counts[g, g] -= inc

    def add_intra_group_traffic_bulk(
        self,
        plan: dict[str, int],
        *,
        tp: int = 8, ep: int = 1, sp: int = 8, cp: int = 1, dp: int = 2, pp: int = 2,
        include_self: bool = False,
    ) -> None:
        """
        一次对多个维度批量增加组内全互连流量。
        例如：plan={'sp': 1024, 'dp': 2048}

        其余参数同 add_intra_group_traffic。
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
        """
        在指定并行维度 'which'（'tp'|'ep'|'sp'|'cp'|'dp'|'pp'）的每个分组内，
        仅让组内位置 src 的节点向组内位置 dst 的节点发送 bytes_per_pair 字节（单播）。

        例如：which="sp", sp=8, src=0, dst=1，则对每个 SP 组（如 [0,8,16,24,32,40,48,56]）
        增加 0->8、1->9、... 的有向流量。

        断言：0 <= src < gsize 且 0 <= dst < gsize，其中 gsize 为对应维度大小。
        """
        # tm.add_intra_group_traffic_pair("dp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        # 对应维度大小
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
            # 每个组的第 src 向第 dst 发送
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
        """
        在指定并行维度 which 的每个分组内，按多组三元组 [bytes, src, dst] 批量添加单播。
        示例：add_intra_group_traffic_pair_bulk("dp", [[2048,0,1],[2048,1,0]], tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
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

        # 归一化与校验 triples
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
        """
        仅对指定维度 which 的“第 group_index 个组”，让组内位置 src 的节点
        向组内位置 dst 的节点发送 bytes_per_pair 字节（单播）。

        例：N=256, (tp,ep,sp,cp,dp,pp)=(2,8,4,1,2,2)，在 EP 维度：
          - 第 0 组 = [0, 2, 4, 6, 8, 10, 12, 14]
          - 第 1 组 = [16, 18, 20, 22, 24, 26, 28, 30]
        若选择 group_index=1, src=0, dst=2，则会在全局节点 16 -> 20 增加 bytes_per_pair。

        断言：0 <= src,dst < gsize 且 0 <= group_index < num_groups。
        """
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        # 对应维度大小
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
        """
        仅对指定维度 which 的“第 group_index 个组”，按多组三元组 [bytes, src, dst]
        批量添加单播。与 add_intra_group_traffic_pair_bulk 的区别是仅作用于单个组。

        断言：0 <= src,dst < gsize 且 0 <= group_index < num_groups。
        """
        which = which.lower()
        if which not in {"tp", "ep", "sp", "cp", "dp", "pp"}:
            raise ValueError("which 必须是 'tp'|'ep'|'sp'|'cp'|'dp'|'pp' 之一")

        dim_size_map = {"tp": tp, "ep": ep, "sp": sp, "cp": cp, "dp": dp, "pp": pp}
        gsize = int(dim_size_map[which])

        # 归一化与校验 triples
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
        """
        统计每个并行维度（tp/ep/sp/cp/dp/pp）的“组内流量”总和与均值。
        - 组内流量定义：对该维度的每个分组 G，统计子矩阵 counts[G, G]，
          默认排除对角（i != j）。维度的 total = 各组子矩阵之和。
        - 返回：对每个维度给出
            group_size, num_groups, stride,
            total_intragroup, avg_per_group, avg_per_node_out
          说明：
            * total_intragroup: 该维度所有组内 i->j(i!=j) 的和
            * avg_per_group: total / num_groups
            * avg_per_node_out: total / N   （平均到每个节点的组内“出流量”）
          若 include_self=True，则包含对角计数。
        """
        # 拿到分组
        groups = self.group_by_parallelism(tp=tp, ep=ep, sp=sp, cp=cp, dp=dp, pp=pp)
        dims = [("tp", tp), ("ep", ep), ("sp", sp), ("cp", cp), ("dp", dp), ("pp", pp)]

        # 计算各维度的 stride（内到外的前缀乘积，不含自身）
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
            # 平均每组总量
            avg_per_group = float(total) / float(num_groups) if num_groups > 0 else 0.0
            # 平均每个节点（出向）= total / N
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

    # ======= 矩阵相加（支持饱和加法，避免 uint64 溢出） =======
    def _saturating_add_arrays(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        对两个 uint64 数组做饱和相加：超过 2^64-1 的位置钳制到最大值。
        """
        assert a.dtype == np.uint64 and b.dtype == np.uint64 and a.shape == b.shape
        res = a + b
        overflow = res < a  # uint64 下，溢出会回绕为更小的数
        if np.any(overflow):
            res = res.copy()
            res[overflow] = np.uint64(0xFFFFFFFFFFFFFFFF)
        return res

    def __add__(self, other: "TrafficMatrix") -> "TrafficMatrix":
        """
        返回一个新的 TrafficMatrix，等于元素级相加（饱和相加）。
        """
        if not isinstance(other, TrafficMatrix) or other.N != self.N:
            raise ValueError("只能与同尺寸的 TrafficMatrix 相加")
        out = TrafficMatrix(self.N)
        out.counts = self._saturating_add_arrays(self.counts, other.counts)
        return out

    def __iadd__(self, other: "TrafficMatrix"):
        """
        原地相加（饱和相加），返回 self。
        """
        if not isinstance(other, TrafficMatrix) or other.N != self.N:
            raise ValueError("只能与同尺寸的 TrafficMatrix 相加")
        self.counts[:] = self._saturating_add_arrays(self.counts, other.counts)
        return self

    def add_matrix(self, other: "TrafficMatrix", *, saturating: bool = True, in_place: bool = False) -> "TrafficMatrix":
        """
        显式的相加接口：
          - saturating=True：饱和相加；False：普通 uint64 回绕相加
          - in_place=True：原地加；False：返回新对象
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
                self.counts += other.counts  # 可能回绕
                return self
            else:
                out = TrafficMatrix(self.N)
                out.counts = (self.counts + other.counts).astype(np.uint64, copy=True)
                return out




# # ======== 简单用例 ========
def test_TrafficMatrix_only():
    tm = TrafficMatrix(256)
    # tm.add(1, 2, 1500)             # 1->2 加 1500
    # tm.add(1, 2, 500)              # 1->2 再加 500
    # tm.add_many([0, 5, 5], [3, 7, 7], [100, 200, 300])  # 批量

    # 在 SP 维度上：每个 SP 组（大小 8，总 32 组）内两两添加 4KB
    tm.add_intra_group_traffic("dp", 1024, tp=8, ep=1, sp=4, cp=1, dp=4, pp=2)

    # 也可以一次对多个维度加不同量，比如：
    # tm.add_intra_group_traffic_bulk(
    #     {"tp": 8 * 1024, "sp": 4 * 1024, "dp": 2 * 1024, "pp": 1 * 1024},  # SP 组内每对 +4KB，DP 组内每对 +2KB，PP 组内每对 +2KB
    #     tp=8, ep=1, sp=8, cp=1, dp=2, pp=2
    # )

    
    total, out_arr, in_arr = tm.totals()
    log.info("total: %s", total)
    log.info("out(1): %s in(2): %s", tm.out_bytes(1), tm.in_bytes(2))
    log.info("top5: %s", tm.topk_pairs(5))
    tm.save_heatmap("traffic_matrix.png")
    


    # 测试并行分组功能
    groups = tm.group_by_parallelism(tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    log.info("并行分组示例 (TP=8, EP=1, SP=8, CP=1, DP=2, PP=2):")
    log.info("TP groups (前3组): %s", groups['tp_groups'][:3])
    log.info("EP groups (前3组): %s", groups['ep_groups'][:3])
    log.info("SP groups (前3组): %s", groups['sp_groups'][:3])
    log.info("CP groups (前3组): %s", groups['cp_groups'][:3])
    log.info("DP groups (前3组): %s", groups['dp_groups'][:3])
    log.info("PP groups (前3组): %s", groups['pp_groups'][:3])
    
    # 可视化分组
    # tm.visualize_parallelism_groups(tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)

    tm2 = TrafficMatrix(256)
    # tm2.add_intra_group_traffic("dp", 16*1024, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    tm2.add_intra_group_traffic("dp", 2**35, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # tm2.save_heatmap("tm2_traffic_matrix_dp.png")

    stats = tm.intragroup_traffic_stats(tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    log.info("SP 组统计： %s", stats["sp"])
    # -> 包含 total_intragroup / avg_per_group / avg_per_node_out 等字段

    # 2) 两个矩阵相加（返回新对象，饱和相加）
    tm_sum = tm + tm2

    # tm_sum.save_heatmap("tm_sum_traffic_matrix_sum.png")
    log.info("top5: %s", tm_sum.topk_pairs(5))

def test_TrafficMatrix_ep_all_to_all():
    tm = TrafficMatrix(256)
    # tm.add(1, 2, 1500)             # 1->2 加 1500
    # tm.add(1, 2, 500)              # 1->2 再加 500
    # tm.add_many([0, 5, 5], [3, 7, 7], [100, 200, 300])  # 批量

    # 在 SP 维度上：每个 SP 组（大小 8，总 32 组）内两两添加 4KB
    # tm.add_intra_group_traffic("dp", 1024, tp=2, ep=8, sp=4, cp=1, dp=2, pp=2)
    tm.add_intra_group_traffic_pair_at_group_bulk(
    "ep",
    triples=[
        [2048, 0, 2],  # 组内 0->2（若取第1个组即全局 16->20）
        [1024, 1, 3],
    ],
    group_index=1,
    tp=2, ep=8, sp=4, cp=1, dp=2, pp=2,
    )

    # 也可以一次对多个维度加不同量，比如：
    # tm.add_intra_group_traffic_bulk(
    #     {"tp": 8 * 1024, "sp": 4 * 1024, "dp": 2 * 1024, "pp": 1 * 1024},  # SP 组内每对 +4KB，DP 组内每对 +2KB，PP 组内每对 +2KB
    #     tp=8, ep=1, sp=8, cp=1, dp=2, pp=2
    # )

    total, out_arr, in_arr = tm.totals()
    log.info("total: %s", total)
    log.info("out(1): %s in(2): %s", tm.out_bytes(1), tm.in_bytes(2))
    log.info("top5: %s", tm.topk_pairs(5))
    tm.save_heatmap("traffic_matrix.png")
    


    # 测试并行分组功能
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
    """
    验证 add_intra_group_traffic_pair 在小规模 N=16、tp=4, sp=4 下的行为：
      - which="sp", bytes=100, src=0, dst=1
      - 期望生成 4 条边：0->4, 1->5, 2->6, 3->7，各自为 100
      - 其他位置应为 0
    """
    tm = TrafficMatrix(256)
    # tm.add_intra_group_traffic_pair("sp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # tm.add_intra_group_traffic_pair("tp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # tm.add_intra_group_traffic_pair("dp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    tm.add_intra_group_traffic_pair("dp", 100, src=0, dst=1, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    tm.add_intra_group_traffic_pair("dp", 100, src=1, dst=0, tp=8, ep=1, sp=8, cp=1, dp=2, pp=2)
    # expected = {(0, 4): 100, (1, 5): 100, (2, 6): 100, (3, 7): 100}
    # 校验期望边
    # for (s, d), v in expected.items():
    #     assert int(tm.counts[s, d]) == v, f"{s}->{d} 应为 {v}, 实际 {int(tm.counts[s, d])}"
    # # 校验总和
    # total, _, _ = tm.totals()
    # assert total == sum(expected.values()), f"总和应为 {sum(expected.values())}, 实际 {total}"
    # log.info("test_intra_group_pair_unit passed: total=%s", total)

    # tm.save_heatmap("tm_traffic_matrix_intra_group_pair_unit.png")


def test_traffic_matrix_intra_group_pair_bulk_unit():
    """
    验证 add_intra_group_traffic_pair_bulk 在 N=16、tp=4, sp=4 下的行为：
      - sp: bytes=200, src=2, dst=3  -> 8->12, 9->13, 10->14, 11->15
      - tp: bytes=50,  src=1, dst=0  -> 1->0, 5->4, 9->8, 13->12
      - 总和应为 4*200 + 4*50 = 1000
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
    #     assert int(tm.counts[s, d]) == v, f"{s}->{d} 应为 {v}, 实际 {int(tm.counts[s, d])}"
    # total, _, _ = tm.totals()
    # assert total == 1000, f"总和应为 1000, 实际 {total}"
    # log.info("test_intra_group_pair_bulk_unit passed: total=%s", total)
    # tm.save_heatmap("tm_traffic_matrix_intra_group_pair_bulk_unit.png")



if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,                              # 全局日志级别
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
