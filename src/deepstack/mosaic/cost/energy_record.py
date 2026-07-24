from dataclasses import dataclass
from typing import Optional, Tuple
from mosaic.noc.traffic_matrix import TrafficMatrix

@dataclass
class EnergyRecord:
    """
    数据结构：记录一次 ``EnergyModel.get_energy`` 计算所需的输入。
    """

    # --- 网络流量与拓扑 ---
    tm: TrafficMatrix                        # TrafficMatrix
    # h: Any                        # Hierarchy

    # --- Memory Read/Write: 各层读写字节数 (read, write) ---
    reg_rw: Tuple[float, float] = (0.0, 0.0)
    smem_rw: Tuple[float, float] = (0.0, 0.0)
    l2_rw: Tuple[float, float] = (0.0, 0.0)
    dram_rw: Tuple[float, float] = (0.0, 0.0)

    # --- Compute ---
    sfu_ops: float = 0.0
    cuda_ops: float = 0.0
    tensor_ops: float = 0.0

    # # --- 可选附加信息 ---
    # name: str = ""                          # 自定义标签
    # metadata: dict = field(default_factory=dict)  # 额外信息

    def __init__(self, num_nodes: int) -> None:
        """
        使用节点数量初始化：
          - 创建一个 N=num_nodes 的 TrafficMatrix
          - 其余读写与计算字段初始化为 0
        """
        self.tm = TrafficMatrix(num_nodes)
        # Memory R/W 默认 0
        self.reg_rw = (0.0, 0.0)
        self.smem_rw = (0.0, 0.0)
        self.l2_rw = (0.0, 0.0)
        self.dram_rw = (0.0, 0.0)
        # Compute 默认 0
        self.sfu_ops = 0.0
        self.cuda_ops = 0.0
        self.tensor_ops = 0.0

    # ======== 便捷接口：合并外部 TrafficMatrix ========
    def add_tm(self, tm_to_add: TrafficMatrix, *, saturating: bool = True) -> None:
        """
        将另一个 TrafficMatrix 合并到当前记录中。
        - saturating=True：饱和相加；False：普通回绕相加
        """
        if not isinstance(tm_to_add, TrafficMatrix):
            raise TypeError("tm_to_add 必须是 TrafficMatrix")
        self.tm.add_matrix(tm_to_add, saturating=saturating, in_place=True)

    # ======== 便捷接口：对 Memory R/W 做加法 ========
    def add_reg_rw(self, read: float = 0.0, write: float = 0.0) -> None:
        r0, w0 = self.reg_rw
        self.reg_rw = (float(r0) + float(read), float(w0) + float(write))

    def add_smem_rw(self, read: float = 0.0, write: float = 0.0) -> None:
        r0, w0 = self.smem_rw
        self.smem_rw = (float(r0) + float(read), float(w0) + float(write))

    def add_l2_rw(self, read: float = 0.0, write: float = 0.0) -> None:
        r0, w0 = self.l2_rw
        self.l2_rw = (float(r0) + float(read), float(w0) + float(write))

    def add_dram_rw(self, read: float = 0.0, write: float = 0.0) -> None:
        r0, w0 = self.dram_rw
        self.dram_rw = (float(r0) + float(read), float(w0) + float(write))

    def add_memory_rw(
        self,
        reg: Optional[Tuple[float, float]] = None,
        smem: Optional[Tuple[float, float]] = None,
        l2: Optional[Tuple[float, float]] = None,
        dram: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        批量对各层读写字节数做加法；仅对传入的项进行累加。
        例如：add_memory_rw(reg=(1e6, 2e6), dram=(1e9, 1e9))
        """
        if reg is not None:
            self.add_reg_rw(reg[0], reg[1])
        if smem is not None:
            self.add_smem_rw(smem[0], smem[1])
        if l2 is not None:
            self.add_l2_rw(l2[0], l2[1])
        if dram is not None:
            self.add_dram_rw(dram[0], dram[1])

    # ======== 便捷接口：对 Compute Ops 做加法 ========
    def add_compute_ops(self, sfu_ops: float = 0.0, cuda_ops: float = 0.0, tensor_ops: float = 0.0) -> None:
        self.sfu_ops = float(self.sfu_ops) + float(sfu_ops)
        self.cuda_ops = float(self.cuda_ops) + float(cuda_ops)
        self.tensor_ops = float(self.tensor_ops) + float(tensor_ops)

    # ======== 聚合便捷接口：同时累加 Compute 与 Memory ========
    def add_compute_memory(
        self,
        *,
        # reg: Optional[Tuple[float, float]] = None,
        smem: Optional[Tuple[float, float]] = None,
        l2: Optional[Tuple[float, float]] = None,
        dram: Optional[Tuple[float, float]] = None,
        sfu_ops: float = 0.0,
        cuda_ops: float = 0.0,
        tensor_ops: float = 0.0,
    ) -> None:
        """
        一次性对内存读写与计算量做增量更新（关键字参数形式，便于可读）。
        仅对传入的项进行累加；未提供的项不变。
        """
        # Memory
        # if reg is not None:
            # self.add_reg_rw(reg[0], reg[1])

        if smem is not None:
            self.add_smem_rw(smem[0], smem[1])
        else:
            smem = (0, 0)

        if l2 is not None:
            self.add_l2_rw(l2[0], l2[1])
        else:
            l2 = (0, 0)
        if dram is not None:
            self.add_dram_rw(dram[0], dram[1])
        # Compute
        if sfu_ops or cuda_ops or tensor_ops:
            self.add_compute_ops(sfu_ops=sfu_ops, cuda_ops=cuda_ops, tensor_ops=tensor_ops)

        compute_reg_ops = sfu_ops + cuda_ops + tensor_ops / 2
        reg_read = max(smem[0] - l2[0], 0) + compute_reg_ops
        reg_write = max(smem[1] - l2[1], 0) + compute_reg_ops
        self.add_reg_rw(reg_read, reg_write)
    
    # ======== 便捷接口：两个 EnergyRecord 相加（就地累加） ========
    def add_energy_record(self, other: "EnergyRecord", *, saturating: bool = True) -> None:
        """
        将另一个 EnergyRecord 的以下字段累加到当前对象（就地）：
        - tm（TrafficMatrix）：调用 add_matrix（可选饱和相加）
        - reg_rw / smem_rw / l2_rw / dram_rw：元素级相加
        - sfu_ops / cuda_ops / tensor_ops：逐项相加
        要求两者的 TrafficMatrix 规模一致（N 相同）。
        """
        if not isinstance(other, EnergyRecord):
            raise TypeError("other 必须是 EnergyRecord")
        if int(self.tm.N) != int(other.tm.N):
            raise ValueError(f"TrafficMatrix 大小不一致：{self.tm.N} vs {other.tm.N}")
        # tm 累加
        self.tm.add_matrix(other.tm, saturating=saturating, in_place=True)
        # Memory 累加
        self.reg_rw = (float(self.reg_rw[0]) + float(other.reg_rw[0]),
                       float(self.reg_rw[1]) + float(other.reg_rw[1]))
        self.smem_rw = (float(self.smem_rw[0]) + float(other.smem_rw[0]),
                        float(self.smem_rw[1]) + float(other.smem_rw[1]))
        self.l2_rw = (float(self.l2_rw[0]) + float(other.l2_rw[0]),
                      float(self.l2_rw[1]) + float(other.l2_rw[1]))
        self.dram_rw = (float(self.dram_rw[0]) + float(other.dram_rw[0]),
                        float(self.dram_rw[1]) + float(other.dram_rw[1]))
        # Compute 累加
        self.sfu_ops = float(self.sfu_ops) + float(other.sfu_ops)
        self.cuda_ops = float(self.cuda_ops) + float(other.cuda_ops)
        self.tensor_ops = float(self.tensor_ops) + float(other.tensor_ops)
