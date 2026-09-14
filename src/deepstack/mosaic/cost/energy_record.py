from dataclasses import dataclass
from typing import Optional, Tuple
from mosaic.noc.traffic_matrix import TrafficMatrix

@dataclass
class EnergyRecord:
    """Inputs required for one EnergyModel.get_energy calculation."""

    # --- Network traffic and topology ---
    tm: TrafficMatrix                        # TrafficMatrix
    # h: Any                        # Hierarchy

    # --- Memory Read/Write: bytes read and written at each level (read, write) ---
    reg_rw: Tuple[float, float] = (0.0, 0.0)
    smem_rw: Tuple[float, float] = (0.0, 0.0)
    l2_rw: Tuple[float, float] = (0.0, 0.0)
    dram_rw: Tuple[float, float] = (0.0, 0.0)

    # --- Compute ---
    sfu_ops: float = 0.0
    cuda_ops: float = 0.0
    tensor_ops: float = 0.0

    # # --- Optional additional information ---
    # name: str = ""                          # Custom label.
    # metadata: dict = field(default_factory=dict)  # Additional information.

    def __init__(self, num_nodes: int) -> None:
        """Initialize a TrafficMatrix with N=num_nodes and set all memory-access and compute counters to zero."""
        self.tm = TrafficMatrix(num_nodes)
        # Memory R/W defaults to 0.
        self.reg_rw = (0.0, 0.0)
        self.smem_rw = (0.0, 0.0)
        self.l2_rw = (0.0, 0.0)
        self.dram_rw = (0.0, 0.0)
        # Compute defaults to 0.
        self.sfu_ops = 0.0
        self.cuda_ops = 0.0
        self.tensor_ops = 0.0

    # ======== Convenience interface: merge an external TrafficMatrix ========
    def add_tm(self, tm_to_add: TrafficMatrix, *, saturating: bool = True) -> None:
        """Merge another TrafficMatrix into this record.
        Use saturating addition when saturating=True; otherwise use ordinary wrapping addition.
        """
        if not isinstance(tm_to_add, TrafficMatrix):
            raise TypeError("tm_to_add 必须是 TrafficMatrix")
        self.tm.add_matrix(tm_to_add, saturating=saturating, in_place=True)

    # ======== Convenience interface: add Memory R/W ========
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
        """Increment the supplied memory read/write byte counters.
        For example: add_memory_rw(reg=(1e6, 2e6), dram=(1e9, 1e9)).
        """
        if reg is not None:
            self.add_reg_rw(reg[0], reg[1])
        if smem is not None:
            self.add_smem_rw(smem[0], smem[1])
        if l2 is not None:
            self.add_l2_rw(l2[0], l2[1])
        if dram is not None:
            self.add_dram_rw(dram[0], dram[1])

    # ======== Convenience interface: add Compute Ops ========
    def add_compute_ops(self, sfu_ops: float = 0.0, cuda_ops: float = 0.0, tensor_ops: float = 0.0) -> None:
        self.sfu_ops = float(self.sfu_ops) + float(sfu_ops)
        self.cuda_ops = float(self.cuda_ops) + float(cuda_ops)
        self.tensor_ops = float(self.tensor_ops) + float(tensor_ops)

    # ======== Aggregation convenience interface: accumulate Compute and Memory together ========
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
        """Increment memory-access and compute counters using keyword arguments.
        Only supplied fields are updated; omitted fields remain unchanged.
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
    
    # ======== Convenience interface: add two EnergyRecords in place ========
    def add_energy_record(self, other: "EnergyRecord", *, saturating: bool = True) -> None:
        """Accumulate another EnergyRecord in place.
        Merge tm using add_matrix, optionally with saturation. Add reg_rw, smem_rw, l2_rw,
        and dram_rw elementwise, and add sfu_ops, cuda_ops, and tensor_ops individually.
        Both traffic matrices must have the same size N.
        """
        if not isinstance(other, EnergyRecord):
            raise TypeError("other 必须是 EnergyRecord")
        if int(self.tm.N) != int(other.tm.N):
            raise ValueError(f"TrafficMatrix 大小不一致：{self.tm.N} vs {other.tm.N}")
        # Accumulate tm.
        self.tm.add_matrix(other.tm, saturating=saturating, in_place=True)
        # Accumulate Memory.
        self.reg_rw = (float(self.reg_rw[0]) + float(other.reg_rw[0]),
                       float(self.reg_rw[1]) + float(other.reg_rw[1]))
        self.smem_rw = (float(self.smem_rw[0]) + float(other.smem_rw[0]),
                        float(self.smem_rw[1]) + float(other.smem_rw[1]))
        self.l2_rw = (float(self.l2_rw[0]) + float(other.l2_rw[0]),
                      float(self.l2_rw[1]) + float(other.l2_rw[1]))
        self.dram_rw = (float(self.dram_rw[0]) + float(other.dram_rw[0]),
                        float(self.dram_rw[1]) + float(other.dram_rw[1]))
        # Accumulate Compute.
        self.sfu_ops = float(self.sfu_ops) + float(other.sfu_ops)
        self.cuda_ops = float(self.cuda_ops) + float(other.cuda_ops)
        self.tensor_ops = float(self.tensor_ops) + float(other.tensor_ops)
