from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
import math
from itertools import product
import logging

log = logging.getLogger(__name__) 

@dataclass
class GlobalParallel:
    dp: int = 1            # Data Parallel（批次维切，全局）
    pp: int = 1            # Pipeline Parallel（层级切，全局）
    fsdp: bool = False     # FSDP 是 DP 的可选特性（是否在 DP 内做完全分片）
    # 注：tp/cp/sp/ep 不再是全局配置，而是 per-op 的搜索域

    def world_size(self) -> int:
        return self.dp * self.pp

@dataclass
class LLM_Parallel:
    swiglu_para: ParallelScheme
    rms_norm_para: ParallelScheme
    mla_para: ParallelScheme
    gqa_para: ParallelScheme
    add_residual_para: ParallelScheme
    rope_para: ParallelScheme
    moe_para: ParallelScheme

@dataclass()
class ParallelScheme:

    tp: int = 1
    ep: int = 1
    sp: int = 1
    cp: int = 1
    dp: int = 1
    fsdp: bool = False
    pp: int = 1
    ep1: Optional[int] = None # ep1 sharding bs
    ep2: Optional[int] = None # ep2 sharding seq

    # return f"tp={self.tp}, ep={self.ep}, sp={self.sp}, cp={self.cp}, dp={self.dp}, fsdp={self.fsdp}, pp={self.pp}"

    def __post_init__(self) -> None:
        # ep1*ep2 也可以小于ep，因为有时候shard_seq*shard_bs < ep,会可以少duplicate 一些

        if self.ep1 is not None or self.ep2 is not None:
            part1 = self.ep1 if self.ep1 is not None else 1
            part2 = self.ep2 if self.ep2 is not None else 1
            # assert part1 * part2 == self.ep, "ep1 * ep2 != ep"
            # self.ep = part1 * part2
            if part1 * part2 != self.ep:
                log.warning("ep1 * ep2 != ep, ep1: %s, ep2: %s, ep: %s", part1, part2, self.ep)
            

    def world_size(self) -> int:
        return self.dp * self.pp * self.tp * self.cp * self.sp * self.ep
    
    def as_axes(self) -> Dict[str, int]:
        return {"dp": self.dp, "pp": self.pp, "fsdp": self.fsdp, "tp": self.tp, "cp": self.cp, "sp": self.sp, "ep": self.ep}
    
    def as_dict(self) -> Dict[str, int]:
        return {"dp": self.dp, "pp": self.pp, "fsdp": self.fsdp, "tp": self.tp, "cp": self.cp, "sp": self.sp, "ep": self.ep}

    def equal(self, other: "ParallelScheme") -> bool:
        # 忽略 fsdp，仅比较其余并行维度是否一致
        if not isinstance(other, ParallelScheme):
            return False
        return (
            self.dp == other.dp
            and self.pp == other.pp
            and self.tp == other.tp
            and self.cp == other.cp
            and self.sp == other.sp
            and self.ep == other.ep
            and self.fsdp == other.fsdp
        )

    def strong_equal(self, other: "ParallelScheme") -> bool:
        # 强相等：包含 fsdp 在内的所有维度都需要一致
        if not isinstance(other, ParallelScheme):
            return False
        return (
            self.dp == other.dp
            and self.pp == other.pp
            and self.tp == other.tp
            and self.cp == other.cp
            and self.sp == other.sp
            and self.ep == other.ep
            and self.fsdp == other.fsdp
        )

    def __eq__(self, other: object) -> bool:
        # 使得 a == b 忽略 fsdp，仅比较其余维度
        if not isinstance(other, ParallelScheme):
            return False
        return self.equal(other)

    def __str__(self) -> str:
        # 自定义打印顺序：tp, ep, sp, cp, dp, pp, fsdp
        # return f"tp={self.tp}, ep={self.ep}, sp={self.sp}, cp={self.cp}, dp={self.dp}, fsdp={self.fsdp}, pp={self.pp}"
        if self.ep1 is not None and self.ep2 is not None:
            return f"tp={self.tp}, ep={self.ep}, ep1={self.ep1}, ep2={self.ep2}, sp={self.sp}, cp={self.cp}, dp={self.dp}, fsdp={self.fsdp}, pp={self.pp}"
        else:
            return f"tp={self.tp}, ep={self.ep}, sp={self.sp}, cp={self.cp}, dp={self.dp}, fsdp={self.fsdp}, pp={self.pp}"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        # 与 __eq__ 一致：忽略 fsdp，仅比较/哈希以下维度
        return hash((self.dp, self.pp, self.tp, self.cp, self.sp, self.ep, self.fsdp))