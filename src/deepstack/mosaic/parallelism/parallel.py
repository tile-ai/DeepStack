from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
import math
from itertools import product
import logging

log = logging.getLogger(__name__) 

@dataclass
class GlobalParallel:
    dp: int = 1
    pp: int = 1
    fsdp: bool = False

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
        if not isinstance(other, ParallelScheme):
            return False
        return self.equal(other)

    def __str__(self) -> str:
        # return f"tp={self.tp}, ep={self.ep}, sp={self.sp}, cp={self.cp}, dp={self.dp}, fsdp={self.fsdp}, pp={self.pp}"
        if self.ep1 is not None and self.ep2 is not None:
            return f"tp={self.tp}, ep={self.ep}, ep1={self.ep1}, ep2={self.ep2}, sp={self.sp}, cp={self.cp}, dp={self.dp}, fsdp={self.fsdp}, pp={self.pp}"
        else:
            return f"tp={self.tp}, ep={self.ep}, sp={self.sp}, cp={self.cp}, dp={self.dp}, fsdp={self.fsdp}, pp={self.pp}"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.dp, self.pp, self.tp, self.cp, self.sp, self.ep, self.fsdp))