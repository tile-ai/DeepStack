from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


Mode = Literal["coarse", "fine", "roof"]


@dataclass
class Modeling_Granularity:
    mode: Mode
    comp_comm_overlap: bool
    auto_tune: bool
    dump_perf_log: bool = False   # 是否在 finalize 后自动 dump_log；默认关闭

    def __post_init__(self) -> None:
        if self.mode not in ("coarse", "fine", "roof"):
            raise ValueError("mode must be one of: 'coarse', 'fine', 'roof'")
        if not isinstance(self.comp_comm_overlap, bool):
            raise TypeError("comp_comm_overlap must be a bool")
        if not isinstance(self.auto_tune, bool):
            raise TypeError("auto_tune must be a bool")
        if not isinstance(self.dump_perf_log, bool):
            raise TypeError("dump_perf_log must be a bool")

    # getters
    def get_mode(self) -> Mode:
        return self.mode

    def get_comp_comm_overlap(self) -> bool:
        return self.comp_comm_overlap

    def get_auto_tune(self) -> bool:
        return self.auto_tune

    def get_dump_perf_log(self) -> bool:
        return self.dump_perf_log

    # setters with validation to avoid unsafe direct assignments
    def set_mode(self, mode: Mode) -> None:
        if mode not in ("coarse", "fine", "roof"):
            raise ValueError("mode must be one of: 'coarse', 'fine', 'roof'")
        self.mode = mode

    def set_comp_comm_overlap(self, comp_comm_overlap: bool) -> None:
        if not isinstance(comp_comm_overlap, bool):
            raise TypeError("comp_comm_overlap must be a bool")
        self.comp_comm_overlap = comp_comm_overlap

    def set_auto_tune(self, auto_tune: bool) -> None:
        if not isinstance(auto_tune, bool):
            raise TypeError("auto_tune must be a bool")
        self.auto_tune = auto_tune

    def set_dump_perf_log(self, dump_perf_log: bool) -> None:
        if not isinstance(dump_perf_log, bool):
            raise TypeError("dump_perf_log must be a bool")
        self.dump_perf_log = dump_perf_log

    # bulk update with validation
    def update(
        self,
        *,
        mode: Optional[Mode] = None,
        comp_comm_overlap: Optional[bool] = None,
        auto_tune: Optional[bool] = None,
        dump_perf_log: Optional[bool] = None,
    ) -> "Modeling_Granularity":
        if mode is not None:
            self.set_mode(mode)
        if comp_comm_overlap is not None:
            self.set_comp_comm_overlap(comp_comm_overlap)
        if auto_tune is not None:
            self.set_auto_tune(auto_tune)
        if dump_perf_log is not None:
            self.set_dump_perf_log(dump_perf_log)
        return self

    # immutable-style helpers
    def copy(self) -> "Modeling_Granularity":
        return Modeling_Granularity(
            mode=self.mode,
            comp_comm_overlap=self.comp_comm_overlap,
            auto_tune=self.auto_tune,
            dump_perf_log=self.dump_perf_log,
        )

    def copy_with(
        self,
        *,
        mode: Optional[Mode] = None,
        comp_comm_overlap: Optional[bool] = None,
        auto_tune: Optional[bool] = None,
        dump_perf_log: Optional[bool] = None,
    ) -> "Modeling_Granularity":
        return Modeling_Granularity(
            mode=self.mode if mode is None else mode,
            comp_comm_overlap=self.comp_comm_overlap if comp_comm_overlap is None else comp_comm_overlap,
            auto_tune=self.auto_tune if auto_tune is None else auto_tune,
            dump_perf_log=self.dump_perf_log if dump_perf_log is None else dump_perf_log,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "comp_comm_overlap": self.comp_comm_overlap,
            "auto_tune": self.auto_tune,
            "dump_perf_log": self.dump_perf_log,
        }
