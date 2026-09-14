"""Deterministic FIFO reuse-distance cache model.

TileSight's existing reuse-distance models use a random permutation within a
CTA wave and turn reuse distance into a probabilistic SDCM hit rate.  That is
useful for set-associative sensitivity studies, but it is deliberately not
used here: the bank model needs a reproducible yes/no decision before a
logical sector is decoded into a DRAM bank.

The GEMM integration models a fully shared L2 as a FIFO of logical cache-line
IDs.  Every configured line in a tile is probed independently, so two tiles may
fully or partially overlap and only their missing lines continue to DRAM.  A
miss inserts the line and evicts the oldest line at capacity; a hit does not
refresh FIFO order.  Consequently a resident line whose FIFO age is within
capacity is a deterministic hit.  Cache sets and associativity are
deliberately outside this simple fully-associative model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Hashable, Optional, Tuple


@dataclass(frozen=True)
class L2CacheSpec:
    """Configuration for the deterministic L2 model.

    ``capacity_bytes`` and ``line_bytes`` are physical byte counts.  Bank-model
    integration requires the L2 line size to match the configured DRAM sector
    size.  Output writes remain on the
    existing DDR write path; ``output_pressure`` only decides whether those
    output tiles occupy L2 and age resident input tiles.
    """

    capacity_bytes: int
    line_bytes: int = 128
    policy: str = "fifo"
    output_pressure: bool = True

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0:
            raise ValueError("L2 capacity_bytes must be positive")
        if self.line_bytes <= 0:
            raise ValueError("L2 line_bytes must be positive")
        if self.capacity_bytes % self.line_bytes:
            raise ValueError("L2 capacity must contain an integer number of lines")
        if self.policy != "fifo":
            raise ValueError("the deterministic L2 prototype supports only fifo")

    @property
    def capacity_lines(self) -> int:
        return self.capacity_bytes // self.line_bytes

    @classmethod
    def from_arch(
        cls,
        arch: Any,
        *,
        line_bytes: int = 128,
        output_pressure: bool = True,
    ) -> "L2CacheSpec":
        if not hasattr(arch, "l2_capacity"):
            raise ValueError("architecture does not define l2_capacity")
        return cls(
            capacity_bytes=int(arch.l2_capacity),
            line_bytes=line_bytes,
            output_pressure=output_pressure,
        )


@dataclass(frozen=True)
class FifoAccessResult:
    """One deterministic FIFO lookup result.

    ``reuse_distance_lines`` is the number of line insertions from the start
    of the block's previous FIFO insertion to this lookup.  It is ``None`` on
    a cold miss.  For a hit it is necessarily no larger than cache capacity;
    equality is a hit until the next insertion evicts the oldest block.
    """

    hit: bool
    cold: bool
    footprint_lines: int
    reuse_distance_lines: Optional[int]


@dataclass(frozen=True)
class L2OperandStats:
    """Deterministic request- and cache-line-level statistics.

    ``hits`` counts requests for which every line hit.  ``partial_hits``
    counts requests containing both hits and misses, and ``misses`` counts all
    requests that send at least one line to DDR.  The model intentionally
    avoids retaining an unbounded history of evicted tags, so misses are not
    split into compulsory versus capacity classes.
    """

    accesses: int
    hits: int
    partial_hits: int
    misses: int
    accessed_lines: int
    hit_lines: int
    miss_lines: int

    @property
    def hit_rate(self) -> float:
        if self.accesses == 0:
            return 0.0
        return self.hits / self.accesses

    @property
    def line_hit_rate(self) -> float:
        if self.accessed_lines == 0:
            return 0.0
        return self.hit_lines / self.accessed_lines

    @property
    def ddr_miss_rate(self) -> float:
        if self.accessed_lines == 0:
            return 0.0
        return self.miss_lines / self.accessed_lines


@dataclass(frozen=True)
class GemmL2Result:
    """Operator-level L2 result retained by :class:`GemmBankResult`."""

    spec: L2CacheSpec
    a: L2OperandStats
    b: L2OperandStats
    output_pressure: L2OperandStats
    peak_resident_lines: int
    final_resident_lines: int
    trace_accesses: int

    @property
    def input_line_hit_rate(self) -> float:
        accessed = self.a.accessed_lines + self.b.accessed_lines
        if accessed == 0:
            return 0.0
        return (self.a.hit_lines + self.b.hit_lines) / accessed

    @property
    def input_ddr_miss_rate(self) -> float:
        return 1.0 - self.input_line_hit_rate


class DeterministicFifoCache:
    """Mutable FIFO with line-fast and generic block access methods."""

    def __init__(self, spec: L2CacheSpec) -> None:
        self.spec = spec
        self._queue: Deque[Hashable] = deque()
        # key -> (footprint lines, insertion start on the monotonic line clock)
        self._resident: Dict[Hashable, Tuple[int, int]] = {}
        self._last_insertion: Dict[Hashable, int] = {}
        self._resident_lines = 0
        self._inserted_lines = 0
        self._peak_resident_lines = 0
        self._last_line_sequence = -1

    @property
    def resident_lines(self) -> int:
        return self._resident_lines

    @property
    def peak_resident_lines(self) -> int:
        return self._peak_resident_lines

    @property
    def resident_keys(self) -> frozenset:
        """Immutable resident-tag snapshot for concurrent epoch replay."""

        return frozenset(self._resident)

    @property
    def oldest_resident_sequence(self) -> int:
        if not self._queue:
            return self._inserted_lines
        return self._resident[self._queue[0]][1]

    @property
    def last_line_sequence(self) -> int:
        return self._last_line_sequence

    def probe_line(self, key: Hashable) -> bool:
        """Return whether one line is resident without changing FIFO state.

        Distributed request replay needs all probes in one logical epoch to
        observe the same cache snapshot.  Keeping probe and fill separate
        also prevents a second concurrent requester from being reported as a
        cache hit merely because the first request inserted the line while
        the epoch was being walked.

        A probe deliberately does not update ``last_line_sequence``.  That
        field belongs to the immediate ``access_line`` fast path and
        its resident-request memoization.
        """

        return key in self._resident

    def fill_line(self, key: Hashable) -> Optional[Hashable]:
        """Insert one line after a miss and return the evicted key, if any.

        Filling an already resident line is idempotent and does not refresh
        FIFO order.  This is useful when several requests were merged behind
        one in-flight miss and all complete in the same logical epoch.
        """

        resident = self._resident.get(key)
        if resident is not None:
            self._last_line_sequence = resident[1]
            return None

        victim: Optional[Hashable] = None
        if self._resident_lines >= self.spec.capacity_lines:
            victim = self._queue.popleft()
            victim_lines, _ = self._resident.pop(victim)
            self._resident_lines -= victim_lines

        insertion_start = self._inserted_lines
        self._queue.append(key)
        self._resident[key] = (1, insertion_start)
        self._last_line_sequence = insertion_start
        self._resident_lines += 1
        self._inserted_lines += 1
        self._peak_resident_lines = max(
            self._peak_resident_lines, self._resident_lines
        )
        return victim

    def clone(self) -> "DeterministicFifoCache":
        """Return an independent cache with exactly the same FIFO state."""

        cloned = DeterministicFifoCache(self.spec)
        cloned._queue = deque(self._queue)
        cloned._resident = dict(self._resident)
        cloned._last_insertion = dict(self._last_insertion)
        cloned._resident_lines = self._resident_lines
        cloned._inserted_lines = self._inserted_lines
        cloned._peak_resident_lines = self._peak_resident_lines
        cloned._last_line_sequence = self._last_line_sequence
        return cloned

    def access_line(self, key: Hashable) -> int:
        """Fast one-line access: 0=hit, 1=miss.

        Unlike generic block diagnostics, this hot path retains only resident
        tags.  Memory therefore scales with cache capacity rather than the
        operator's complete address history.
        """

        resident = self._resident.get(key)
        if resident is not None:
            self._last_line_sequence = resident[1]
            return 0
        self.fill_line(key)
        return 1

    def access(self, key: Hashable, footprint_lines: int) -> FifoAccessResult:
        """Access one indivisible tile block.

        Hits do not change the queue.  Oversized blocks stream through the
        cache, evict all prior state, and are not retained as a partial tile.
        """

        lines = int(footprint_lines)
        if lines <= 0:
            raise ValueError("FIFO block footprint must be positive")

        resident = self._resident.get(key)
        if resident is not None:
            resident_lines, insertion_start = resident
            if resident_lines != lines:
                raise ValueError("one FIFO key cannot change footprint")
            distance = self._inserted_lines - insertion_start
            if distance > self.spec.capacity_lines:
                raise AssertionError("resident FIFO block exceeded capacity age")
            return FifoAccessResult(True, False, lines, distance)

        previous = self._last_insertion.get(key)
        distance = (
            None if previous is None else self._inserted_lines - previous
        )
        cold = previous is None

        if lines > self.spec.capacity_lines:
            self._queue.clear()
            self._resident.clear()
            self._resident_lines = 0
            self._last_insertion[key] = self._inserted_lines
            self._inserted_lines += lines
            return FifoAccessResult(False, cold, lines, distance)

        while self._resident_lines + lines > self.spec.capacity_lines:
            victim = self._queue.popleft()
            victim_lines, _ = self._resident.pop(victim)
            self._resident_lines -= victim_lines

        insertion_start = self._inserted_lines
        self._queue.append(key)
        self._resident[key] = (lines, insertion_start)
        self._last_insertion[key] = insertion_start
        self._resident_lines += lines
        self._inserted_lines += lines
        self._peak_resident_lines = max(
            self._peak_resident_lines, self._resident_lines
        )
        return FifoAccessResult(False, cold, lines, distance)


class OperandStatsAccumulator:
    """Small internal helper shared by the GEMM trace builder and tests."""

    def __init__(self) -> None:
        self.accesses = 0
        self.hits = 0
        self.partial_hits = 0
        self.misses = 0
        self.accessed_lines = 0
        self.hit_lines = 0
        self.miss_lines = 0

    def add(self, result: FifoAccessResult) -> None:
        self.accesses += 1
        self.accessed_lines += result.footprint_lines
        if result.hit:
            self.hits += 1
            self.hit_lines += result.footprint_lines
        else:
            self.misses += 1
            self.miss_lines += result.footprint_lines

    def add_line_request(
        self,
        *,
        total_lines: int,
        hit_lines: int,
    ) -> None:
        miss_lines = total_lines - hit_lines
        if total_lines <= 0 or hit_lines + miss_lines != total_lines:
            raise ValueError("invalid line-request accounting")
        self.accesses += 1
        self.accessed_lines += total_lines
        self.hit_lines += hit_lines
        self.miss_lines += miss_lines
        if miss_lines == 0:
            self.hits += 1
        else:
            self.misses += 1
            if hit_lines:
                self.partial_hits += 1

    def freeze(self) -> L2OperandStats:
        return L2OperandStats(
            accesses=self.accesses,
            hits=self.hits,
            partial_hits=self.partial_hits,
            misses=self.misses,
            accessed_lines=self.accessed_lines,
            hit_lines=self.hit_lines,
            miss_lines=self.miss_lines,
        )
