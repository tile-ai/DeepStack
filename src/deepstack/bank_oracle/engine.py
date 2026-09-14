"""Fast sector histogram and bank service model."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .layout import BankSwizzle
from .spec import DramBankSpec


def _make_popcount8() -> np.ndarray:
    values = np.arange(1 << 8, dtype=np.uint8)
    return np.unpackbits(values[:, None], axis=1).sum(axis=1).astype(np.uint8)


_POPCOUNT8 = _make_popcount8()


def _popcount_masks(masks: np.ndarray) -> np.ndarray:
    """Return the population count of each unsigned 64-bit row mask."""

    values = np.asarray(masks, dtype=np.uint64)
    if values.size == 0:
        return np.empty(0, dtype=np.uint8)
    byte_view = values.reshape(-1).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT8[byte_view].sum(axis=1, dtype=np.uint16)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@dataclass(frozen=True)
class BankHistogram:
    """One row job per unique (request, bank, row) after sector merging."""

    banks: np.ndarray
    rows: np.ndarray
    sector_masks: np.ndarray
    request_ids: np.ndarray

    @property
    def job_count(self) -> int:
        return int(self.banks.size)

    @property
    def sector_count(self) -> int:
        if self.sector_masks.size == 0:
            return 0
        return int(_popcount_masks(self.sector_masks).sum())


@dataclass(frozen=True)
class BankServiceResult:
    """Service result for one memory wave.

    ``bank_cycles`` and ``active_banks`` describe physical row buffers.  The
    scalar ``cycles`` additionally includes shared-port scheduling and the
    topology connectivity cap; ``overall_efficiency`` is therefore normalized
    by TSV ports, i.e. it is the achieved fraction of raw interface bandwidth.
    """

    cycles: float
    bank_cycles: np.ndarray
    row_job_cycles: np.ndarray
    sector_count: int
    transferred_bytes: int
    active_banks: int
    transaction_efficiency: float
    bank_balance_efficiency: float
    overall_efficiency: float
    fixed_job_lower_bound_cycles: float
    sector_lower_bound_cycles: float

    @property
    def bank_attainment(self) -> float:
        if self.cycles <= 0:
            return 1.0
        return self.sector_lower_bound_cycles / self.cycles


def decode_sector_ids(
    sector_ids: np.ndarray,
    spec: DramBankSpec,
    *,
    connectivity: str = "direct",
    swizzle: BankSwizzle = BankSwizzle(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode logical sectors to flattened (bank, row, sector-in-row)."""

    q = np.asarray(sector_ids, dtype=np.int64)
    if np.any(q < 0):
        raise ValueError("sector IDs must be non-negative")
    if connectivity == "direct":
        block, sector = np.divmod(q, spec.sectors_per_row)
        bank = block % spec.bank_count
        row = block // spec.bank_count
    elif connectivity == "interleaved":
        bank = q % spec.bank_count
        stream = q // spec.bank_count
        row, sector = np.divmod(stream, spec.sectors_per_row)
    else:
        raise ValueError("connectivity must be direct or interleaved")
    bank = swizzle.map_bank(bank, row, sector, spec)
    return bank.astype(np.int64), row.astype(np.int64), sector.astype(np.int64)


def build_bank_histogram(
    requests: Sequence[np.ndarray],
    spec: DramBankSpec,
    *,
    connectivity: str = "direct",
    swizzle: BankSwizzle = BankSwizzle(),
    coalesce_scope: str = "request",
) -> BankHistogram:
    """Map sector requests and merge duplicate sectors into row masks.

    ``request`` merges only within each tile/request.  ``wave`` permits all
    requests in the modeled memory wave to coalesce, representing an L2/MSHR
    de-duplicated trace.  The choice is explicit because cross-CTA reuse must
    not silently disappear inside the bank model.
    """

    if coalesce_scope not in {"request", "wave"}:
        raise ValueError("coalesce_scope must be request or wave")
    sectors_parts: List[np.ndarray] = []
    request_parts: List[np.ndarray] = []
    for request_id, request in enumerate(requests):
        sectors = np.unique(np.asarray(request, dtype=np.int64))
        if sectors.size == 0:
            continue
        sectors_parts.append(sectors)
        rid = 0 if coalesce_scope == "wave" else request_id
        request_parts.append(np.full(sectors.size, rid, dtype=np.int64))
    if not sectors_parts:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_u64 = np.empty(0, dtype=np.uint64)
        return BankHistogram(empty_i64, empty_i64, empty_u64, empty_i64)

    sectors = np.concatenate(sectors_parts)
    request_ids = np.concatenate(request_parts)
    banks, rows, offsets = decode_sector_ids(
        sectors, spec, connectivity=connectivity, swizzle=swizzle)
    masks = np.left_shift(
        np.uint64(1),
        offsets.astype(np.uint64),
    ).astype(np.uint64)

    # Lexicographic unique key.  request IDs are deliberately part of the key
    # unless wave-wide coalescing was requested.
    order = np.lexsort((rows, banks, request_ids))
    request_ids = request_ids[order]
    banks = banks[order]
    rows = rows[order]
    masks = masks[order]
    starts = np.empty(masks.size, dtype=bool)
    starts[0] = True
    starts[1:] = (
        (request_ids[1:] != request_ids[:-1])
        | (banks[1:] != banks[:-1])
        | (rows[1:] != rows[:-1])
    )
    indices = np.flatnonzero(starts)
    merged_masks = np.bitwise_or.reduceat(masks, indices)
    return BankHistogram(
        banks=banks[indices],
        rows=rows[indices],
        sector_masks=merged_masks,
        request_ids=request_ids[indices],
    )


def _capacity_per_bank(target_cycles: int, spec: DramBankSpec) -> int:
    """Maximum sectors one bank can serve within target_cycles."""

    if target_cycles < spec.recharge_cycles + spec.sector_cycles:
        return 0
    full_rows, remainder = divmod(target_cycles, spec.full_row_cycles)
    sectors = full_rows * spec.sectors_per_row
    if remainder >= spec.recharge_cycles + spec.sector_cycles:
        sectors += min(
            spec.sectors_per_row,
            (remainder - spec.recharge_cycles) // spec.sector_cycles,
        )
    return sectors


def _physical_bank_sector_lower_bound_cycles(
    sector_count: int,
    spec: DramBankSpec,
) -> int:
    """Sector-packing bound with freely assigned physical row buffers."""

    if sector_count <= 0:
        return 0
    low = 0
    high = spec.full_row_cycles * (
        (sector_count + spec.physical_bank_count - 1)
        // spec.physical_bank_count
        + 1
    )
    while (
        _capacity_per_bank(high, spec) * spec.physical_bank_count
        < sector_count
    ):
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if (
            _capacity_per_bank(mid, spec) * spec.physical_bank_count
            >= sector_count
        ):
            high = mid
        else:
            low = mid + 1
    return low


def sector_parallel_lower_bound_cycles(
    sector_count: int,
    spec: DramBankSpec,
) -> float:
    """Absolute lower bound allowing optimal packing and bank assignment.

    It combines three independent relaxations: physical-bank row packing,
    aggregate port data throughput, and the connected-vs-stacked
    steady-state cap.  It is independent of a particular candidate layout.
    """

    if sector_count <= 0:
        return 0
    physical_bank_bound = _physical_bank_sector_lower_bound_cycles(
        sector_count, spec
    )
    data_cycles = spec.sector_cycles * sector_count
    # Sector transfers are indivisible caller-configured quanta on a port.
    port_data_bound = (
        spec.sector_cycles * _ceil_div(sector_count, spec.port_count)
    )
    connectivity_bound = data_cycles / (
        spec.port_count * spec.connectivity_efficiency
    )
    return float(max(
        physical_bank_bound,
        port_data_bound,
        connectivity_bound,
    ))


def fixed_job_parallel_lower_bound_cycles(
    row_job_cycles: np.ndarray,
    bank_count: int,
) -> int:
    """Strict lower bound for freely assigning indivisible row jobs to banks.

    A continuous ``sum(jobs) / bank_count`` bound can be too optimistic when
    row activations are indivisible.

    Besides the total-work and largest-job bounds, use a floor-quantum dual
    feasible-function bound.  For each integer quantum ``d``, replace a job
    of size ``p`` by ``floor(p / d)`` units.  Any bank finishing by ``T`` can
    hold at most ``floor(T / d)`` such units, yielding the strict bound
    ``d * ceil(sum(floor(p / d)) / bank_count)``.  Scanning all possible
    quanta is inexpensive for the bounded row jobs accepted by the model.
    """

    jobs = np.asarray(row_job_cycles)
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if jobs.size == 0:
        return 0
    if np.any(jobs <= 0) or np.any(jobs != np.floor(jobs)):
        raise ValueError("row job cycles must be positive integers")
    jobs_i64 = jobs.astype(np.int64, copy=False)
    lower_bound = max(
        int(jobs_i64.max()),
        _ceil_div(int(jobs_i64.sum()), bank_count),
    )
    sizes, counts = np.unique(jobs_i64, return_counts=True)
    for quantum in range(1, int(sizes[-1]) + 1):
        quantum_work = int(np.dot(sizes // quantum, counts))
        lower_bound = max(
            lower_bound,
            quantum * _ceil_div(quantum_work, bank_count),
        )
    # Every feasible bank load is a multiple of the common job quantum, so the
    # optimum makespan is as well.  Round the lower bound without weakening
    # its safety.
    job_gcd = int(np.gcd.reduce(sizes))
    return _ceil_div(lower_bound, job_gcd) * job_gcd


def _combine_histograms(histograms: Sequence[BankHistogram]) -> BankHistogram:
    """Merge already-decoded jobs without losing their request scope."""

    nonempty = [histogram for histogram in histograms if histogram.job_count]
    if not nonempty:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_u64 = np.empty(0, dtype=np.uint64)
        return BankHistogram(empty_i64, empty_i64, empty_u64, empty_i64)

    banks = np.concatenate([histogram.banks for histogram in nonempty])
    rows = np.concatenate([histogram.rows for histogram in nonempty])
    masks = np.concatenate([histogram.sector_masks for histogram in nonempty])
    request_ids = np.concatenate(
        [histogram.request_ids for histogram in nonempty]
    )
    order = np.lexsort((rows, banks, request_ids))
    request_ids = request_ids[order]
    banks = banks[order]
    rows = rows[order]
    masks = masks[order]
    starts = np.empty(masks.size, dtype=bool)
    starts[0] = True
    starts[1:] = (
        (request_ids[1:] != request_ids[:-1])
        | (banks[1:] != banks[:-1])
        | (rows[1:] != rows[:-1])
    )
    indices = np.flatnonzero(starts)
    return BankHistogram(
        banks=banks[indices],
        rows=rows[indices],
        sector_masks=np.bitwise_or.reduceat(masks, indices),
        request_ids=request_ids[indices],
    )


def _schedule_shared_ports(
    histogram: BankHistogram,
    data_job_cycles: np.ndarray,
    spec: DramBankSpec,
) -> float:
    """Deterministic ready/LPT schedule for partially connected stacks.

    Each bank column contains ``total_layers`` physical banks sharing
    ``connected_layers`` interchangeable data ports.  A port carries only one
    job's data at a time.  A physical bank cannot issue its next row until its
    previous data transfer plus recharge have completed.  Among ready banks we
    choose the longest remaining row job, with stable physical-address ties.
    """

    if histogram.job_count == 0:
        return 0.0

    # The fully connected case has one port per physical bank.  Preserve the
    # original vectorized critical path exactly, both for speed and regression
    # compatibility.
    if spec.connected_layers == spec.total_layers:
        intrinsic_job_cycles = data_job_cycles + spec.recharge_cycles
        bank_cycles = np.bincount(
            histogram.banks,
            weights=intrinsic_job_cycles,
            minlength=spec.physical_bank_count,
        )
        return float(bank_cycles.max())

    overall_finish = 0.0
    bank_width = spec.banks_per_layer
    for column in range(bank_width):
        column_jobs = np.flatnonzero(histogram.banks % bank_width == column)
        if column_jobs.size == 0:
            continue

        jobs_by_bank = {}
        positions = {}
        bank_ready = {}
        remaining_work = {}
        for bank in np.unique(histogram.banks[column_jobs]):
            bank_i = int(bank)
            indices = column_jobs[histogram.banks[column_jobs] == bank]
            ordered = sorted(
                map(int, indices),
                key=lambda index: (
                    -float(data_job_cycles[index]),
                    int(histogram.rows[index]),
                    int(histogram.request_ids[index]),
                    index,
                ),
            )
            jobs_by_bank[bank_i] = ordered
            positions[bank_i] = 0
            bank_ready[bank_i] = 0.0
            remaining_work[bank_i] = float(data_job_cycles[ordered].sum())

        # Ports within a bank column are interchangeable.  The index is only
        # a deterministic tie-breaker.
        ports = [
            (0.0, port_index)
            for port_index in range(spec.connected_layers)
        ]
        heapq.heapify(ports)
        remaining = int(column_jobs.size)
        column_finish = 0.0

        while remaining:
            port_ready, port_index = heapq.heappop(ports)
            available_banks = [
                bank
                for bank, jobs in jobs_by_bank.items()
                if positions[bank] < len(jobs)
                and bank_ready[bank] <= port_ready
            ]
            if not available_banks:
                port_ready = min(
                    bank_ready[bank]
                    for bank, jobs in jobs_by_bank.items()
                    if positions[bank] < len(jobs)
                )
                available_banks = [
                    bank
                    for bank, jobs in jobs_by_bank.items()
                    if positions[bank] < len(jobs)
                    and bank_ready[bank] <= port_ready
                ]

            def candidate_key(
                bank: int,
            ) -> Tuple[float, float, int, int, int, int]:
                index = jobs_by_bank[bank][positions[bank]]
                return (
                    -float(data_job_cycles[index]),
                    -remaining_work[bank],
                    bank,
                    int(histogram.rows[index]),
                    int(histogram.request_ids[index]),
                    index,
                )

            chosen_bank = min(available_banks, key=candidate_key)
            job_index = jobs_by_bank[chosen_bank][positions[chosen_bank]]
            positions[chosen_bank] += 1
            remaining -= 1
            remaining_work[chosen_bank] -= float(data_job_cycles[job_index])

            data_finish = port_ready + float(data_job_cycles[job_index])
            bank_ready[chosen_bank] = data_finish + spec.recharge_cycles
            column_finish = max(column_finish, bank_ready[chosen_bank])
            heapq.heappush(ports, (data_finish, port_index))

        overall_finish = max(overall_finish, column_finish)

    return float(overall_finish)


def service_histogram(histogram: BankHistogram, spec: DramBankSpec) -> BankServiceResult:
    if histogram.job_count == 0:
        zeros = np.zeros(spec.physical_bank_count, dtype=np.float64)
        empty = np.empty(0, dtype=np.float64)
        return BankServiceResult(
            cycles=0.0,
            bank_cycles=zeros,
            row_job_cycles=empty,
            sector_count=0,
            transferred_bytes=0,
            active_banks=0,
            transaction_efficiency=1.0,
            bank_balance_efficiency=1.0,
            overall_efficiency=1.0,
            fixed_job_lower_bound_cycles=0.0,
            sector_lower_bound_cycles=0.0,
        )
    counts = _popcount_masks(histogram.sector_masks).astype(np.float64)
    data_job_cycles = spec.sector_cycles * counts
    row_cycles = spec.recharge_cycles + data_job_cycles
    bank_cycles = np.bincount(
        histogram.banks,
        weights=row_cycles,
        minlength=spec.physical_bank_count,
    ).astype(np.float64)
    sector_count = int(counts.sum())
    data_cycles = float(spec.sector_cycles * sector_count)
    scheduled_cycles = _schedule_shared_ports(
        histogram, data_job_cycles, spec
    )
    connectivity_cap_cycles = data_cycles / (
        spec.port_count * spec.connectivity_efficiency
    )
    cycles = max(float(scheduled_cycles), float(connectivity_cap_cycles))
    total_job_cycles = float(row_cycles.sum())
    transaction_efficiency = data_cycles / total_job_cycles
    # Physical-bank balance remains a row-buffer diagnostic.  In a partial
    # stack recharge can overlap port data, so it is intentionally not a
    # factorization partner for the raw TSV-port utilization below.
    bank_balance_efficiency = total_job_cycles / (
        spec.physical_bank_count * cycles
    )
    overall_efficiency = data_cycles / (
        spec.port_count * cycles
    )
    sector_lb = float(sector_parallel_lower_bound_cycles(sector_count, spec))
    # Both are valid relaxations of the same fixed-row-job scheduling problem.
    # The sector-packing bound can be stronger when a complete bank wave is
    # followed by a one-sector tail.
    # A second relaxation keeps each row-data burst indivisible but freely
    # assigns bursts to ports.  Whichever port finishes last still owes one
    # final recharge before the wave is complete.
    port_job_lb = (
        fixed_job_parallel_lower_bound_cycles(
            data_job_cycles, spec.port_count
        )
        + spec.recharge_cycles
    )
    fixed_job_lb = max(
        float(fixed_job_parallel_lower_bound_cycles(
            row_cycles, spec.physical_bank_count
        )),
        float(port_job_lb),
        sector_lb,
    )
    return BankServiceResult(
        cycles=cycles,
        bank_cycles=bank_cycles,
        row_job_cycles=row_cycles,
        sector_count=sector_count,
        transferred_bytes=sector_count * spec.sector_bytes,
        active_banks=int(np.count_nonzero(bank_cycles)),
        transaction_efficiency=transaction_efficiency,
        bank_balance_efficiency=bank_balance_efficiency,
        overall_efficiency=overall_efficiency,
        fixed_job_lower_bound_cycles=fixed_job_lb,
        sector_lower_bound_cycles=sector_lb,
    )


def service_requests(
    requests: Sequence[np.ndarray],
    spec: DramBankSpec,
    *,
    connectivity: str = "direct",
    swizzle: BankSwizzle = BankSwizzle(),
    coalesce_scope: str = "request",
) -> BankServiceResult:
    histogram = build_bank_histogram(
        requests,
        spec,
        connectivity=connectivity,
        swizzle=swizzle,
        coalesce_scope=coalesce_scope,
    )
    return service_histogram(histogram, spec)


def service_request_groups(
    request_groups: Sequence[Tuple[Sequence[np.ndarray], BankSwizzle]],
    spec: DramBankSpec,
    *,
    connectivity: str = "direct",
    coalesce_scope: str = "request",
) -> BankServiceResult:
    """Service request groups with different static swizzles in one schedule.

    This is the A/B GEMM path: each operand can use a different reversible
    layout transform, but both operands contend for the same physical banks
    and TSV ports.  Request IDs are offset between groups in both coalescing
    modes, so distinct tensors cannot merge accidentally.  Wave scope still
    coalesces all CTA requests *within* each operand before the joint schedule.

    ``request_groups`` contains ``(requests, swizzle)`` pairs.
    """

    if coalesce_scope not in {"request", "wave"}:
        raise ValueError("coalesce_scope must be request or wave")
    histograms: List[BankHistogram] = []
    request_offset = 0
    for requests, swizzle in request_groups:
        histogram = build_bank_histogram(
            requests,
            spec,
            connectivity=connectivity,
            swizzle=swizzle,
            coalesce_scope=coalesce_scope,
        )
        # Coalescing is legal among CTAs requesting the same operand, but not
        # across distinct tensors merely because two operand-specific
        # swizzles happen to produce the same (bank,row,sector) coordinate.
        if histogram.job_count:
            histogram = BankHistogram(
                banks=histogram.banks,
                rows=histogram.rows,
                sector_masks=histogram.sector_masks,
                request_ids=histogram.request_ids + request_offset,
            )
        histograms.append(histogram)
        request_offset += len(requests)
    return service_histogram(_combine_histograms(histograms), spec)


def service_sectors(
    sector_ids: np.ndarray,
    spec: DramBankSpec,
    *,
    connectivity: str = "direct",
    swizzle: BankSwizzle = BankSwizzle(),
) -> BankServiceResult:
    """Convenience wrapper treating all sectors as one coalesced request."""

    return service_requests(
        [np.asarray(sector_ids, dtype=np.int64)],
        spec,
        connectivity=connectivity,
        swizzle=swizzle,
        coalesce_scope="wave",
    )


def apply_littles_law_cycles(
    bank_cycles: float,
    transferred_bytes: int,
    outstanding_bytes: float,
    spec: DramBankSpec,
) -> Tuple[float, bool, float]:
    """Apply the operator-specific Little's-Law time floor exactly once."""

    if transferred_bytes <= 0:
        return float(bank_cycles), False, 0.0
    if outstanding_bytes <= 0:
        return float("inf"), True, float("inf")
    # A finite pending window cannot have more unique DDR bytes in flight than
    # it actually transfers.  The cap enforces at least one round-trip latency
    # per non-empty window.  Multi-stage benefit remains: a W-iteration window
    # pays that latency once and the caller normalizes it across W K steps.
    effective_outstanding = min(float(outstanding_bytes), transferred_bytes)
    ll_cycles = (
        transferred_bytes
        * spec.round_trip_latency_data_cycles
        / effective_outstanding
    )
    final_cycles = max(float(bank_cycles), float(ll_cycles))
    return final_cycles, ll_cycles > bank_cycles, float(ll_cycles)
