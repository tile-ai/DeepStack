"""Small candidate-level parallel execution helper.

``auto`` deliberately chooses only between serial execution and processes.
The current bank mapper performs many short Python/NumPy operations, so a
thread pool commonly loses to the GIL and scheduling overhead.  Threads remain
available as an explicit backend for callers that want to measure them.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
from typing import Callable, Literal, Sequence, Tuple, TypeVar


ParallelBackend = Literal["serial", "thread", "process", "auto"]

# A conservative threshold measured in mapper work units.  It keeps the common
# small 15-candidate search serial while allowing large spatial grids or exact
# residue searches to amortize process startup.
AUTO_PROCESS_WORK_UNITS = 8_192

_T = TypeVar("_T")
_R = TypeVar("_R")


def resolve_backend(
    backend: str,
    workers: int,
    task_count: int,
    work_units: int,
) -> ParallelBackend:
    """Resolve a requested backend without ever auto-selecting threads."""

    if backend not in {"serial", "thread", "process", "auto"}:
        raise ValueError("backend must be serial, thread, process, or auto")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError("workers must be a non-negative integer")
    if task_count < 0 or work_units < 0:
        raise ValueError("task_count and work_units must be non-negative")
    if task_count <= 1 or workers <= 1:
        return "serial"
    if backend == "auto":
        if work_units >= AUTO_PROCESS_WORK_UNITS:
            return "process"
        return "serial"
    return backend  # type: ignore[return-value]


def map_jobs(
    worker: Callable[[_T], _R],
    jobs: Sequence[_T],
    *,
    backend: str,
    workers: int,
    work_units: int,
) -> Tuple[_R, ...]:
    """Map top-level ``worker`` over candidate jobs in deterministic order."""

    selected = resolve_backend(backend, workers, len(jobs), work_units)
    if selected == "process" and jobs:
        try:
            pickle.dumps((worker, jobs[0]))
        except (AttributeError, pickle.PickleError, TypeError) as error:
            if backend == "auto":
                selected = "serial"
            else:
                raise TypeError(
                    "process backend requires a module-level worker and "
                    "pickleable job arguments"
                ) from error
    if selected == "serial":
        return tuple(worker(job) for job in jobs)

    executor_type = (
        ThreadPoolExecutor if selected == "thread" else ProcessPoolExecutor
    )
    with executor_type(max_workers=min(workers, len(jobs))) as executor:
        # Executor.map preserves input order, keeping tie-breaking identical to
        # serial search.  Process callers must pass a module-level worker and
        # pickleable job fields.
        return tuple(executor.map(worker, jobs))
