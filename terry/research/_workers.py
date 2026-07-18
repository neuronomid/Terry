"""Shared local-worker helpers for Jesse-compatible research functions."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os


def resolve_workers(cpu_cores: int | None, tasks: int) -> int:
    """Validate and clamp the requested workers, defaulting to 80% of CPUs."""
    available = os.cpu_count() or 1
    if cpu_cores is None:
        requested = max(1, int(available * 0.8))
    else:
        if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or cpu_cores < 1:
            raise ValueError("cpu_cores must be an integer greater than 0")
        requested = cpu_cores
    return max(1, min(requested, available, max(1, tasks)))


def resolve_strategy_classes(routes, strategies_dir, strategy_classes,
                             strategy_sources):
    """Load strategy classes once before concurrent backtests start."""
    from ..loader import load_strategy_class, load_strategy_from_source

    resolved = dict(strategy_classes or {})
    for route in routes:
        name = route["strategy"]
        if name in resolved:
            continue
        if strategy_sources and name in strategy_sources:
            resolved[name] = load_strategy_from_source(name, strategy_sources[name])
        else:
            resolved[name] = load_strategy_class(name, strategies_dir)
    return resolved


def parallel_results(function, indices, workers, should_cancel=None):
    """Yield bounded worker results as they finish, submitting at most N at once."""
    iterator = iter(indices)
    pending = {}
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="terry-research") as executor:
        for _ in range(workers):
            try:
                index = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(function, index)] = index
        while pending:
            if should_cancel and should_cancel():
                raise InterruptedError("Research run canceled")
            ready, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=0.25)
            if not ready:
                continue
            for future in ready:
                index = pending.pop(future)
                yield index, future.result()
                try:
                    next_index = next(iterator)
                except StopIteration:
                    continue
                pending[executor.submit(function, next_index)] = next_index
