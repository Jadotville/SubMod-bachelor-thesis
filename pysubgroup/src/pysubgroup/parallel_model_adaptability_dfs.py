"""
Parallel search for model-adaptability quality functions (chapters 3–4).

Once the global model is trained, local fits are independent. This module
enumerates every conjunction up to ``task.depth`` with the same operator as
:class:`pysubgroup.DFS` and evaluates the candidates in parallel.

:class:`ProcessModelAdaptabilityDFS` is the search used in the experiments
(Linux ``fork``, copy-on-write). :class:`ParallelModelAdaptabilityDFS` evaluates
in threads; that helps when the local work releases the GIL (LightGBM, TabPFN
API waits) and is slower than sequential DFS for logistic regression, random
forests and the MLP.

There is no tight anti-monotone bound on Q_roh, so optimistic-estimate pruning
is omitted. Min-support and scorability are applied before each local fit.
Q_ga is computed after the search from ``observed_raw_qualities``.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

import pysubgroup as ps
from pysubgroup.algorithms import constraints_satisfied
from pysubgroup.model_adaptability_target import (
    STATUS_NAN_SCORE,
    STATUS_OK,
    STATUS_SIZE_ONLY,
)


def _enumerate_subgroups(operator, subgroup, max_depth, candidates):
    """Collect all subgroup descriptions up to ``max_depth`` (DFS order)."""
    candidates.append(subgroup)
    if subgroup.depth < max_depth:
        for refined in operator.refinements(subgroup):
            _enumerate_subgroups(operator, refined, max_depth, candidates)


def _evaluate_subgroup(task, subgroup):
    """
    Evaluate one subgroup; return ``(quality, subgroup, statistics)`` or None.

    Monotone constraints (min-support) run on cover sizes before the local fit.
    Other constraints run afterwards. Callers pass ``check_constraints=False``
    to ``add_if_required``.
    """
    qf = task.qf
    if hasattr(qf, "calculate_size_statistics"):
        size_stats = qf.calculate_size_statistics(
            subgroup, task.target, task.data
        )
        if not constraints_satisfied(
            task.constraints_monotone, subgroup, size_stats, task.data
        ):
            return None
        statistics = qf.calculate_statistics(
            subgroup, task.target, task.data, size_stats
        )
    else:
        statistics = qf.calculate_statistics(subgroup, task.target, task.data)
        if not constraints_satisfied(
            task.constraints_monotone, subgroup, statistics, task.data
        ):
            return None

    if task.constraints_other and not constraints_satisfied(
        task.constraints_other, subgroup, statistics, task.data
    ):
        return None

    quality = qf.evaluate(subgroup, task.target, task.data, statistics)
    return quality, subgroup, statistics


def _push_result(result, task, subgroup, quality, statistics) -> None:
    ps.add_if_required(
        result,
        subgroup,
        quality,
        task,
        statistics=statistics,
        check_constraints=False,
    )


def _finished_result(result, task):
    return ps.SubgroupDiscoveryResult(
        ps.prepare_subgroup_discovery_result(result, task), task
    )


class ParallelModelAdaptabilityDFS:
    """
    Depth-first enumeration with threaded evaluation in one process.

    Enumeration stays sequential; evaluation runs inside the bitset
    representation so covers match sequential DFS.
    """

    def __init__(self, max_workers=None, apply_representation=None):
        if max_workers is None:
            max_workers = min(32, os.cpu_count() or 1)
        self.max_workers = max_workers
        if apply_representation is None:
            apply_representation = ps.BitSetRepresentation
        self.apply_representation = apply_representation

    def execute(self, task):
        """Run parallel subgroup discovery on ``task``."""
        task.qf.calculate_constant_statistics(task.data, task.target)
        operator = ps.StaticSpecializationOperator(task.search_space)
        candidates = []

        with self.apply_representation(task.data, task.search_space) as representation:
            _enumerate_subgroups(
                operator, representation.Conjunction([]), task.depth, candidates
            )
            if not candidates:
                return _finished_result([], task)
            result = self._evaluate_candidates(task, candidates)

        return _finished_result(result, task)

    def _evaluate_candidates(self, task, candidates):
        result = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(_evaluate_subgroup, task, sg) for sg in candidates
            ]
            # The heap is only touched in this thread: ``add_if_required`` is
            # check-then-act and is not safe to run in the workers.
            for future in as_completed(futures):
                outcome = future.result()
                if outcome is None:
                    continue
                quality, subgroup, statistics = outcome
                _push_result(result, task, subgroup, quality, statistics)
        return result


# Inherited by forked children. Sending the task per work item would pickle
# the dataset for every candidate.
_FORK_STATE: dict = {}


def _limit_worker_threads() -> None:
    """
    Keep each worker single-threaded.

    BLAS reads thread-count environment variables at import time, which already
    happened before the fork. Without a runtime limiter each process spawns its
    own BLAS pool and the search is slower than sequential DFS.
    """
    try:
        import threadpoolctl
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        return
    # Kept in a module global so the limit outlives this call. Some
    # threadpoolctl versions restore the original limits when collected.
    _FORK_STATE["_threadpool_limiter"] = threadpoolctl.threadpool_limits(1)


def _evaluate_index(index: int):
    """Evaluate one inherited candidate; return only picklable scalars."""
    task = _FORK_STATE["task"]
    outcome = _evaluate_subgroup(task, _FORK_STATE["candidates"][index])
    if outcome is None:
        return index, None, None
    quality, _, statistics = outcome
    return index, quality, statistics


def fork_available() -> bool:
    """Whether processes can inherit parent memory instead of pickling it."""
    return "fork" in multiprocessing.get_all_start_methods()


class ProcessModelAdaptabilityDFS(ParallelModelAdaptabilityDFS):
    """
    Like :class:`ParallelModelAdaptabilityDFS`, but evaluates in processes.

    Workers inherit the enumerated candidates and the task (data and global
    model) copy-on-write. They receive an index and return quality plus
    statistics. ``observed_raw_qualities`` and ``failure_counts`` are rebuilt in
    the parent; see :meth:`_replay_bookkeeping`.

    Requires ``fork`` (Linux). Use the thread-based class elsewhere.
    """

    def __init__(
        self,
        max_workers=None,
        apply_representation=None,
        chunksize: int = 4,
    ):
        super().__init__(
            max_workers=max_workers, apply_representation=apply_representation
        )
        self.chunksize = max(1, int(chunksize))

    def execute(self, task):
        if not fork_available():
            raise RuntimeError(
                "ProcessModelAdaptabilityDFS requires the 'fork' start method; "
                "use ParallelModelAdaptabilityDFS on this platform"
            )
        return super().execute(task)

    @staticmethod
    def _replay_bookkeeping(qf, subgroup, quality, statistics) -> None:
        """Reproduce ``evaluate`` side effects lost across the process boundary."""
        status = getattr(statistics, "status", STATUS_OK)
        if status not in (STATUS_OK, STATUS_SIZE_ONLY):
            qf.failure_counts[status] += 1
        elif quality is None or not np.isfinite(quality):
            qf.failure_counts[STATUS_NAN_SCORE] += 1

        if quality is not None and np.isfinite(quality):
            qf.record_observation(subgroup, statistics)

    def _evaluate_candidates(self, task, candidates):
        result = []
        _FORK_STATE["task"] = task
        _FORK_STATE["candidates"] = candidates
        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_limit_worker_threads,
            ) as executor:
                outcomes = executor.map(
                    _evaluate_index,
                    range(len(candidates)),
                    chunksize=self.chunksize,
                )
                # Index order matches sequential DFS for the result heap.
                for index, quality, statistics in outcomes:
                    if quality is None and statistics is None:
                        continue
                    subgroup = candidates[index]
                    self._replay_bookkeeping(task.qf, subgroup, quality, statistics)
                    _push_result(result, task, subgroup, quality, statistics)
        finally:
            _FORK_STATE.clear()
        return result
