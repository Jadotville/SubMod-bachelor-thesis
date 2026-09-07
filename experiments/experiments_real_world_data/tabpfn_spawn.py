"""
CUDA-safe process evaluation for TabPFN.

``ProcessModelAdaptabilityDFS`` forks after the parent has used the GPU.
Children then cannot create a CUDA context, local fits fail, and the search
returns no subgroups. Spawn starts fresh interpreters, so each worker can
load TabPFN onto the same card. The parent still trains the global model
once and only sends covers plus cached matrices.

This module must stay importable without running a sweep: spawn children
re-import it.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pysubgroup as ps
from pysubgroup.algorithms import constraints_satisfied
from pysubgroup.model_adaptability_target import (
    ModelPerformanceQF_parameters,
    STATUS_OK,
)
from pysubgroup.parallel_model_adaptability_dfs import (
    ProcessModelAdaptabilityDFS,
    _enumerate_subgroups,
    _finished_result,
    _limit_worker_threads,
    _push_result,
)

_STATE: dict[str, Any] = {}


def _is_text_column(series: pd.Series) -> bool:
    """Object, string, and categorical columns that TabPFN ordinal-encodes."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(dtype):
        return True
    if pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        return True
    return False


def _prepare_tabpfn_frame(X):
    """Keep text columns as pandas StringDtype so later iloc batches stay text.

    A 1024-row slice of an object column that happens to look numeric (ICD-9
    codes such as ``"428"``) is downcast to int/float. TabPFN then calls
    ``np.isnan`` on the string categories from ``fit`` and raises TypeError.
    """
    if not isinstance(X, pd.DataFrame):
        return X
    out = X.copy()
    for column in out.columns:
        if _is_text_column(out[column]):
            # StringDtype keeps all-NA batches as text. Object columns of only
            # ``np.nan`` become Int64 under ``DataFrame.convert_dtypes``, which
            # TabPFN runs on every predict batch.
            out[column] = out[column].astype("string")
    return out


def _iloc_preserve_dtypes(X: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
    """Slice a frame without letting pandas change column dtypes."""
    batch = X.iloc[start:stop].copy()
    expected = X.dtypes.to_dict()
    mismatched = {
        column: dtype
        for column, dtype in expected.items()
        if batch[column].dtype != dtype
    }
    if mismatched:
        batch = batch.astype(mismatched)
    return batch


def _init_worker(
    x_train,
    y_train: np.ndarray,
    x_test,
    y_test: np.ndarray,
    y_pred_global: np.ndarray,
    seed: int,
    n_estimators: int,
) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _limit_worker_threads()
    _STATE["x_train"] = _prepare_tabpfn_frame(x_train)
    _STATE["y_train"] = y_train
    _STATE["x_test"] = _prepare_tabpfn_frame(x_test)
    _STATE["y_test"] = y_test
    _STATE["y_pred_global"] = y_pred_global
    _STATE["seed"] = seed
    _STATE["n_estimators"] = n_estimators


def _predict_proba_positive(model, X, batch_size: int = 1024) -> np.ndarray:
    """Score in batches so large covers do not OOM several workers on one GPU."""
    n = len(X)
    if n <= batch_size:
        return model.predict_proba(X)[:, 1]
    chunks = []
    for start in range(0, n, batch_size):
        stop = start + batch_size
        batch = (
            _iloc_preserve_dtypes(X, start, stop)
            if isinstance(X, pd.DataFrame)
            else X[start:stop]
        )
        chunks.append(model.predict_proba(batch)[:, 1])
    return np.concatenate(chunks)


def _eval_cover(payload: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    """Fit one local TabPFN and score train/test plus the cached global on test."""
    train_cover, test_cover = payload
    from tabpfn import TabPFNClassifier

    x_tr = _prepare_tabpfn_frame(_STATE["x_train"].iloc[train_cover])
    y_tr = _STATE["y_train"][train_cover]
    x_te = _prepare_tabpfn_frame(_STATE["x_test"].iloc[test_cover])
    y_te = _STATE["y_test"][test_cover]
    y_g = _STATE["y_pred_global"][test_cover]
    model = TabPFNClassifier(
        n_estimators=int(_STATE["n_estimators"]),
        device="cuda",
        ignore_pretraining_limits=True,
        memory_saving_mode="auto",
        random_state=int(_STATE["seed"]),
    )
    model.fit(x_tr, y_tr)
    pred_tr = _predict_proba_positive(model, x_tr)
    pred_te = _predict_proba_positive(model, x_te)
    return (
        float(roc_auc_score(y_tr, pred_tr)),
        float(roc_auc_score(y_te, pred_te)),
        float(roc_auc_score(y_te, y_g)),
    )


class TabPFNSpawnProcessDFS(ProcessModelAdaptabilityDFS):
    """
    Same candidate set as process DFS, but workers start with ``spawn``.

    Local TabPFN fits run in the children; the parent only assembles qualities.
    Several workers share one GPU — that contention is the measurement.
    """

    def __init__(
        self,
        max_workers=None,
        apply_representation=None,
        chunksize: int = 4,
        n_estimators: int = 1,
    ):
        super().__init__(
            max_workers=max_workers,
            apply_representation=apply_representation,
            chunksize=chunksize,
        )
        self.n_estimators = int(n_estimators)
        self.n_local_fits = 0

    def execute(self, task):
        qf = task.qf
        qf.calculate_constant_statistics(task.data, task.target)
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
        qf = task.qf
        jobs: list[tuple[int, np.ndarray, np.ndarray]] = []
        result = []
        for index, subgroup in enumerate(candidates):
            size_stats = qf.calculate_size_statistics(
                subgroup, task.target, task.data
            )
            if not constraints_satisfied(
                task.constraints_monotone, subgroup, size_stats, task.data
            ):
                continue
            info = size_stats.cover_info
            if qf._reject_reason(info) is not None:
                statistics = qf.calculate_statistics(
                    subgroup, task.target, task.data, size_stats
                )
                quality = qf.evaluate(subgroup, task.target, task.data, statistics)
                _push_result(result, task, subgroup, quality, statistics)
                continue
            jobs.append(
                (
                    index,
                    np.asarray(info.train_cover, dtype=bool),
                    np.asarray(info.test_cover, dtype=bool),
                    float(info.class_balance),
                    int(info.size_sg),
                    int(info.size_sg_train),
                    int(info.size_sg_test),
                )
            )

        self.n_local_fits = len(jobs)
        print(
            f"TabPFNSpawnProcessDFS: {len(jobs)} local fits, "
            f"{self.max_workers} spawn workers, chunksize={self.chunksize}",
            flush=True,
        )
        if not jobs:
            return result

        x_train = qf._X_train.reset_index(drop=True)
        x_test = qf._X_test.reset_index(drop=True)
        y_train = np.asarray(qf._y_train)
        y_test = np.asarray(qf._y_test)
        y_pred = np.asarray(qf._y_pred_global_test)
        n_full = float(qf._resolve_dataset_size() or (len(y_train) + len(y_test)))
        size_weight = float(qf.subgroup_size_weight or 0.0)
        balance_weight = float(qf.subgroup_class_balance_weight or 0.0)

        ctx = get_context("spawn")
        payloads = [(train, test) for _, train, test, *_ in jobs]
        scores: list[tuple[float, float, float] | None] = [None] * len(jobs)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                x_train,
                y_train,
                x_test,
                y_test,
                y_pred,
                int(qf.random_state),
                self.n_estimators,
            ),
        ) as executor:
            futures = {
                executor.submit(_eval_cover, payload): i
                for i, payload in enumerate(payloads)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                scores[i] = fut.result()
                done += 1
                if done == 1 or done % 50 == 0 or done == len(jobs):
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed if elapsed else 0.0
                    print(
                        f"TabPFNSpawnProcessDFS: {done}/{len(jobs)} local fits "
                        f"({rate:.2f}/s, {elapsed:.0f}s)",
                        flush=True,
                    )
        if any(score is None for score in scores):
            raise RuntimeError("spawn workers returned fewer scores than local fits")

        for (
            index,
            _train_cover,
            _test_cover,
            class_balance,
            size_sg,
            size_tr,
            size_te,
        ), (local_tr, local_te, global_te) in zip(jobs, scores):
            subgroup = candidates[index]
            statistics = ModelPerformanceQF_parameters(
                size_sg=size_sg,
                size_sg_train=size_tr,
                size_sg_test=size_te,
                class_balance=class_balance,
                global_test_score=global_te,
                local_test_score=local_te,
                local_train_score=local_tr,
                status=STATUS_OK,
            )
            raw = float(local_te - global_te)
            quality = raw
            if size_weight:
                quality *= (size_sg / n_full) ** size_weight
            if balance_weight:
                quality *= class_balance ** balance_weight
            self._replay_bookkeeping(qf, subgroup, quality, statistics)
            _push_result(result, task, subgroup, quality, statistics)
        return result
