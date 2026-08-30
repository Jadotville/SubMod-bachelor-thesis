"""
Equivalence tests for the process-based adaptability search.

The process pool exists purely for speed, so the bar is that it must be
indistinguishable from the sequential DFS in what it returns *and* in the
bookkeeping the quality function accumulates — the latter is not automatic,
because worker processes mutate their own copies of the quality function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pysubgroup as ps
from pysubgroup.model_adaptability_target import ModelAdaptabilityDiscoveryResult
from pysubgroup.parallel_model_adaptability_dfs import (
    ParallelModelAdaptabilityDFS,
    ProcessModelAdaptabilityDFS,
    fork_available,
)

pytestmark = pytest.mark.skipif(
    not fork_available(), reason="process pool requires the fork start method"
)


def _data(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """Two regions with opposite signal, so several subgroups are adaptable."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=n)
    sign = np.where(a > 0, 1.0, -1.0)
    logit = sign * (2.0 * b)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"a": a, "b": b, "noise": noise, "target": y})


def _build_task(df: pd.DataFrame, depth: int = 2, min_support: int = 40):
    def build():
        return LogisticRegression(max_iter=200)

    def train(model, X, y):
        model.fit(X, y)
        return model

    def predict(model, X):
        return model.predict_proba(X)[:, 1]

    def split_fn(data: pd.DataFrame):
        tr, te = train_test_split(
            data, test_size=0.5, random_state=0, stratify=data["target"]
        )
        return tr.reset_index(drop=True), te.reset_index(drop=True).copy()

    qf = ps.LocalSoftClassifierPerformanceQF(
        model_builder_global=build,
        model_builder_local=build,
        training_global=train,
        training_local=train,
        prediction_global=predict,
        prediction_local=predict,
        split_fn=split_fn,
        random_state=0,
        subgroup_size_weight=0.75,
        subgroup_class_balance_weight=0.25,
    )
    target = ps.ModelAdaptabilityTarget(
        label_column="target", feature_columns=["a", "b", "noise"]
    )
    search_space = ps.create_selectors(df, ignore=["target"], nbins=4)
    return ps.ModelAdaptabilityDiscoveryTask(
        data=df,
        target=target,
        search_space=search_space,
        qf=qf,
        depth=depth,
        result_set_size=15,
        constraints=[ps.MinSupportConstraint(min_support)],
        generalization_awareness=True,
    )


@pytest.fixture(scope="module")
def sequential():
    task = _build_task(_data())
    return ps.DFS().execute(task), task.qf


@pytest.fixture(scope="module")
def process_based():
    task = _build_task(_data())
    return ProcessModelAdaptabilityDFS(max_workers=2).execute(task), task.qf


def _frame(result) -> pd.DataFrame:
    """Wrap as the pipelines do; ``quality_ga`` only exists on the wrapped result."""
    return ModelAdaptabilityDiscoveryResult.from_discovery_result(result).to_dataframe()


def test_result_matches_sequential_dfs(sequential, process_based) -> None:
    expected, actual = _frame(sequential[0]), _frame(process_based[0])
    assert list(actual["subgroup"]) == list(expected["subgroup"])
    for column in ("quality", "quality_ga", "quality_raw", "size_sg", "local_test_score"):
        np.testing.assert_allclose(
            pd.to_numeric(actual[column], errors="coerce").to_numpy(float),
            pd.to_numeric(expected[column], errors="coerce").to_numpy(float),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"column {column} differs",
        )


def test_generalization_awareness_survives_the_process_boundary(
    sequential, process_based
) -> None:
    """
    ``quality_ga`` is computed from ``observed_raw_qualities``, which the workers
    fill in their own memory. If the parent does not replay it, Q_ga falls back
    to the trimmed result set and can miss a parent.
    """
    _, qf_process = process_based
    _, qf_sequential = sequential
    assert qf_process.observed_raw_qualities, "bookkeeping was lost across the fork"
    assert set(qf_process.observed_raw_qualities) == set(
        qf_sequential.observed_raw_qualities
    )
    for key, quality in qf_sequential.observed_raw_qualities.items():
        assert qf_process.observed_raw_qualities[key] == pytest.approx(quality)

    frame = _frame(process_based[0]).iloc[1:]
    raw = pd.to_numeric(frame["quality_raw"], errors="coerce")
    ga = pd.to_numeric(frame["quality_ga"], errors="coerce")
    assert (ga < raw - 1e-12).any(), "quality_ga never penalises a generalization"


def test_failure_counts_are_reconstructed(sequential, process_based) -> None:
    _, qf_process = process_based
    _, qf_sequential = sequential
    assert dict(qf_process.failure_counts) == dict(qf_sequential.failure_counts)


def test_matches_the_thread_based_implementation() -> None:
    task_threads = _build_task(_data())
    task_process = _build_task(_data())
    threads = ParallelModelAdaptabilityDFS(max_workers=2).execute(task_threads)
    process = ProcessModelAdaptabilityDFS(max_workers=2).execute(task_process)
    assert list(_frame(process)["subgroup"]) == list(_frame(threads)["subgroup"])


@pytest.mark.parametrize("chunksize", [1, 3, 64])
def test_chunksize_does_not_change_the_result(chunksize: int, sequential) -> None:
    task = _build_task(_data())
    result = ProcessModelAdaptabilityDFS(
        max_workers=2, chunksize=chunksize
    ).execute(task)
    assert list(_frame(result)["subgroup"]) == list(_frame(sequential[0])["subgroup"])


def test_empty_search_space_returns_the_dataset_row() -> None:
    task = _build_task(_data())
    task.search_space = []
    result = ProcessModelAdaptabilityDFS(max_workers=2).execute(task)
    assert len(_frame(result)) <= 1
