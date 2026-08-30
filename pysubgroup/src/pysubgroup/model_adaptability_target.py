"""
Model-adaptability target and quality functions for subgroup discovery.

Implements the method of the thesis (chapters 3–4) as a pysubgroup extension.
A search returns three quantities per subgroup:

    Q_roh(S) = AUC_{S_test}(f_{S_train}) - AUC_{S_test}(f_{D_train})
    Q_gew(S) = Q_roh(S) * cb(S)^β * (|S|/n)^α
    Q_ga(S)  = Q_roh(S) - max({0} ∪ {Q_roh(H) | H generalizes S})

Search and local fits use the training half; Q_roh is measured on the test half.
``quality`` in the result is Q_gew (rank key). ``quality_raw`` is Q_roh (effect
size). ``quality_ga`` is the redundancy score. A subgroup is interesting when it
is scorable, Q_roh > 0 and Q_ga ≥ 0.
"""
import threading
import warnings
from collections import Counter, namedtuple
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn import metrics

import pysubgroup as ps

# Protocol defaults (chapter 4). α ranks by size-corrected Q_gew; β = 0 after
# the size correction (remaining class-balance correlation is small).
DEFAULT_TEST_SIZE = 0.5
DEFAULT_SUBGROUP_SIZE_WEIGHT = 0.7
DEFAULT_SUBGROUP_CLASS_BALANCE_WEIGHT = 0.0
DEFAULT_MIN_SIZE_TRAIN = 5
DEFAULT_MIN_SIZE_TEST = 5

# ``status`` on a statistics row. ``SIZE_ONLY`` is the cheap placeholder before
# the local fit; anything other than ``OK`` means no quality was computed.
STATUS_SIZE_ONLY = "size_only"
STATUS_OK = "ok"
STATUS_TOO_SMALL_TRAIN = "too_small_train"
STATUS_TOO_SMALL_TEST = "too_small_test"
STATUS_SINGLE_CLASS_TRAIN = "single_class_train"
STATUS_RARE_CLASS_TRAIN = "rare_class_train"
STATUS_SINGLE_CLASS_TEST = "single_class_test"
STATUS_NAN_SCORE = "nan_score"

##########
# target #
##########


class ModelAdaptabilityTarget:
    """
    Target concept for local-model subgroup quality functions.

    ``label_column`` is the prediction target; ``feature_columns`` are model
    inputs (default: all columns except the label). Global predictions are
    cached on the quality function, not written back into the dataframe.
    """

    statistic_types = (
        "size_sg",
        "size_sg_train",
        "size_sg_test",
        "class_balance",
        "global_test_score",
        "local_test_score",
        "local_train_score",
        "status",
    )

    def __init__(
        self,
        label_column: str = "label",
        feature_columns: Optional[list[str]] = None,
    ):
        self.label_column = label_column
        self.feature_columns = feature_columns

    def calculate_statistics(self, subgroup, data: pd.DataFrame, statistics=None):
        """Return cached statistics as a dict (for ``SubgroupDiscoveryResult.to_table``)."""
        if statistics is None:
            return {}
        if isinstance(statistics, dict):
            return statistics
        if hasattr(statistics, "_asdict"):
            return statistics._asdict()
        return statistics


class ModelAdaptabilityDiscoveryTask(ps.SubgroupDiscoveryTask):
    """
    Subgroup discovery task for model-adaptability quality functions.

    Pass the **full** dataset. The quality function splits it (stratified 50/50
    by default) and returns the training half as search data.

    Put ``MinSupportConstraint`` on this task: it bounds the full cover
    (|S| = train + test) and is applied before the local fit. Scorability
    (minimum rows and both classes on each split) is checked by the quality
    function, independently of min-support.

    Parameters
    ----------
    generalization_awareness:
        If True (default), ``to_dataframe`` adds ``quality_ga`` (Q_ga on the
        raw scale, using every evaluated candidate). Ranking stays on the
        weighted ``quality`` (Q_gew). If False, the column is omitted.
    """

    def __init__(
        self,
        data,
        target,
        search_space,
        qf,
        result_set_size=10,
        depth=3,
        min_quality=float("-inf"),
        constraints=None,
        generalization_awareness: bool = True,
    ):
        self.generalization_awareness = generalization_awareness
        if hasattr(qf, "resolve_search_data"):
            data = qf.resolve_search_data(data, target)
        super().__init__(
            data,
            target,
            search_space,
            qf,
            result_set_size=result_set_size,
            depth=depth,
            min_quality=min_quality,
            constraints=constraints,
        )

    def execute(self, algorithm=None):
        """
        Run subgroup discovery and return a result with the full dataset row first.

        The dataset summary is always prepended and does not count towards
        ``result_set_size``. ``result_set_size <= 0`` keeps all subgroups.

        The default algorithm is :class:`ProcessModelAdaptabilityDFS` where
        ``fork`` is available, otherwise :class:`ParallelModelAdaptabilityDFS`.
        """
        if algorithm is None:
            from pysubgroup.parallel_model_adaptability_dfs import (
                ParallelModelAdaptabilityDFS,
                ProcessModelAdaptabilityDFS,
                fork_available,
            )

            algorithm = (
                ProcessModelAdaptabilityDFS()
                if fork_available()
                else ParallelModelAdaptabilityDFS()
            )
        return ModelAdaptabilityDiscoveryResult.from_discovery_result(
            algorithm.execute(self)
        )


def _subgroup_selector_key(subgroup) -> frozenset:
    """Stable key for a subgroup description (empty = dataset / no selectors)."""
    if _is_dataset_subgroup(subgroup):
        return frozenset()
    if not hasattr(subgroup, "selectors"):
        return frozenset()
    selectors = subgroup.selectors
    if not selectors:
        return frozenset()
    return frozenset(repr(sel) for sel in selectors)


def _status_of(statistics) -> Optional[str]:
    if statistics is None:
        return None
    if isinstance(statistics, dict):
        return statistics.get("status")
    return getattr(statistics, "status", None)


def _generalization_aware_qualities(results, quality_by_key=None, qf=None) -> list[float]:
    """
    Post-hoc Q_ga as in ``ps.GeneralizationAwareQF``:

        Q_ga(S) = Q_roh(S) - max({0} ∪ {Q_roh(H) | H generalizes S})

    ``quality_by_key`` must cover **all** evaluated candidates, not only the
    reported ones: a missing generalization contributes baseline 0 and inflates
    Q_ga. The quality function records every candidate in
    ``observed_raw_qualities`` so the map is independent of ``result_set_size``.

    Both terms of the difference come from that map, on the *unweighted* scale.
    On the weighted scale a larger generalization is favoured by ``(|S|/n)^α``
    alone, which would turn Q_ga into a second size filter.

    The empty description uses baseline 0, as in pysubgroup. A generalization
    with negative Q_roh does not move the max, because 0 stays in the set.
    """
    quality_by_key = dict(quality_by_key) if quality_by_key else {}
    if qf is not None:
        for _, sg, stats in results:
            key = _subgroup_selector_key(sg)
            if key in quality_by_key or not key:
                continue
            raw = _quality_raw_from_stats(qf, stats)
            if np.isfinite(raw):
                quality_by_key[key] = raw

    max_gen_memo: dict[frozenset, float] = {}

    def max_generalization_quality(key: frozenset) -> float:
        if key in max_gen_memo:
            return max_gen_memo[key]
        if not key:
            max_gen_memo[key] = 0.0
            return 0.0

        max_q = 0.0
        for sel in key:
            parent = key - {sel}
            if parent in quality_by_key:
                max_q = max(
                    max_q,
                    quality_by_key[parent],
                    max_generalization_quality(parent),
                )
            else:
                max_q = max(max_q, max_generalization_quality(parent))
        max_gen_memo[key] = max_q
        return max_q

    scores = []
    for _, sg, stats in results:
        key = _subgroup_selector_key(sg)
        if key in quality_by_key:
            own = float(quality_by_key[key])
        elif qf is not None:
            own = _quality_raw_from_stats(qf, stats)
        else:
            own = float("nan")
        scores.append(own - max_generalization_quality(key))
    return scores


def meets_success_criterion(
    quality_raw: float,
    quality_ga: Optional[float] = None,
    status: str = STATUS_OK,
) -> bool:
    """
    Whether a subgroup is interesting (chapter 3).

    Scorable, Q_roh > 0, and — when ``quality_ga`` is given — Q_ga ≥ 0.
    Q_ga ≥ 0 drops refinements that fall below a generalization; it does not
    drop refinements that only raise Q_roh slightly.
    """
    if status != STATUS_OK:
        return False
    if quality_raw is None or not np.isfinite(quality_raw) or quality_raw <= 0:
        return False
    if quality_ga is None:
        return True
    return bool(np.isfinite(quality_ga) and quality_ga >= 0)


def _quality_raw_from_stats(qf, statistics) -> float:
    """Q_roh from a finished statistics row; NaN if the candidate was not scored."""
    if _status_of(statistics) != STATUS_OK:
        return float("nan")
    try:
        raw = qf.raw_quality(statistics)
    except (AttributeError, TypeError):
        return float("nan")
    return float(raw) if np.isfinite(raw) else float("nan")


class ModelAdaptabilityDiscoveryResult(ps.SubgroupDiscoveryResult):
    """
    Discovery result with a dataset summary row first.

    ``to_dataframe`` reports Q_roh (``quality_raw``), Q_gew (``quality``) and,
    when enabled, Q_ga (``quality_ga``). ``interesting`` is the success
    criterion. Ranking of the non-dataset rows is by Q_gew.
    """

    def __init__(self, results, task, global_model_scores=None):
        super().__init__(results, task)
        self.global_model_scores = global_model_scores or {}
        self.evaluation_failures: dict[str, int] = {}

    def failure_report(self) -> str:
        """One-line summary of candidates that could not be evaluated."""
        if not self.evaluation_failures:
            return "no evaluation failures"
        total = sum(self.evaluation_failures.values())
        detail = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(
                self.evaluation_failures.items(), key=lambda kv: -kv[1]
            )
        )
        return f"{total} candidates not evaluated ({detail})"

    @classmethod
    def from_discovery_result(cls, result: ps.SubgroupDiscoveryResult):
        """
        Wrap any algorithm ``SubgroupDiscoveryResult``.

        Prepends a Dataset summary row, trims to ``result_set_size``, and
        attaches global model scores. Not DFS-specific.
        """
        task = result.task
        qf = task.qf
        data = task.data
        target = task.target

        dataset_sg = _dataset_subgroup(data)
        dataset_stats = qf.calculate_statistics(dataset_sg, target, data)
        dataset_quality = qf.evaluate(dataset_sg, target, data, dataset_stats)

        subgroup_results = [
            entry for entry in result.results if not _is_dataset_subgroup(entry[1])
        ]
        subgroup_results.sort(key=lambda entry: entry[0], reverse=True)
        # result_set_size <= 0 means keep all subgroups
        if task.result_set_size > 0:
            subgroup_results = subgroup_results[: task.result_set_size]

        combined = [(dataset_quality, dataset_sg, dataset_stats)] + subgroup_results
        wrapped = cls(
            combined,
            task,
            global_model_scores={
                "global_train_score": qf.global_train_score,
                "global_test_score": qf.global_test_score,
            },
        )
        wrapped.evaluation_failures = dict(getattr(qf, "failure_counts", {}))
        return wrapped

    def to_dataframe(
        self,
        statistics_to_show=None,
        autoround=False,
        include_target=False,
        interesting_only=False,
    ):
        """
        Result table. Columns ``quality_raw``, ``quality`` and (by default)
        ``quality_ga`` are Q_roh, Q_gew and Q_ga. ``interesting`` is the
        success criterion. Pass ``interesting_only=True`` to drop rows that
        fail it (the dataset summary row is kept).
        """
        df = super().to_dataframe(statistics_to_show, autoround, include_target)
        if len(df) == 0:
            return df

        qf = self.task.qf
        quality_raw = [
            _quality_raw_from_stats(qf, stats) for _, _, stats in self.results
        ]

        df = df.copy()
        df.insert(0, "quality_raw", quality_raw)

        use_ga = bool(getattr(self.task, "generalization_awareness", False))
        ga_qualities = None
        if use_ga:
            observed = getattr(qf, "observed_raw_qualities", None) or {}
            ga_qualities = _generalization_aware_qualities(
                self.results, observed, qf
            )
            df.insert(2, "quality_ga", ga_qualities)

        interesting = []
        for i, (_, sg, stats) in enumerate(self.results):
            if _is_dataset_subgroup(sg):
                interesting.append(False)
                continue
            ga = ga_qualities[i] if use_ga else None
            interesting.append(
                meets_success_criterion(
                    quality_raw[i], ga, _status_of(stats) or STATUS_OK
                )
            )
        df.insert(3 if use_ga else 2, "interesting", interesting)

        if len(df) <= 1:
            return df

        dataset_row = df.iloc[[0]]
        other_rows = df.iloc[1:]
        if interesting_only:
            other_rows = other_rows.loc[other_rows["interesting"]]
        other_rows = other_rows.sort_values("quality", ascending=False)
        return pd.concat([dataset_row, other_rows], ignore_index=True)


########################
# performance measures #
########################


def _label_balance_fraction(labels: pd.Series):
    """
    Class-balance factor cb(S) from SubROC (https://doi.org/10.48550/arXiv.2505.11283).

    For two classes this is min(n_+/n_-, n_-/n_+); otherwise 0. Computed with
    ``value_counts`` so a non-unique index does not break the ratio.
    """
    counts = labels.value_counts()
    if len(counts) != 2:
        return 0
    ratio = counts.iloc[0] / counts.iloc[1]
    return float(ratio) if ratio <= 1 else float(1 / ratio)


def _default_train_test_split(
    data: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
    label_column: Optional[str] = None,
):
    """Stratified split when ``label_column`` is set and stratification is feasible."""
    from sklearn.model_selection import train_test_split

    stratify = None
    if label_column is not None and label_column in data.columns:
        labels = data[label_column]
        # Need ≥2 classes and at least 2 samples per class for stratify.
        counts = labels.value_counts()
        if len(counts) >= 2 and int(counts.min()) >= 2:
            stratify = labels

    try:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )


def _feature_columns(data: pd.DataFrame, target: ModelAdaptabilityTarget) -> list[str]:
    return [col for col in data.columns if col != target.label_column]


def _subgroup_cover(subgroup, data: pd.DataFrame) -> np.ndarray:
    """Boolean cover array for a subgroup evaluated on the given dataset."""
    if subgroup == slice(None):
        return np.ones(len(data), dtype=bool)

    # Always re-evaluate selectors on `data` (do not reuse a bitset from another split).
    if hasattr(subgroup, "covers"):
        return np.asarray(subgroup.covers(data), dtype=bool)

    if hasattr(subgroup, "_selectors"):
        sg = ps.create_subgroup_with_representation(data, subgroup._selectors)
        return np.asarray(sg.representation, dtype=bool)

    return np.ones(len(data), dtype=bool)


def _is_dataset_subgroup(subgroup) -> bool:
    return str(subgroup) == "Dataset"


def _dataset_subgroup(data: pd.DataFrame):
    sel_conjunction = ps.Conjunction.from_str("Dataset")
    return ps.create_subgroup_with_representation(data, sel_conjunction.selectors)


#####################
# quality functions #
#####################


def scorability_status(
    *,
    size_train: int,
    size_test: int,
    n_classes_train: int,
    min_class_count_train: int,
    n_classes_test: int,
    min_size_train: int = DEFAULT_MIN_SIZE_TRAIN,
    min_size_test: int = DEFAULT_MIN_SIZE_TEST,
    requires_two_test_classes: bool = True,
) -> Optional[str]:
    """
    Whether a subgroup can yield a quality, and if not, why.

    Shared by the quality function (checked before the local fit) and the
    ground-truth evaluation in the synthetic experiments, so thresholds and
    status tokens cannot drift. Defaults match chapter 4: at least five
    training and five test rows, both classes on both splits. One point per
    class is enough; the thresholds make the AUC defined, not well estimated.

    All conditions are anti-monotone: a rejection also prunes every refinement.

    Returns None when the subgroup is scorable.
    """
    if size_train < max(1, min_size_train):
        return STATUS_TOO_SMALL_TRAIN
    if size_test < max(1, min_size_test):
        return STATUS_TOO_SMALL_TEST
    if n_classes_train < 2:
        return STATUS_SINGLE_CLASS_TRAIN
    if min_class_count_train < 1:
        return STATUS_RARE_CLASS_TRAIN
    if requires_two_test_classes and n_classes_test < 2:
        return STATUS_SINGLE_CLASS_TEST
    return None


_CoverInfo = namedtuple(
    "_CoverInfo",
    (
        "train_cover",
        "test_cover",
        "size_sg_train",
        "size_sg_test",
        "size_sg",
        "class_balance",
        "n_classes_train",
        "n_classes_test",
        "min_class_count_train",
    ),
)


# Defined at module level rather than inside the quality function so that result
# rows can be pickled: a namedtuple created in a class body is not reachable under
# its own qualified name, which makes it unpicklable and would prevent workers of
# a process pool from returning statistics.
ModelPerformanceQF_parameters = namedtuple(
    "ModelPerformanceQF_parameters",
    (
        "size_sg",
        "size_sg_train",
        "size_sg_test",
        "class_balance",
        "global_test_score",
        "local_test_score",
        "local_train_score",
        "status",
    ),
)


class _SizeStatsWithCovers:
    """
    Size-only statistics plus train/test covers and class counts for reuse in
    ``calculate_statistics``.

    Public result rows use the plain ``tpl`` namedtuple without covers.
    """

    __slots__ = ("_stats", "cover_info")

    def __init__(self, stats, cover_info: _CoverInfo):
        self._stats = stats
        self.cover_info = cover_info

    def __getattr__(self, name):
        return getattr(self._stats, name)


class BaseLocalModelPerformanceQF(ps.BoundedInterestingnessMeasure):
    """
    Quality function: local vs. global model performance on a subgroup.

    Pass the full dataset to ``ModelAdaptabilityDiscoveryTask``. The task calls
    :meth:`resolve_search_data`, which splits stratified 50/50 by default and
    returns the training half for the search. Q_roh is the AUC difference on
    the test half (equation in the module docstring). Global predictions are
    cached here, not written into the dataframe.

    ``MinSupportConstraint`` on the task bounds |S| = train + test and is
    applied before the fit. Scorability (rows and both classes on each split)
    is independent of that bound and is also applied before the fit.

    Parameters
    ----------
    subgroup_size_weight:
        α in Q_gew. Default 0.7 (chapter 4).
    subgroup_class_balance_weight:
        β in Q_gew. Default 0.
    test_size:
        Held-out fraction for the default split (0.5).
    min_size_train, min_size_test:
        Scorability bounds per split (default 5). Anti-monotone, checked
        before the local fit. Min-support on |S| cannot replace them.
    """

    #: Ranking metrics (ROC-AUC) are undefined on a single-class test cover.
    #: Set False for metrics that stay defined, e.g. plain accuracy.
    requires_two_test_classes = True

    #: Reasons a candidate could not be scored. Counted in ``failure_counts``.
    #: Aliases of the module-level constants used by :func:`scorability_status`.
    STATUS_TOO_SMALL_TRAIN = STATUS_TOO_SMALL_TRAIN
    STATUS_TOO_SMALL_TEST = STATUS_TOO_SMALL_TEST
    STATUS_SINGLE_CLASS_TRAIN = STATUS_SINGLE_CLASS_TRAIN
    STATUS_RARE_CLASS_TRAIN = STATUS_RARE_CLASS_TRAIN
    STATUS_SINGLE_CLASS_TEST = STATUS_SINGLE_CLASS_TEST
    STATUS_NAN_SCORE = STATUS_NAN_SCORE

    tpl = ModelPerformanceQF_parameters

    def __init__(
        self,
        performance_measure,
        performance_measure_type: Literal["score", "loss"],
        subgroup_class_balance_weight: float = DEFAULT_SUBGROUP_CLASS_BALANCE_WEIGHT,
        subgroup_size_weight: float = DEFAULT_SUBGROUP_SIZE_WEIGHT,
        model_builder_global=None,
        model_builder_local=None,
        training_global=None,
        training_local=None,
        prediction_global=None,
        prediction_local=None,
        test_size: float = DEFAULT_TEST_SIZE,
        split_fn: Optional[Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = None,
        random_state: int = 42,
        min_size_train: int = DEFAULT_MIN_SIZE_TRAIN,
        min_size_test: int = DEFAULT_MIN_SIZE_TEST,
    ):
        self.performance_measure = performance_measure
        self.performance_measure_type = performance_measure_type

        self.subgroup_class_balance_weight = subgroup_class_balance_weight
        self.subgroup_size_weight = subgroup_size_weight

        self.model_builder_global = model_builder_global
        self.model_builder_local = model_builder_local or model_builder_global
        self.training_global = training_global
        self.training_local = training_local or training_global
        self.prediction_global = prediction_global
        self.prediction_local = prediction_local or prediction_global

        self.test_size = test_size
        self.split_fn = split_fn or _default_train_test_split
        self.random_state = random_state
        self.min_size_train = int(min_size_train)
        self.min_size_test = int(min_size_test)

        # Unscorable-candidate counts, and Q_roh of every scored candidate
        # (needed so Q_ga does not depend on ``result_set_size``).
        self.failure_counts: Counter = Counter()
        self.observed_raw_qualities: dict[frozenset, float] = {}
        self._bookkeeping_lock = threading.Lock()

        self._full_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

        self.has_constant_statistics = False
        self.global_train_score = None
        self.global_test_score = None
        self.required_stat_attrs = self.tpl._fields

        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._y_pred_global_test = None

    def _perform_split(
        self, data: pd.DataFrame, target: Optional[ModelAdaptabilityTarget] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.split_fn is _default_train_test_split:
            label_column = target.label_column if target is not None else None
            return _default_train_test_split(
                data,
                test_size=self.test_size,
                random_state=self.random_state,
                label_column=label_column,
            )
        return self.split_fn(data)

    def resolve_search_data(
        self, data: pd.DataFrame, target: ModelAdaptabilityTarget
    ) -> pd.DataFrame:
        """
        Split the full dataset and return the training split for subgroup search.

        Called by ``ModelAdaptabilityDiscoveryTask`` with the full dataframe.
        The default split is stratified on ``target.label_column`` when feasible.
        Resets the cached global model and the Q_ga / failure bookkeeping.
        """
        self._full_data = data
        self.train_data, self.test_data = self._perform_split(data, target)
        self.has_constant_statistics = False
        self.failure_counts = Counter()
        self.observed_raw_qualities = {}
        return self.train_data

    def _resolve_label_column(self, target: ModelAdaptabilityTarget) -> str:
        return target.label_column

    def _resolve_dataset_size(self) -> Optional[int]:
        if self._full_data is not None:
            return len(self._full_data)
        return None

    def _resolve_feature_columns(
        self, data: pd.DataFrame, target: ModelAdaptabilityTarget
    ) -> list[str]:
        if target.feature_columns is not None:
            return target.feature_columns
        return _feature_columns(data, target)

    def _statistics_complete(self, statistics) -> bool:
        """
        Whether ``statistics`` is a finished evaluation (including a failed one).

        Completeness is keyed on ``status``, not on a finite score, so a failed
        attempt is not retried in ``evaluate``.
        """
        if statistics is None:
            return False
        if isinstance(statistics, dict):
            if not all(key in statistics for key in self.required_stat_attrs):
                return False
        elif not all(hasattr(statistics, attr) for attr in self.required_stat_attrs):
            return False
        status = _status_of(statistics)
        return status is not None and status != STATUS_SIZE_ONLY

    def ensure_statistics(self, subgroup, target, data, statistics=None):
        if not self.has_constant_statistics:
            self.calculate_constant_statistics(data, target)

        if not self._statistics_complete(statistics):
            return self.calculate_statistics(subgroup, target, data, statistics or {})
        return statistics

    def calculate_constant_statistics(
        self, data: pd.DataFrame, target: ModelAdaptabilityTarget
    ):
        """Train the global model once and cache train/test feature matrices."""
        label_col = self._resolve_label_column(target)

        if self.train_data is None or self.test_data is None:
            raise ValueError(
                "Train/test split not available. Pass the full dataset to "
                "ModelAdaptabilityDiscoveryTask so resolve_search_data can split it."
            )

        feature_cols = self._resolve_feature_columns(self.train_data, target)

        self._X_train = self.train_data.loc[:, feature_cols]
        self._y_train = self.train_data[label_col]
        self._X_test = self.test_data.loc[:, feature_cols]
        self._y_test = self.test_data[label_col]

        model_global = self.model_builder_global()
        self.training_global(model_global, self._X_train, self._y_train)

        pred_train = self.prediction_global(model_global, self._X_train)
        pred_test = self.prediction_global(model_global, self._X_test)

        try:
            self.global_train_score = self.performance_measure(
                self._y_train, pred_train
            )
            self.global_test_score = self.performance_measure(self._y_test, pred_test)
        except Exception as exc:  # noqa: BLE001 - surfaced as a warning, not hidden
            self.global_train_score = None
            self.global_test_score = None
            warnings.warn(
                "Global model could not be scored on the full split "
                f"({type(exc).__name__}: {exc}). Subgroup qualities remain "
                "computable, but the dataset-level reference scores are missing.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._y_pred_global_test = pd.Series(pred_test, index=self.test_data.index)

        self.has_constant_statistics = True

    def _empty_score_stats(
        self,
        size_sg: int,
        size_sg_train: int,
        size_sg_test: int,
        class_balance: float,
        status: str,
    ):
        return self.tpl(
            size_sg=size_sg,
            size_sg_train=size_sg_train,
            size_sg_test=size_sg_test,
            class_balance=class_balance,
            global_test_score=-np.inf,
            local_test_score=-np.inf,
            local_train_score=-np.inf,
            status=status,
        )

    def _covers_and_balance(self, subgroup, data: pd.DataFrame) -> _CoverInfo:
        """
        Train/test covers, sizes, class balance and per-split class counts.

        ``size_sg`` is the full cover (train + test). The class counts are needed
        to reject unscorable candidates before the expensive local fit.
        """
        train_cover = _subgroup_cover(subgroup, data)
        test_cover = _subgroup_cover(subgroup, self.test_data)
        size_sg_train = int(train_cover.sum())
        size_sg_test = int(test_cover.sum())
        size_sg = size_sg_train + size_sg_test

        class_balance = 0.0
        n_classes_train = n_classes_test = 0
        min_class_count_train = 0
        if size_sg > 0:
            y_train_sg = self._y_train.iloc[train_cover]
            y_test_sg = self._y_test.iloc[test_cover]
            train_counts = y_train_sg.value_counts()
            n_classes_train = int(len(train_counts))
            min_class_count_train = int(train_counts.min()) if n_classes_train else 0
            n_classes_test = int(y_test_sg.nunique())
            labels_full = pd.concat([y_train_sg, y_test_sg], ignore_index=True)
            class_balance = float(_label_balance_fraction(labels_full))

        return _CoverInfo(
            train_cover=train_cover,
            test_cover=test_cover,
            size_sg_train=size_sg_train,
            size_sg_test=size_sg_test,
            size_sg=size_sg,
            class_balance=class_balance,
            n_classes_train=n_classes_train,
            n_classes_test=n_classes_test,
            min_class_count_train=min_class_count_train,
        )

    def _reject_reason(self, info: _CoverInfo) -> Optional[str]:
        """Why ``info`` cannot yield a quality, or None if it can."""
        return scorability_status(
            size_train=info.size_sg_train,
            size_test=info.size_sg_test,
            n_classes_train=info.n_classes_train,
            min_class_count_train=info.min_class_count_train,
            n_classes_test=info.n_classes_test,
            min_size_train=self.min_size_train,
            min_size_test=self.min_size_test,
            requires_two_test_classes=self.requires_two_test_classes,
        )

    def _record_failure(self, reason: str) -> None:
        with self._bookkeeping_lock:
            self.failure_counts[reason] += 1

    def calculate_size_statistics(
        self,
        subgroup,
        target: ModelAdaptabilityTarget,
        data: pd.DataFrame,
        statistics=None,
    ):
        """
        Cover sizes and class balance only — no local model fit.

        Returns a placeholder that also carries covers so
        ``calculate_statistics`` can reuse them. Used to apply
        ``task.constraints`` (e.g. MinSupport) before expensive retraining.
        """
        if not self.has_constant_statistics:
            self.calculate_constant_statistics(data, target)

        info = self._covers_and_balance(subgroup, data)
        stats = self._empty_score_stats(
            info.size_sg,
            info.size_sg_train,
            info.size_sg_test,
            info.class_balance,
            STATUS_SIZE_ONLY,
        )
        return _SizeStatsWithCovers(stats, info)

    def calculate_statistics(
        self,
        subgroup,
        target: ModelAdaptabilityTarget,
        data: pd.DataFrame,
        statistics=None,
    ):
        """
        Train a local model on the train subgroup; score local and global on the test subgroup.

        Unscorable candidates are rejected before the fit. ``status`` carries
        the reason; ``failure_counts`` tallies them.
        """
        if not self.has_constant_statistics:
            self.calculate_constant_statistics(data, target)

        if isinstance(statistics, _SizeStatsWithCovers):
            info = statistics.cover_info
        else:
            info = self._covers_and_balance(subgroup, data)

        reason = self._reject_reason(info)
        if reason is not None:
            self._record_failure(reason)
            return self._empty_score_stats(
                info.size_sg,
                info.size_sg_train,
                info.size_sg_test,
                info.class_balance,
                reason,
            )

        X_train_sg = self._X_train.iloc[info.train_cover]
        y_train_sg = self._y_train.iloc[info.train_cover]
        X_test_sg = self._X_test.iloc[info.test_cover]
        y_test_sg = self._y_test.iloc[info.test_cover]
        y_pred_global_sg = self._y_pred_global_test.iloc[info.test_cover]

        try:
            local_model = self.model_builder_local()
            self.training_local(local_model, X_train_sg, y_train_sg)

            pred_train_local = self.prediction_local(local_model, X_train_sg)
            local_train_score = self.performance_measure(y_train_sg, pred_train_local)

            pred_test_local = self.prediction_local(local_model, X_test_sg)
            local_test_score = self.performance_measure(y_test_sg, pred_test_local)

            global_test_score = self.performance_measure(y_test_sg, y_pred_global_sg)
        except Exception as exc:  # noqa: BLE001 - reason is recorded, not swallowed
            reason = f"local_fit_failed: {type(exc).__name__}"
            self._record_failure(reason)
            return self._empty_score_stats(
                info.size_sg,
                info.size_sg_train,
                info.size_sg_test,
                info.class_balance,
                reason,
            )

        if not (np.isfinite(local_test_score) and np.isfinite(global_test_score)):
            self._record_failure(self.STATUS_NAN_SCORE)
            return self._empty_score_stats(
                info.size_sg,
                info.size_sg_train,
                info.size_sg_test,
                info.class_balance,
                self.STATUS_NAN_SCORE,
            )

        return self.tpl(
            size_sg=info.size_sg,
            size_sg_train=info.size_sg_train,
            size_sg_test=info.size_sg_test,
            class_balance=info.class_balance,
            global_test_score=global_test_score,
            local_test_score=local_test_score,
            local_train_score=local_train_score,
            status=STATUS_OK,
        )

    def _get_quality_weight(self, statistics):
        size_factor = 1.0
        if self.subgroup_size_weight:
            dataset_size = self._resolve_dataset_size()
            subgroup_size = statistics.size_sg
            if dataset_size:
                subgroup_size = subgroup_size / dataset_size
            size_factor = subgroup_size ** self.subgroup_size_weight
        return (statistics.class_balance ** self.subgroup_class_balance_weight) * size_factor

    def raw_quality(self, statistics) -> float:
        """Q_roh: unweighted local-minus-global difference, sign-adjusted for losses."""
        diff = statistics.local_test_score - statistics.global_test_score
        return float(diff if self.performance_measure_type == "score" else -diff)

    def record_observation(self, subgroup, statistics) -> None:
        """
        Remember a scored candidate so Q_ga can use the full evaluated set.

        Process workers cannot share this dict, so
        :meth:`ProcessModelAdaptabilityDFS._replay_bookkeeping` calls this in
        the parent with the statistics the worker returned.
        """
        key = _subgroup_selector_key(subgroup)
        if not key:
            # Empty description is the Q_ga baseline 0, not an observed parent.
            return
        raw = self.raw_quality(statistics)
        if np.isfinite(raw):
            with self._bookkeeping_lock:
                self.observed_raw_qualities[key] = raw

    def evaluate(
        self,
        subgroup,
        target: ModelAdaptabilityTarget,
        data: pd.DataFrame,
        statistics=None,
    ):
        if subgroup == slice(None) or _is_dataset_subgroup(subgroup):
            subgroup = _dataset_subgroup(data)

        statistics = self.ensure_statistics(subgroup, target, data, statistics)

        if _status_of(statistics) != STATUS_OK:
            return -np.inf

        quality = self.raw_quality(statistics)

        if self.subgroup_class_balance_weight or self.subgroup_size_weight:
            quality *= self._get_quality_weight(statistics)

        if not np.isfinite(quality):
            self._record_failure(self.STATUS_NAN_SCORE)
            return -np.inf

        self.record_observation(subgroup, statistics)
        return quality

    def optimistic_estimate(
        self,
        subgroup,
        target: ModelAdaptabilityTarget,
        data: pd.DataFrame,
        statistics=None,
    ):
        """No tight anti-monotone bound on Q_roh; the search scores every candidate."""
        return np.inf


class LocalSoftClassifierPerformanceQF(BaseLocalModelPerformanceQF):
    """
    Local vs. global ROC-AUC for binary soft classifiers (Q_roh of chapter 3).

    Local and global models belong to the same class and share hyperparameters.
    Defaults: α = 0.7, β = 0, 50/50 split, scorability 5/5 rows.
    """

    def __init__(
        self,
        model_builder_global,
        training_global,
        test_size: float = DEFAULT_TEST_SIZE,
        split_fn=None,
        random_state: int = 42,
        model_builder_local=None,
        training_local=None,
        prediction_global=None,
        prediction_local=None,
        performance_measure=metrics.roc_auc_score,
        subgroup_class_balance_weight=DEFAULT_SUBGROUP_CLASS_BALANCE_WEIGHT,
        subgroup_size_weight=DEFAULT_SUBGROUP_SIZE_WEIGHT,
        min_size_train: int = DEFAULT_MIN_SIZE_TRAIN,
        min_size_test: int = DEFAULT_MIN_SIZE_TEST,
    ):
        default_prediction = lambda model, X: model.predict_proba(X)[:, 1]

        super().__init__(
            performance_measure=performance_measure,
            performance_measure_type="score",
            subgroup_class_balance_weight=subgroup_class_balance_weight,
            subgroup_size_weight=subgroup_size_weight,
            model_builder_global=model_builder_global,
            training_global=training_global,
            prediction_global=prediction_global or default_prediction,
            model_builder_local=model_builder_local,
            training_local=training_local,
            prediction_local=prediction_local or default_prediction,
            test_size=test_size,
            split_fn=split_fn,
            random_state=random_state,
            min_size_train=min_size_train,
            min_size_test=min_size_test,
        )

