"""
Qualitätsfunktion, Ground-Truth-Bewertung, Sweeps und Suche.

``run.py`` importiert von hier ``evaluate_gt_adaptability``, die Sweep-
Funktionen und ``run_sd``. A–E bewerten eine bekannte Beschreibung;
F1 und G1 enumerieren den Selektorraum. Beide Wege erzeugen dieselbe
``LocalSoftClassifierPerformanceQF``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Vor dem ersten NumPy-Import: sonst überlagern sich BLAS-Threads und
# Such-Worker.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup.parallel_model_adaptability_dfs import (
    ParallelModelAdaptabilityDFS,
    ProcessModelAdaptabilityDFS,
    fork_available,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datasets import DatasetSpec

MODEL_ALIASES = {
    "lr": "logistic_regression",
    "rf": "random_forest",
    "lgbm": "lightgbm",
    "mlp": "mlp",
}

# Protokoll der Arbeit: α = 0.7, β = 0 (ROC-AUC ist bereits prävalenzunabhängig).
DEFAULT_SIZE_WEIGHT = 0.7
DEFAULT_BALANCE_WEIGHT = 0.0
DEFAULT_MIN_SUPPORT = 20
MLP_MAX_ITER = 3000


def model_display_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def default_min_support(n: int, fraction: float = 0.02, floor: int = DEFAULT_MIN_SUPPORT) -> int:
    """Mindestsupport: max(floor, Anteil an n). Die Läufe nutzen DEFAULT_MIN_SUPPORT=20."""
    return max(floor, int(n * fraction))


def best_ground_truth_quality(info: dict[str, Any]) -> float | None:
    qualities = [
        q for q in info.get("ground_truth_quality", {}).values() if q is not None
    ]
    return max(qualities) if qualities else None


def best_ground_truth_rank(info: dict[str, Any]) -> int | None:
    ranks = [r for r in info.get("ground_truth_rank", {}).values() if r is not None]
    return min(ranks) if ranks else None


NOMINAL_FEATURE_NAMES = frozenset({"a", "region", "group"})


def nominal_columns_for_model(
    feature_columns: list[str],
    search_columns: list[str],
    df: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Kategoriale Regionsindikatoren, die für alle Modelle one-hot-encodiert werden.

    ``df`` guards against exploding a *continuous* search column into one dummy
    per distinct value: a search column only counts as nominal if its dtype is
    integer, boolean or object.
    """
    candidates = (set(search_columns) & set(feature_columns)) | (
        NOMINAL_FEATURE_NAMES & set(feature_columns)
    )
    if df is not None:
        candidates = {
            c
            for c in candidates
            if c not in df.columns or not pd.api.types.is_float_dtype(df[c])
        }
    return [c for c in feature_columns if c in candidates]


def encode_nominal_indicator_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    search_columns: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """
    Always one-hot-encode subgroup indicators (a / region / group) for model input.

    Applies for every model (LR, RF, MLP, …), including binary indicators, so
    regions are never treated as an ordinal numeric scale. Search stays on the
    original columns (e.g. ``region==2``); only model features are encoded
    (``region_0``, ``region_1``, …).
    """
    nominal = nominal_columns_for_model(feature_columns, search_columns, df)
    if not nominal:
        return df, list(feature_columns), {}

    out = df.copy()
    new_features: list[str] = []
    encoding_map: dict[str, list[str]] = {}

    for col in feature_columns:
        if col not in nominal or col not in out.columns:
            new_features.append(col)
            continue

        ohe = OneHotEncoder(sparse_output=False, dtype=float)
        encoded = ohe.fit_transform(out[[col]])
        names = [f"{col}_{int(cat)}" for cat in ohe.categories_[0]]
        for i, name in enumerate(names):
            out[name] = encoded[:, i]
        encoding_map[col] = names
        new_features.extend(names)

    return out, new_features, encoding_map


# Backwards-compatible alias (encoding is no longer MLP-specific).
encode_nominal_features_for_mlp = encode_nominal_indicator_features


def prepare_data_for_model(
    df: pd.DataFrame,
    spec: DatasetSpec,
    model: str,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Prepare features for ``model``; always OHE region indicators for all models."""
    del model  # encoding rule is model-independent
    return encode_nominal_indicator_features(
        df, spec.feature_columns, spec.search_columns
    )


def build_search_space(df: pd.DataFrame, search_columns: list[str]):
    """Selectors for ``df``. Callers must pass the training half.

    Numeric attributes become five equal-frequency interval selectors
    (pysubgroup defaults ``nbins=5``, ``intervals_only=True``). A single
    integer or boolean search column becomes one equality selector per
    observed value.
    """
    if not search_columns:
        return ps.create_selectors(df)
    if len(search_columns) == 1 and search_columns[0] in df.columns:
        col = search_columns[0]
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            return ps.create_nominal_selectors_for_attribute(df, col)
    return ps.create_selectors(df[search_columns])


def model_builder(
    name: str, kwargs: Optional[dict] = None
) -> Callable:
    kw = dict(kwargs or {})
    if name == "lr":
        defaults = {"max_iter": 1000}
        defaults.update(kw)
        return lambda: LogisticRegression(**defaults)
    if name == "rf":
        defaults = {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20}
        defaults.update(kw)
        return lambda: RandomForestClassifier(**defaults)
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 1,
            "verbosity": -1,
        }
        defaults.update(kw)
        return lambda: LGBMClassifier(**defaults)
    if name == "mlp":
        defaults = {
            "hidden_layer_sizes": (16, 8),
            "max_iter": MLP_MAX_ITER,
            "early_stopping": False,
            "alpha": 0.01,
        }
        defaults.update(kw)
        return lambda: make_pipeline(StandardScaler(), MLPClassifier(**defaults))
    raise ValueError(f"Unbekanntes Modell: {name!r} (lr, rf, lgbm, mlp)")


def mlp_convergence_info(model) -> dict[str, Any]:
    """Iteration count / convergence flag for a (possibly pipelined) MLP."""
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("mlpclassifier", model)
    n_iter = getattr(estimator, "n_iter_", None)
    max_iter = getattr(estimator, "max_iter", None)
    if n_iter is None or max_iter is None:
        return {}
    return {
        "n_iter": int(n_iter),
        "max_iter": int(max_iter),
        "converged": bool(int(n_iter) < int(max_iter)),
    }


def _subgroup_metrics(
    result_df: pd.DataFrame, model: Optional[str] = None
) -> dict[str, Any]:
    """Kennzahlen über gefundene Subgruppen. Interessant: Q_roh > 0 und Q_ga ≥ 0."""
    empty = {
        "max_quality": None,
        "max_quality_ga": None,
        "max_quality_raw": None,
        "n_subgroups": 0,
        "n_interesting": 0,
        "interesting_rate": 0.0,
    }
    if result_df is None or len(result_df) <= 1:
        return empty

    sub = result_df.iloc[1:].copy()
    qualities = pd.to_numeric(sub["quality"], errors="coerce")
    max_q = float(qualities.max()) if qualities.notna().any() else None

    max_ga = None
    gas = None
    if "quality_ga" in sub.columns:
        gas = pd.to_numeric(sub["quality_ga"], errors="coerce")
        max_ga = float(gas.max()) if gas.notna().any() else None

    max_raw, n_int = None, 0
    have_raw = {"local_test_score", "global_test_score"} <= set(sub.columns)
    if have_raw:
        raw = pd.to_numeric(sub["local_test_score"], errors="coerce") - pd.to_numeric(
            sub["global_test_score"], errors="coerce"
        )
        if raw.notna().any():
            max_raw = float(raw.max())
        ok = raw > 0
        if gas is not None:
            ok = ok & (gas >= 0)
        if "status" in sub.columns:
            ok = ok & (sub["status"].astype(str) == "ok")
        n_int = int(ok.fillna(False).sum())

    n_sg = int(len(sub))
    return {
        "max_quality": max_q,
        "max_quality_ga": max_ga,
        "max_quality_raw": max_raw,
        "n_subgroups": n_sg,
        "n_interesting": n_int,
        "interesting_rate": (n_int / n_sg) if n_sg else 0.0,
    }


def _threshold_accuracy(y_true, y_score) -> float:
    """Accuracy from soft scores (same prediction path as the AUC QF)."""
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0:
        return float("nan")
    pred = (np.asarray(y_score, dtype=float) >= 0.5).astype(int)
    return float((pred == y_true).mean())


def adaptability_split_fn(label_column: str, seed: int):
    """Stratified 50/50 split, identical for GT evaluation and search."""

    def split_fn(data: pd.DataFrame):
        y = data[label_column]
        stratify = y if y.nunique() >= 2 and y.value_counts().min() >= 2 else None
        try:
            train, test = train_test_split(
                data,
                test_size=0.5,
                random_state=seed,
                shuffle=True,
                stratify=stratify,
            )
        except ValueError:
            train, test = train_test_split(
                data, test_size=0.5, random_state=seed, shuffle=True
            )
        return train, test.copy()

    return split_fn


def make_adaptability_qf(
    model: str,
    *,
    seed: int,
    label_column: str,
    model_kwargs: Optional[dict] = None,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
    balance_weight: float = DEFAULT_BALANCE_WEIGHT,
    metric: str = "auc",
    fitted_models: Optional[dict] = None,
) -> ps.LocalSoftClassifierPerformanceQF:
    """
    The thesis quality function, shared by search (``run_sd``) and GT scoring.

    ``metric='accuracy'`` is only used for the B1 counter-check; the reported
    series use ROC-AUC.
    """
    kw = dict(model_kwargs or {})
    if model in {"mlp", "lgbm", "rf", "lr"}:
        kw.setdefault("random_state", seed)
    builder = model_builder(model, kw)
    store = fitted_models if fitted_models is not None else {}

    def train_global(est, X, y):
        est.fit(X, y)
        store["global"] = est

    def train_local(est, X, y):
        est.fit(X, y)
        store["local"] = est

    extra: dict[str, Any] = {}
    if metric == "accuracy":
        extra["performance_measure"] = _threshold_accuracy
    elif metric != "auc":
        raise ValueError(f"Unknown metric: {metric!r}")

    qf = ps.LocalSoftClassifierPerformanceQF(
        model_builder_global=builder,
        training_global=train_global,
        training_local=train_local,
        split_fn=adaptability_split_fn(label_column, seed),
        random_state=seed,
        subgroup_size_weight=size_weight,
        subgroup_class_balance_weight=balance_weight,
        **extra,
    )
    if metric == "accuracy":
        qf.requires_two_test_classes = False
    return qf


def _search_algorithm(max_workers: int):
    """Prozess-Pool wenn fork verfügbar, sonst Thread-Pool (GIL in den Fits)."""
    if fork_available():
        return ProcessModelAdaptabilityDFS(max_workers=max_workers)
    return ParallelModelAdaptabilityDFS(max_workers=max_workers)


def run_sd(
    spec: DatasetSpec,
    model: str,
    *,
    seed: int = 42,
    min_support: int = DEFAULT_MIN_SUPPORT,
    depth: int = 1,
    result_set_size: int = 10,
    model_kwargs: Optional[dict] = None,
    max_workers: int = 4,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
    balance_weight: float = DEFAULT_BALANCE_WEIGHT,
    generalization_awareness: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Subgruppensuche für ein DatasetSpec.

    Returns
    -------
    result_df, info
        info enthält global scores, GT-Ranks, Laufzeit, Overfitting-Hinweise.
    """
    df = spec.df.copy()
    model_df, model_features, feature_encoding = prepare_data_for_model(
        df, spec, model
    )

    qf = make_adaptability_qf(
        model,
        seed=seed,
        label_column=spec.label_column,
        model_kwargs=model_kwargs,
        size_weight=size_weight,
        balance_weight=balance_weight,
    )
    # Same split the QF will draw: selectors must not see D_test.
    train_for_search, _ = adaptability_split_fn(spec.label_column, seed)(model_df)
    target = ps.ModelAdaptabilityTarget(
        label_column=spec.label_column,
        feature_columns=model_features,
    )
    task = ps.ModelAdaptabilityDiscoveryTask(
        data=model_df,
        target=target,
        search_space=build_search_space(train_for_search, spec.search_columns),
        qf=qf,
        depth=depth,
        result_set_size=result_set_size,
        constraints=[ps.MinSupportConstraint(min_support)],
        generalization_awareness=generalization_awareness,
    )

    t0 = time.perf_counter()
    result = task.execute(algorithm=_search_algorithm(max_workers))
    runtime = time.perf_counter() - t0

    result_df = result.to_dataframe()
    sub_rows = result_df.iloc[1:].copy()
    rank_col = "quality" if "quality" in sub_rows.columns else "quality_ga"
    sub_rows = sub_rows.sort_values(rank_col, ascending=False)
    ranked = [str(s) for s in sub_rows["subgroup"]]

    gt_rank = {}
    gt_quality = {}
    gt_quality_ga = {}
    for gt in spec.ground_truth_subgroups:
        if gt in ranked:
            gt_rank[gt] = ranked.index(gt) + 1
            row = sub_rows.loc[sub_rows["subgroup"].astype(str) == gt].iloc[0]
            gt_quality[gt] = float(row["quality"])
            if "quality_ga" in sub_rows.columns:
                gt_quality_ga[gt] = float(row["quality_ga"])
            else:
                gt_quality_ga[gt] = None
        else:
            gt_rank[gt] = None
            gt_quality[gt] = None
            gt_quality_ga[gt] = None

    overfit = []
    for _, row in sub_rows.head(5).iterrows():
        gap = float(row["local_train_score"]) - float(row["local_test_score"])
        if gap > 0.1:
            overfit.append(f"{row['subgroup']}: local train-test gap {gap:.3f}")

    g_train = float(result.global_model_scores.get("global_train_score", 0) or 0)
    g_test = float(result.global_model_scores.get("global_test_score", 0) or 0)
    if g_train - g_test > 0.1:
        overfit.append(f"global: train-test gap {g_train - g_test:.3f}")

    sg_metrics = _subgroup_metrics(result_df, model=model)

    info = {
        "experiment_id": spec.experiment_id,
        "experiment_name": spec.name,
        "model": model,
        "model_name": model_display_name(model),
        "seed": seed,
        "runtime_sec": runtime,
        "global_train_score": g_train,
        "global_test_score": g_test,
        "ground_truth_rank": gt_rank,
        "ground_truth_quality": gt_quality,
        "ground_truth_quality_ga": gt_quality_ga,
        "ground_truth_hits": gt_rank,
        "best_ground_truth_quality": best_ground_truth_quality(
            {"ground_truth_quality": gt_quality}
        ),
        "best_ground_truth_rank": best_ground_truth_rank(
            {"ground_truth_rank": gt_rank}
        ),
        "overfitting_flags": overfit,
        "feature_columns": spec.feature_columns,
        "model_feature_columns": model_features,
        "feature_encoding": feature_encoding,
        "search_columns": spec.search_columns,
        "description": spec.description,
        "ground_truth_subgroups": spec.ground_truth_subgroups,
        "size_weight": size_weight,
        "balance_weight": balance_weight,
        "generalization_awareness": generalization_awareness,
        "min_support": min_support,
        **sg_metrics,
    }
    return result_df, info


def serialize_run(
    result_df: pd.DataFrame,
    info: dict[str, Any],
    *,
    scenario: str,
    dgp_params: dict[str, Any] | None = None,
    run_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sub_rows = result_df.iloc[1:].copy()
    top_subgroups = []
    for _, row in sub_rows.head(10).iterrows():
        item = {
            "quality": float(row["quality"]),
            "subgroup": str(row["subgroup"]),
            "size_sg": int(row["size_sg"]),
            "size_sg_train": int(row["size_sg_train"]),
            "size_sg_test": int(row["size_sg_test"]),
            "global_test_score": float(row["global_test_score"]),
            "local_test_score": float(row["local_test_score"]),
            "local_train_score": float(row["local_train_score"]),
        }
        if "quality_ga" in row.index and pd.notna(row["quality_ga"]):
            item["quality_ga"] = float(row["quality_ga"])
        top_subgroups.append(item)

    payload = {
        "scenario": scenario,
        "experiment_id": info["experiment_id"],
        "experiment_name": info["experiment_name"],
        "model": info["model"],
        "model_name": info["model_name"],
        "seed": info["seed"],
        "runtime_sec": info["runtime_sec"],
        "global_test_score": info["global_test_score"],
        "global_train_score": info["global_train_score"],
        "top_subgroups": top_subgroups,
        "ground_truth_hits": info["ground_truth_rank"],
        "ground_truth_qualities": info["ground_truth_quality"],
        "ground_truth_qualities_ga": info.get("ground_truth_quality_ga", {}),
        "best_ground_truth_quality": info["best_ground_truth_quality"],
        "best_ground_truth_rank": info["best_ground_truth_rank"],
        "max_quality": info.get("max_quality"),
        "max_quality_ga": info.get("max_quality_ga"),
        "n_subgroups": info.get("n_subgroups"),
        "n_interesting": info.get("n_interesting"),
        "interesting_rate": info.get("interesting_rate"),
        "overfitting_flags": info["overfitting_flags"],
        "description": info["description"],
        "ground_truth_subgroups": info["ground_truth_subgroups"],
        "feature_columns": info["feature_columns"],
        "model_feature_columns": info.get("model_feature_columns", info["feature_columns"]),
        "feature_encoding": info.get("feature_encoding", {}),
        "dgp_params": dgp_params or {},
        "run_params": run_params or {},
    }
    return payload


def save_run_result(
    out_dir: Path,
    result_df: pd.DataFrame,
    info: dict[str, Any],
    *,
    scenario: str,
    dgp_params: dict[str, Any] | None = None,
    run_params: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = info["seed"]
    csv_path = out_dir / f"seed_{seed}.csv"
    json_path = out_dir / f"seed_{seed}.json"
    result_df.to_csv(csv_path, index=False, encoding="utf-8")
    payload = serialize_run(
        result_df,
        info,
        scenario=scenario,
        dgp_params=dgp_params,
        run_params=run_params,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return csv_path, json_path


def print_summary(spec: DatasetSpec, info: dict[str, Any]) -> None:
    print(f"=== {spec.experiment_id} {spec.name} ===")
    print(spec.description)
    print(f"Modell: {info['model']}  |  Seed: {info['seed']}  |  {info['runtime_sec']:.2f}s")
    print(f"Features (DGP): {info['feature_columns']}")
    if info.get("feature_encoding"):
        print(f"Indikator-Encoding: {info['feature_encoding']}")
        print(f"Modell-Features: {info.get('model_feature_columns')}")
    print(f"Suchraum: {info['search_columns']}")
    print(
        f"Global AUC train/test: {info['global_train_score']:.4f} / "
        f"{info['global_test_score']:.4f}"
    )
    if spec.ground_truth_subgroups:
        print("Ground truth:")
        for gt in spec.ground_truth_subgroups:
            q = info["ground_truth_quality"].get(gt)
            r = info["ground_truth_rank"].get(gt)
            print(f"  {gt}: quality={q}, rank={r}")
    else:
        print(
            f"Null/control: max_quality={info.get('max_quality')}, "
            f"n_interesting={info.get('n_interesting')}/"
            f"{info.get('n_subgroups')}"
        )
    if info["overfitting_flags"]:
        print("Overfitting:", info["overfitting_flags"])


# ---------------------------------------------------------------------------
# GT evaluation, sweeps, plots
# ---------------------------------------------------------------------------

N_DEFAULT = 4_000
SEED_DEFAULT = 42
# Null statements ("model X shows no adaptability") need more than three seeds.
SEEDS_DEFAULT = tuple(range(42, 52))
MIN_SUBGROUP_FOR_FIT = 5


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# core evaluation
# ---------------------------------------------------------------------------


def evaluate_gt_adaptability(
    spec: DatasetSpec,
    model: str,
    *,
    seed: int = SEED_DEFAULT,
    model_kwargs: Optional[dict] = None,
    gt_subgroups: Optional[list[str]] = None,
    metric: str = "auc",
    size_weight: float = DEFAULT_SIZE_WEIGHT,
    balance_weight: float = DEFAULT_BALANCE_WEIGHT,
) -> dict[str, Any]:
    """
    Local-vs-global gap on the ground-truth subgroup(s) via the library QF.

    Does not run subgroup search. ``seed`` controls the train/test split and
    the model's ``random_state``; the DGP seed is whatever built ``spec``.
    """
    model_df, model_features, _ = prepare_data_for_model(spec.df, spec, model)
    gts = gt_subgroups if gt_subgroups is not None else list(spec.ground_truth_subgroups)
    fitted: dict[str, Any] = {}
    qf = make_adaptability_qf(
        model,
        seed=seed,
        label_column=spec.label_column,
        model_kwargs=model_kwargs,
        size_weight=size_weight,
        balance_weight=balance_weight,
        metric=metric,
        fitted_models=fitted,
    )
    target = ps.ModelAdaptabilityTarget(
        label_column=spec.label_column, feature_columns=model_features
    )
    base = {
        "model": model,
        "seed": seed,
        "metric": metric,
        "dataset_size": len(model_df),
        "feature_columns": model_features,
        "size_weight": size_weight,
        "balance_weight": balance_weight,
    }

    try:
        train = qf.resolve_search_data(model_df, target)
        qf.calculate_constant_statistics(train, target)
    except Exception as exc:
        return {
            **base,
            "global_test_full": float("nan"),
            "global_converged": None,
            "per_gt": {},
            "n_failures": len(gts),
            "status": f"global_fit_failed: {type(exc).__name__}",
        }

    global_full = qf.global_test_score
    if global_full is None:
        global_full = float("nan")
    global_conv = (
        mlp_convergence_info(fitted["global"])
        if model == "mlp" and "global" in fitted
        else {}
    )

    per_gt: dict[str, dict[str, Any]] = {}
    n_failures = 0
    for gt in gts:
        entry: dict[str, Any] = {
            "quality": float("nan"),
            "quality_weighted": float("nan"),
            "local_test": float("nan"),
            "global_test": float("nan"),
            "n_train": 0,
            "n_test": 0,
            "class_balance": float("nan"),
            "local_converged": None,
            "status": "ok",
        }
        try:
            subgroup = ps.Conjunction.from_str(gt)
        except Exception as exc:
            entry["status"] = f"selector_error: {type(exc).__name__}"
            per_gt[gt] = entry
            n_failures += 1
            continue

        stats = qf.calculate_statistics(subgroup, target, train)
        entry["n_train"] = int(stats.size_sg_train)
        entry["n_test"] = int(stats.size_sg_test)
        entry["class_balance"] = float(stats.class_balance)
        entry["status"] = stats.status
        if stats.status != "ok":
            n_failures += 1
            per_gt[gt] = entry
            continue

        weighted = qf.evaluate(subgroup, target, train, stats)
        entry["local_test"] = float(stats.local_test_score)
        entry["global_test"] = float(stats.global_test_score)
        entry["quality"] = qf.raw_quality(stats)
        entry["quality_weighted"] = float(weighted)
        if model == "mlp" and "local" in fitted:
            entry["local_converged"] = mlp_convergence_info(fitted["local"]).get(
                "converged"
            )
        per_gt[gt] = entry

    qualities = [e["quality"] for e in per_gt.values() if np.isfinite(e["quality"])]
    return {
        **base,
        "global_test_full": float(global_full),
        "global_converged": global_conv.get("converged"),
        "global_n_iter": global_conv.get("n_iter"),
        "per_gt": per_gt,
        "n_failures": n_failures,
        "status": "ok",
        "best_gt_quality": float(np.max(qualities)) if qualities else float("nan"),
        "mean_gt_quality": float(np.mean(qualities)) if qualities else float("nan"),
    }


def gt_rows(info: dict[str, Any], **extra: Any) -> list[dict[str, Any]]:
    """Flatten one ``evaluate_gt_adaptability`` result into tidy per-GT rows."""
    rows = []
    for gt, entry in info["per_gt"].items():
        rows.append(
            {
                **extra,
                "model": info["model"],
                "seed": info["seed"],
                "metric": info["metric"],
                "gt": gt,
                "quality": entry["quality"],
                "quality_weighted": entry["quality_weighted"],
                "local_test": entry["local_test"],
                "global_test": entry["global_test"],
                "n_train": entry["n_train"],
                "n_test": entry["n_test"],
                "class_balance": entry["class_balance"],
                "status": entry["status"],
                "local_converged": entry["local_converged"],
                "global_test_full": info["global_test_full"],
                "global_converged": info.get("global_converged"),
                "dataset_size": info["dataset_size"],
            }
        )
    if not rows:
        rows.append(
            {
                **extra,
                "model": info["model"],
                "seed": info["seed"],
                "metric": info["metric"],
                "gt": None,
                "quality": float("nan"),
                "quality_weighted": float("nan"),
                "local_test": float("nan"),
                "global_test": float("nan"),
                "n_train": 0,
                "n_test": 0,
                "class_balance": float("nan"),
                "status": info.get("status", "no_gt"),
                "local_converged": None,
                "global_test_full": info["global_test_full"],
                "global_converged": info.get("global_converged"),
                "dataset_size": info["dataset_size"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# matched null reference
# ---------------------------------------------------------------------------


def matched_null_rows(
    spec: DatasetSpec,
    model: str,
    *,
    seed: int = SEED_DEFAULT,
    n_draws: int = 20,
    size_fractions: Iterable[float] = (0.1, 0.25, 0.5),
    metric: str = "auc",
    model_kwargs: Optional[dict] = None,
    **extra: Any,
) -> list[dict[str, Any]]:
    """
    Q on *random* subgroups of a given size — the reference band for "Q>0".

    Random subgroups carry no mechanism, so their Q distribution is what pure
    estimation noise plus local-overfitting produces. A measured Q only counts
    as a real effect if it leaves this band.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    df = spec.df

    for frac in size_fractions:
        _log(f"      null size_fraction={frac} ({n_draws} draws, {model})")
        for draw in range(n_draws):
            n_sub = max(MIN_SUBGROUP_FOR_FIT * 4, int(len(df) * frac))
            idx = rng.choice(len(df), size=min(n_sub, len(df)), replace=False)
            flag = np.zeros(len(df), dtype=int)
            flag[idx] = 1
            spec_rand = DatasetSpec(
                experiment_id=spec.experiment_id,
                name=f"{spec.name}__null",
                df=df.assign(_rand_group=flag),
                label_column=spec.label_column,
                feature_columns=list(spec.feature_columns),
                search_columns=list(spec.search_columns),
                ground_truth_subgroups=["_rand_group==1"],
                description="matched null (random subgroup)",
            )
            info = evaluate_gt_adaptability(
                spec_rand,
                model,
                seed=seed + draw,
                metric=metric,
                model_kwargs=model_kwargs,
            )
            rows.extend(
                gt_rows(info, size_fraction=frac, draw=draw, kind="null", **extra)
            )
    return rows


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def run_param_sweep(
    builder: Callable[..., DatasetSpec],
    *,
    sweep_param: str,
    sweep_values: Iterable[Any],
    models: list[str],
    base_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict[str, dict]] = None,
    seeds: Iterable[int] = SEEDS_DEFAULT,
    data_seed: Optional[int] = None,
    drop_indicator: bool = False,
    postprocess: Optional[Callable[[DatasetSpec], DatasetSpec]] = None,
    metric: str = "auc",
) -> pd.DataFrame:
    """
    Sweep one DGP parameter x models x seeds; return tidy per-GT rows.

    ``data_seed``: festgehaltener DGP-Seed, nur Split/Modell-Seed variiert (D3).
    """
    from datasets import without_indicator_in_features

    rows: list[dict[str, Any]] = []
    base = dict(base_kwargs or {})
    mk = model_kwargs or {}

    seed_list = list(seeds)
    for val in sweep_values:
        _log(f"      {sweep_param}={val} ({len(seed_list)} seeds × {len(models)} models)")
        for seed in seed_list:
            dgp_seed = data_seed if data_seed is not None else seed
            kwargs = {**base, sweep_param: val, "seed": dgp_seed}
            spec = builder(**kwargs)
            if drop_indicator:
                spec = without_indicator_in_features(spec)
            if postprocess is not None:
                spec = postprocess(spec)
            for model in models:
                info = evaluate_gt_adaptability(
                    spec,
                    model,
                    seed=seed,
                    model_kwargs=mk.get(model),
                    metric=metric,
                )
                rows.extend(
                    gt_rows(
                        info,
                        sweep_param=sweep_param,
                        sweep_value=val,
                        data_seed=dgp_seed,
                        experiment_id=spec.experiment_id,
                        description=spec.description,
                        kind="gt",
                    )
                )
    return pd.DataFrame(rows)


def run_model_kwargs_sweep(
    spec_factory: Callable[[int], DatasetSpec],
    *,
    model: str,
    kw_name: str,
    kw_values: Iterable[Any],
    seeds: Iterable[int] = SEEDS_DEFAULT,
    fixed_kwargs: Optional[dict] = None,
    data_seed: Optional[int] = None,
    metric: str = "auc",
) -> pd.DataFrame:
    """Sweep one model hyperparameter on a fixed DGP factory(seed)->spec."""
    rows: list[dict[str, Any]] = []
    fixed = dict(fixed_kwargs or {})
    seed_list = list(seeds)
    for val in kw_values:
        _log(f"      {model} {kw_name}={val} ({len(seed_list)} seeds)")
        for seed in seed_list:
            dgp_seed = data_seed if data_seed is not None else seed
            spec = spec_factory(dgp_seed)
            kw = {**fixed, kw_name: val, "random_state": seed}
            info = evaluate_gt_adaptability(
                spec, model, seed=seed, model_kwargs=kw, metric=metric
            )
            rows.extend(
                gt_rows(
                    info,
                    sweep_param=kw_name,
                    sweep_value=str(val),
                    data_seed=dgp_seed,
                    experiment_id=spec.experiment_id,
                    description=spec.description,
                    kind="gt",
                )
            )
    return pd.DataFrame(rows)


def run_two_param_grid(
    builder: Callable[..., DatasetSpec],
    *,
    param_x: str,
    values_x: Iterable[Any],
    param_y: str,
    values_y: Iterable[Any],
    models: list[str],
    base_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict[str, dict]] = None,
    seeds: Iterable[int] = SEEDS_DEFAULT,
    metric: str = "auc",
) -> pd.DataFrame:
    """Cartesian sweep of two DGP parameters x models x seeds (tidy per-GT rows)."""
    rows: list[dict[str, Any]] = []
    base = dict(base_kwargs or {})
    mk = model_kwargs or {}
    for vx in values_x:
        for vy in values_y:
            for seed in seeds:
                kwargs = {**base, param_x: vx, param_y: vy, "seed": seed}
                spec = builder(**kwargs)
                for model in models:
                    info = evaluate_gt_adaptability(
                        spec,
                        model,
                        seed=seed,
                        model_kwargs=mk.get(model),
                        metric=metric,
                    )
                    rows.extend(
                        gt_rows(
                            info,
                            param_x=param_x,
                            value_x=vx,
                            param_y=param_y,
                            value_y=vy,
                            sweep_param=param_x,
                            sweep_value=vx,
                            experiment_id=spec.experiment_id,
                            kind="gt",
                        )
                    )
    return pd.DataFrame(rows)


def run_capacity_grid(
    builder: Callable[..., DatasetSpec],
    *,
    complexity_param: str,
    complexity_values: Iterable[Any],
    model: str,
    capacity_name: str,
    capacity_values: Iterable[Any],
    base_kwargs: Optional[dict] = None,
    fixed_kwargs: Optional[dict] = None,
    seeds: Iterable[int] = SEEDS_DEFAULT,
    metric: str = "auc",
) -> pd.DataFrame:
    """
    Mixture complexity x model capacity — the core grid for the model comparison.

    For rich hypothesis classes with the region indicator in the features, Q>0
    requires the *budget* to be too small for the global mixture while still
    sufficing for the local sub-problem. That only shows up when complexity and
    capacity are varied together.
    """
    rows: list[dict[str, Any]] = []
    base = dict(base_kwargs or {})
    fixed = dict(fixed_kwargs or {})
    seed_list = list(seeds)
    for cval in complexity_values:
        for cap in capacity_values:
            _log(
                f"      {complexity_param}={cval} {capacity_name}={cap} "
                f"({len(seed_list)} seeds, {model})"
            )
            for seed in seed_list:
                spec = builder(**{**base, complexity_param: cval, "seed": seed})
                kw = {**fixed, capacity_name: cap, "random_state": seed}
                info = evaluate_gt_adaptability(
                    spec, model, seed=seed, model_kwargs=kw, metric=metric
                )
                rows.extend(
                    gt_rows(
                        info,
                        param_x=capacity_name,
                        value_x=str(cap),
                        param_y=complexity_param,
                        value_y=cval,
                        sweep_param=capacity_name,
                        sweep_value=str(cap),
                        experiment_id=spec.experiment_id,
                        kind="gt",
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# reduction / aggregation / plots
# ---------------------------------------------------------------------------


def reduce_gt(
    df: pd.DataFrame,
    *,
    mode: str = "mean",
    gt: Optional[str] = None,
    value_col: str = "quality",
) -> pd.DataFrame:
    """
    Collapse per-GT rows to one row per (sweep cell, model, seed).

    ``mode``:
      - ``mean`` (Standard): Mittel über die Ground-Truth-Subgruppen
      - ``max``: Maximum (wächst mit der Zahl der Subgruppen)
      - ``single``: nur die Subgruppe ``gt``
    """
    if df is None or len(df) == 0:
        return df
    group_cols = [
        c
        for c in (
            "sweep_param",
            "sweep_value",
            "param_x",
            "value_x",
            "param_y",
            "value_y",
            "size_fraction",
            "kind",
            "model",
            "seed",
            "metric",
            "experiment_id",
        )
        if c in df.columns
    ]
    if mode == "single":
        if gt is None:
            raise ValueError("reduce_gt(mode='single') needs gt=...")
        return df[df["gt"] == gt].copy()

    agg_map = {
        value_col: mode,
        "quality_weighted": mode,
        "global_test_full": "mean",
        "local_test": "mean",
        "global_test": "mean",
        "n_test": "sum",
    }
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
    out = df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    if "gt" in df.columns:
        out["gt"] = f"<{mode} over GT>"
    return out


def aggregate_sweep(
    df: pd.DataFrame,
    *,
    gt_mode: str = "mean",
    value_col: str = "quality",
) -> pd.DataFrame:
    """Mean/std of quality over seeds, after reducing per-GT rows."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if "gt" in df.columns:
        df = reduce_gt(df, mode=gt_mode, value_col=value_col)
    keys = [c for c in ("sweep_param", "sweep_value", "model") if c in df.columns]
    agg_map = {
        "mean_Q": (value_col, "mean"),
        "std_Q": (value_col, "std"),
        "n": (value_col, "count"),
    }
    if "quality_weighted" in df.columns:
        agg_map["mean_Q_weighted"] = ("quality_weighted", "mean")
    if "global_test_full" in df.columns:
        agg_map["mean_global"] = ("global_test_full", "mean")
    return df.groupby(keys, dropna=False).agg(**agg_map).reset_index()


def summarize_table(df: pd.DataFrame, *, gt_mode: str = "mean") -> pd.DataFrame:
    return aggregate_sweep(df, gt_mode=gt_mode).sort_values(["model", "sweep_value"])


def failure_report(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per status value — makes swallowed failures visible."""
    if df is None or len(df) == 0 or "status" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["model", "status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["model", "n"], ascending=[True, False])
    )


def plot_sweep(
    df: pd.DataFrame,
    *,
    title: str,
    xlabel: str | None = None,
    ylabel: str = "GT quality Q",
    ax=None,
    numeric_x: bool = True,
    gt_mode: str = "mean",
    null_band: Optional[tuple[float, float]] = None,
):
    """Line plot of mean Q +/- std vs sweep value, one line per model."""
    if df is None or len(df) == 0:
        raise ValueError("plot_sweep: empty DataFrame")
    agg = aggregate_sweep(df, gt_mode=gt_mode)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    models = list(dict.fromkeys(agg["model"]))
    plotted = False
    for model in models:
        sub = agg[agg["model"] == model].copy()
        sub = sub[np.isfinite(sub["mean_Q"])]
        if sub.empty:
            continue
        plotted = True
        if numeric_x:
            try:
                sub["_x"] = pd.to_numeric(sub["sweep_value"])
                sub = sub.sort_values("_x")
                x = sub["_x"].to_numpy()
            except (ValueError, TypeError):
                sub["_x"] = np.arange(len(sub))
                x = sub["_x"].to_numpy()
                ax.set_xticks(x)
                ax.set_xticklabels(sub["sweep_value"].astype(str))
        else:
            sub["_x"] = np.arange(len(sub))
            x = sub["_x"].to_numpy()
            ax.set_xticks(x)
            ax.set_xticklabels(sub["sweep_value"].astype(str), rotation=30, ha="right")

        y = sub["mean_Q"].to_numpy()
        err = sub["std_Q"].fillna(0).to_numpy()
        ax.errorbar(x, y, yerr=err, marker="o", label=model, capsize=3)

    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    if null_band is not None:
        ax.axhspan(
            null_band[0],
            null_band[1],
            color="gray",
            alpha=0.15,
            label="matched null",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel or str(df["sweep_param"].iloc[0]))
    ax.set_ylabel(ylabel)
    if plotted:
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "no finite Q values", ha="center", va="center",
            transform=ax.transAxes,
        )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_quality_heatmap(
    df: pd.DataFrame,
    *,
    model: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax=None,
    gt_mode: str = "mean",
):
    """Heatmap of mean quality for one model over value_x x value_y."""
    sub = df[df["model"] == model]
    if sub.empty:
        raise ValueError(f"plot_quality_heatmap: no rows for model={model!r}")
    if "gt" in sub.columns:
        sub = reduce_gt(sub, mode=gt_mode)
    pivot = (
        sub.groupby(["value_y", "value_x"], dropna=False)["quality"]
        .mean()
        .unstack("value_x")
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    else:
        fig = ax.figure
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel(xlabel or str(df["param_x"].iloc[0]))
    ax.set_ylabel(ylabel or str(df["param_y"].iloc[0]))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean Q")
    fig.tight_layout()
    return fig, ax, pivot


# ---------------------------------------------------------------------------
# full subgroup discovery (used only for the discovery demonstration)
# ---------------------------------------------------------------------------


def run_sd_sweep_point(
    spec: DatasetSpec,
    model: str,
    *,
    seed: int = SEED_DEFAULT,
    model_kwargs: Optional[dict] = None,
    depth: int = 1,
    min_support: int = 20,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
    balance_weight: float = DEFAULT_BALANCE_WEIGHT,
    max_workers: int = 2,
) -> dict[str, Any]:
    """One full subgroup-discovery run (slower); returns summary info."""
    _, info = run_sd(
        spec,
        model,
        seed=seed,
        model_kwargs=model_kwargs,
        depth=depth,
        min_support=min_support,
        size_weight=size_weight,
        balance_weight=balance_weight,
        max_workers=max_workers,
        result_set_size=10,
    )
    return info
