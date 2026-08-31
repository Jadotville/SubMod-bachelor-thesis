"""
Where does search time go, per implementation and model?

Instruments the quality-function path (global fit, cover/balance, local fit,
predict, AUC, weighting) plus search-space enumeration. Sequential DFS, the
thread pool and the process pool are measured on the same real-data task.

    python profile_time_breakdown.py --quick
    python profile_time_breakdown.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_RW_DIR = _PROJECT_ROOT / "experiments" / "experiments_real_world_data"
for _p in (_PROJECT_ROOT, _RW_DIR, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pysubgroup as ps
import pysubgroup.parallel_model_adaptability_dfs as par_dfs
from pysubgroup.parallel_model_adaptability_dfs import (
    ParallelModelAdaptabilityDFS,
    ProcessModelAdaptabilityDFS,
)

from datasets import feature_columns, prepare_dataset, stratified_split_fn
from models import OFFLINE_MODELS, resolve_model_hooks
from run import build_search_space, default_min_support

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STAGES = (
    "global_fit",
    "enumerate",
    "cover_and_balance",
    "local_fit",
    "predict",
    "score",
    "quality_weight",
)

STAGE_LABELS = {
    "global_fit": "globales Modell",
    "enumerate": "Enumeration",
    "cover_and_balance": "Cover / Balance",
    "local_fit": "lokaler Fit",
    "predict": "Vorhersage",
    "score": "AUC",
    "quality_weight": "Gewichtung",
    "other": "Suche / Pool / Rest",
}

ALGO_LABELS = {
    "dfs": "sequenziell (DFS)",
    "threads": "Threads",
    "processes": "Prozesse",
}

MODEL_LABELS = {
    "lr": "logistische Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
}

# Stacked-bar colours, work first, residual last.
STAGE_COLORS = {
    "local_fit": "#C0392B",
    "predict": "#E67E22",
    "score": "#F1C40F",
    "cover_and_balance": "#2980B9",
    "global_fit": "#8E44AD",
    "enumerate": "#16A085",
    "quality_weight": "#7F8C8D",
    "other": "#BDC3C7",
}


class StageClock:
    """Thread- or process-safe accumulators for named stages."""

    def __init__(self, *, shared: bool):
        self.shared = shared
        if shared:
            ctx = multiprocessing.get_context("fork")
            self._lock = ctx.Lock()
            self._sec = {name: ctx.Value("d", 0.0, lock=False) for name in STAGES}
            self._n = {name: ctx.Value("q", 0, lock=False) for name in STAGES}
        else:
            self._lock = threading.Lock()
            self._sec = {name: 0.0 for name in STAGES}
            self._n = {name: 0 for name in STAGES}

    def add(self, stage: str, dt: float) -> None:
        with self._lock:
            if self.shared:
                self._sec[stage].value += dt
                self._n[stage].value += 1
            else:
                self._sec[stage] += dt
                self._n[stage] += 1

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            if self.shared:
                return {
                    name: {
                        "sec": float(self._sec[name].value),
                        "n": int(self._n[name].value),
                    }
                    for name in STAGES
                }
            return {
                name: {"sec": float(self._sec[name]), "n": int(self._n[name])}
                for name in STAGES
            }


def _wrap(fn: Callable, clock: StageClock, stage: str) -> Callable:
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            clock.add(stage, time.perf_counter() - t0)

    wrapped.__name__ = getattr(fn, "__name__", stage)
    wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapped


def _instrument_qf(qf, clock: StageClock) -> None:
    qf.training_global = _wrap(qf.training_global, clock, "global_fit")
    qf.training_local = _wrap(qf.training_local, clock, "local_fit")
    qf.prediction_global = _wrap(qf.prediction_global, clock, "predict")
    qf.prediction_local = _wrap(qf.prediction_local, clock, "predict")
    qf.performance_measure = _wrap(qf.performance_measure, clock, "score")
    qf._covers_and_balance = _wrap(qf._covers_and_balance, clock, "cover_and_balance")
    qf._get_quality_weight = _wrap(qf._get_quality_weight, clock, "quality_weight")


@contextmanager
def _instrument_enumerate(clock: StageClock):
    """Time only the outermost recursive enumeration call."""
    original = par_dfs._enumerate_subgroups
    busy = threading.local()

    def wrapped(*args, **kwargs):
        if getattr(busy, "active", False):
            return original(*args, **kwargs)
        busy.active = True
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            clock.add("enumerate", time.perf_counter() - t0)
            busy.active = False

    par_dfs._enumerate_subgroups = wrapped
    try:
        yield
    finally:
        par_dfs._enumerate_subgroups = original


@dataclass
class ProfileConfig:
    dataset: str = "PhishingWebsites"
    model: str = "lr"
    algorithm: str = "dfs"
    max_workers: int = 32
    seed: int = 42
    depth: int = 2
    row_limit: int | None = None
    max_search_selectors: int | None = None
    result_set_size: int = 10
    size_weight: float = 0.7
    balance_weight: float = 0.0
    train_cap: int | None = None
    test_cap: int | None = None
    tabpfn_n_estimators: int | None = None


def _make_algorithm(cfg: ProfileConfig):
    if cfg.algorithm == "dfs":
        return ps.DFS()
    if cfg.algorithm == "threads":
        return ParallelModelAdaptabilityDFS(max_workers=cfg.max_workers)
    if cfg.algorithm == "processes":
        return ProcessModelAdaptabilityDFS(max_workers=cfg.max_workers)
    raise ValueError(f"Unknown algorithm {cfg.algorithm!r}")


def build_task(cfg: ProfileConfig, counter=None):
    import pandas as pd

    df, spec = prepare_dataset(cfg.dataset, seed=cfg.seed, row_limit=cfg.row_limit)
    split_fn = stratified_split_fn(
        seed=cfg.seed,
        test_size=0.5,
        train_cap=cfg.train_cap,
        test_cap=cfg.test_cap,
    )
    train_for_search, test_heldout = split_fn(df)
    if cfg.train_cap is not None or cfg.test_cap is not None:
        n_train = len(train_for_search)
        df = pd.concat([train_for_search, test_heldout], ignore_index=True)

        def split_fn(data, _n: int = n_train):
            return data.iloc[:_n].copy(), data.iloc[_n:].copy()

        train_for_search = df.iloc[:n_train].copy()

    min_support = default_min_support(len(df))
    features = feature_columns(df)
    search_space = build_search_space(
        train_for_search,
        max(1, min_support // 2),
        cfg.max_search_selectors,
    )
    builder, train_g, train_l, pred_g, pred_l = resolve_model_hooks(
        cfg.model, cfg.seed, counter, cfg.tabpfn_n_estimators
    )
    qf = ps.LocalSoftClassifierPerformanceQF(
        model_builder_global=builder,
        training_global=train_g,
        training_local=train_l,
        prediction_global=pred_g,
        prediction_local=pred_l,
        model_builder_local=builder,
        split_fn=split_fn,
        random_state=cfg.seed,
        subgroup_size_weight=cfg.size_weight,
        subgroup_class_balance_weight=cfg.balance_weight,
    )
    target = ps.ModelAdaptabilityTarget(label_column="target", feature_columns=features)
    task = ps.ModelAdaptabilityDiscoveryTask(
        data=df,
        target=target,
        search_space=search_space,
        qf=qf,
        depth=cfg.depth,
        result_set_size=cfg.result_set_size,
        constraints=[ps.MinSupportConstraint(min_support)],
        generalization_awareness=True,
    )
    meta = {
        "n_rows": int(len(df)),
        "n_train": int(len(train_for_search)),
        "n_test": int(len(df) - len(train_for_search)),
        "n_features": int(len(features)),
        "min_support": int(min_support),
        "n_selectors": int(len(search_space)),
        "max_search_selectors": cfg.max_search_selectors
        if cfg.max_search_selectors is not None
        else "",
        "train_cap": cfg.train_cap if cfg.train_cap is not None else "",
        "test_cap": cfg.test_cap if cfg.test_cap is not None else "",
        "tabpfn_n_estimators": (
            cfg.tabpfn_n_estimators if cfg.tabpfn_n_estimators is not None else ""
        ),
        "openml_id": spec.openml_id,
    }
    return task, meta


def profile_once(cfg: ProfileConfig) -> dict[str, Any]:
    shared = cfg.algorithm == "processes"
    clock = StageClock(shared=shared)
    task, meta = build_task(cfg)
    _instrument_qf(task.qf, clock)
    algorithm = _make_algorithm(cfg)
    t0 = time.perf_counter()
    with _instrument_enumerate(clock):
        result = task.execute(algorithm=algorithm)
    wall = time.perf_counter() - t0
    snap = clock.snapshot()
    n_result = len(getattr(result, "results", result))
    n_fits = int(snap["local_fit"]["n"])
    n_covers = int(snap["cover_and_balance"]["n"])
    work_sec = sum(item["sec"] for item in snap.values())
    other_sec = max(0.0, wall - work_sec)
    row: dict[str, Any] = {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "algorithm": cfg.algorithm,
        "max_workers": 1 if cfg.algorithm == "dfs" else cfg.max_workers,
        "depth": cfg.depth,
        "row_limit": cfg.row_limit if cfg.row_limit is not None else "",
        "seed": cfg.seed,
        "wall_sec": wall,
        "work_sec": work_sec,
        "other_sec": other_sec,
        "n_result_rows": n_result,
        "n_local_fits": n_fits,
        "n_cover_calls": n_covers,
        **meta,
    }
    for name in STAGES:
        row[f"{name}_sec"] = snap[name]["sec"]
        row[f"{name}_n"] = snap[name]["n"]
        row[f"{name}_pct_work"] = (
            100.0 * snap[name]["sec"] / work_sec if work_sec > 0 else 0.0
        )
        row[f"{name}_pct_wall"] = 100.0 * snap[name]["sec"] / wall if wall > 0 else 0.0
    row["other_pct_wall"] = 100.0 * other_sec / wall if wall > 0 else 0.0
    row["work_over_wall"] = work_sec / wall if wall > 0 else float("nan")
    return row


def _long_table(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in raw.iterrows():
        wall = float(row["wall_sec"])
        for stage in STAGES:
            sec = float(row[f"{stage}_sec"])
            records.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "algorithm": row["algorithm"],
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "sec": sec,
                    "n": int(row[f"{stage}_n"]),
                    "pct_work": float(row[f"{stage}_pct_work"]),
                    "pct_wall": float(row[f"{stage}_pct_wall"]),
                    "wall_sec": wall,
                    "work_over_wall": float(row["work_over_wall"]),
                    "n_local_fits": int(row["n_local_fits"]),
                    "n_selectors": int(row["n_selectors"]),
                    "n_rows": int(row["n_rows"]),
                    "depth": int(row["depth"]),
                    "max_workers": int(row["max_workers"]),
                }
            )
        records.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "algorithm": row["algorithm"],
                "stage": "other",
                "stage_label": STAGE_LABELS["other"],
                "sec": float(row["other_sec"]),
                "n": 1,
                "pct_work": 0.0,
                "pct_wall": float(row["other_pct_wall"]),
                "wall_sec": wall,
                "work_over_wall": float(row["work_over_wall"]),
                "n_local_fits": int(row["n_local_fits"]),
                "n_selectors": int(row["n_selectors"]),
                "n_rows": int(row["n_rows"]),
                "depth": int(row["depth"]),
                "max_workers": int(row["max_workers"]),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_breakdown(long_df: pd.DataFrame, path: Path) -> None:
    """Stacked composition of instrumented work (always sums to 100 %)."""
    models = [m for m in ("lr", "rf", "lgbm", "mlp") if m in set(long_df["model"])]
    algos = [a for a in ("dfs", "threads", "processes") if a in set(long_df["algorithm"])]
    fig, axes = plt.subplots(
        1, len(algos), figsize=(4.4 * len(algos) + 1.8, 5.4), sharey=True
    )
    if len(algos) == 1:
        axes = [axes]
    x = np.arange(len(models))
    width = 0.62
    for ax, algo in zip(axes, algos):
        part = long_df[long_df["algorithm"] == algo]
        bottoms = np.zeros(len(models))
        walls = []
        for model in models:
            cell = part[part["model"] == model]
            walls.append(float(cell["wall_sec"].iloc[0]) if len(cell) else 0.0)
        for stage in STAGES:
            heights = []
            for model in models:
                cell = part[(part["model"] == model) & (part["stage"] == stage)]
                heights.append(float(cell["pct_work"].iloc[0]) if len(cell) else 0.0)
            ax.bar(
                x,
                np.asarray(heights),
                width,
                bottom=bottoms,
                color=STAGE_COLORS[stage],
                label=STAGE_LABELS[stage],
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms = bottoms + np.asarray(heights)
        for i, wall in enumerate(walls):
            ax.text(i, 101.5, f"{wall:.1f} s", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS.get(m, m) for m in models], rotation=25, ha="right"
        )
        ax.set_title(ALGO_LABELS[algo], fontsize=12)
        ax.set_ylim(0, 112)
        ax.axhline(100, color="#CCCCCC", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("Anteil an der instrumentierten Arbeit (%)", fontsize=11)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=8.5,
        frameon=False,
    )
    n_rows = int(long_df["n_rows"].iloc[0])
    depth = int(long_df["depth"].iloc[0])
    dataset = str(long_df["dataset"].iloc[0])
    fig.suptitle(
        f"Zeitanteile der Suche — {dataset}, Tiefe {depth}, {n_rows} Zeilen",
        fontsize=13,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_markdown(raw: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Zeitanteile der Subgruppensuche",
        "",
        f"Datensatz: **{raw['dataset'].iloc[0]}**, "
        f"{int(raw['n_rows'].iloc[0])} Zeilen, "
        f"Tiefe {int(raw['depth'].iloc[0])}, "
        f"{int(raw['n_selectors'].iloc[0])} Selektoren, "
        f"{int(raw['n_cover_calls'].iloc[0])} Cover-Aufrufe "
        f"(eine Zelle; gleiche Zahl je Modell/Algorithmus bis auf DFS-Pruning).",
        "",
        "Die Prozentangaben in der Tabelle sind Anteile an der **instrumentierten "
        "Arbeit** (Summe der Stufen). Die Wandzeit steht extra. "
        "`Arbeit/Wand` > 1 bedeutet, dass Worker-Zeiten überlappen "
        "(Prozesse: echte Parallelität; Threads: oft GIL-/BLAS-Contention). "
        "`Rest der Wand` ist nur sinnvoll, wenn Arbeit/Wand ≤ 1.",
        "",
    ]
    for algo in ("dfs", "threads", "processes"):
        sub = raw[raw["algorithm"] == algo]
        if sub.empty:
            continue
        lines.append(f"## {ALGO_LABELS[algo]}")
        lines.append("")
        lines.append(
            "| Modell | Wandzeit | lokaler Fit | Cover/Balance | "
            "Vorhersage+AUC | Rest der Wand | Fits | Arbeit/Wand |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in sub.iterrows():
            pred_auc = float(row["predict_pct_work"] + row["score_pct_work"])
            lines.append(
                "| {model} | {wall:.1f} s | {fit:.1f} % | {cover:.1f} % | "
                "{pred:.1f} % | {other:.1f} % | {fits} | {wow:.2f}× |".format(
                    model=MODEL_LABELS.get(row["model"], row["model"]),
                    wall=row["wall_sec"],
                    fit=row["local_fit_pct_work"],
                    cover=row["cover_and_balance_pct_work"],
                    pred=pred_auc,
                    other=row["other_pct_wall"],
                    fits=int(row["n_local_fits"]),
                    wow=row["work_over_wall"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PhishingWebsites")
    parser.add_argument("--models", nargs="+", default=list(OFFLINE_MODELS))
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["processes", "dfs", "threads"],
        choices=["dfs", "threads", "processes"],
    )
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 1500 rows, depth 1, lr only.",
    )
    parser.add_argument("--tag", default="time_breakdown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models
    algorithms = args.algorithms
    depth = args.depth
    row_limit = args.row_limit
    if args.quick:
        models = ["lr"]
        depth = 1
        row_limit = 1500
        args.tag = "time_breakdown_quick"

    configs = [
        ProfileConfig(
            dataset=args.dataset,
            model=model,
            algorithm=algorithm,
            max_workers=args.workers,
            seed=args.seed,
            depth=depth,
            row_limit=row_limit,
        )
        for model in models
        for algorithm in algorithms
    ]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, cfg in enumerate(configs, start=1):
        label = f"{cfg.dataset}/{cfg.model}/{cfg.algorithm}"
        print(f"[{i}/{len(configs)}] {label} depth={cfg.depth}", flush=True)
        row = profile_once(cfg)
        rows.append(row)
        print(
            f"    wall={row['wall_sec']:.2f}s  "
            f"fit={row['local_fit_pct_wall']:.1f}% of wall  "
            f"fits={row['n_local_fits']}  "
            f"work/wall={row['work_over_wall']:.2f}",
            flush=True,
        )

    raw = pd.DataFrame(rows)
    long_df = _long_table(raw)
    raw_path = RESULTS_DIR / f"{args.tag}_raw.csv"
    long_path = RESULTS_DIR / f"{args.tag}_long.csv"
    md_path = RESULTS_DIR / f"{args.tag}.md"
    png_path = RESULTS_DIR / f"{args.tag}.png"
    raw.to_csv(raw_path, index=False)
    long_df.to_csv(long_path, index=False)
    write_markdown(raw, md_path)
    plot_breakdown(long_df, png_path)
    meta = {
        "dataset": args.dataset,
        "models": models,
        "algorithms": algorithms,
        "depth": depth,
        "workers": args.workers,
        "row_limit": row_limit,
        "elapsed_sec": time.perf_counter() - started,
        "files": [str(p) for p in (raw_path, long_path, md_path, png_path)],
    }
    (RESULTS_DIR / f"{args.tag}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\n{raw.to_string(index=False, float_format=lambda v: f'{v:8.3f}')}")
    print(f"\n{time.perf_counter() - started:.0f}s -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
