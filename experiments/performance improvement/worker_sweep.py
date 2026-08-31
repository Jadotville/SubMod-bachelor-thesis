"""
Worker sweep: threads and processes vs sequential DFS.

Same task as ``profile_time_breakdown.py`` (PhishingWebsites, full search space,
depth 2). Sequential DFS is the baseline (one run per model). Threads and
processes are timed at each worker count. Cells are appended immediately so a
killed tmux session can resume with ``--skip-existing``.

    python worker_sweep.py
    python worker_sweep.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from profile_time_breakdown import (  # noqa: E402
    OFFLINE_MODELS,
    ProfileConfig,
    _make_algorithm,
    build_task,
)

RESULTS_DIR = _HERE / "results"
DEFAULT_WORKERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

MODEL_COLORS = {
    "lr": "#2F6F9F",
    "rf": "#7A3E9D",
    "lgbm": "#B8860B",
    "mlp": "#C0392B",
    "tabpfn": "#1A7A62",
}
MODEL_LABELS = {
    "lr": "logistische Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "tabpfn": "TabPFN",
}


def _cell_key(row: dict[str, Any]) -> tuple:
    return (
        str(row["dataset"]),
        str(row["model"]),
        str(row["algorithm"]),
        int(row["max_workers"]),
        int(row["seed"]),
        int(row["repeat"]),
    )


def _load_done(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    existing = pd.read_csv(path)
    if existing.empty:
        return set()
    return {_cell_key(row) for row in existing.to_dict(orient="records")}


def _append_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    header = not path.is_file()
    frame.to_csv(path, mode="a", header=header, index=False)


def time_once(cfg: ProfileConfig) -> dict[str, Any]:
    task, meta = build_task(cfg)
    algorithm = _make_algorithm(cfg)
    t0 = time.perf_counter()
    _ = task.execute(algorithm=algorithm)
    elapsed = time.perf_counter() - t0
    workers = 1 if cfg.algorithm == "dfs" else cfg.max_workers
    return {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "algorithm": cfg.algorithm,
        "max_workers": workers,
        "depth": cfg.depth,
        "row_limit": cfg.row_limit if cfg.row_limit is not None else "",
        "seed": cfg.seed,
        "runtime_sec": elapsed,
        **meta,
    }


def _configs(args: argparse.Namespace) -> list[ProfileConfig]:
    configs: list[ProfileConfig] = []
    base = dict(
        dataset=args.dataset,
        seed=args.seed,
        depth=args.depth,
        row_limit=args.row_limit,
    )
    for model in args.models:
        configs.append(ProfileConfig(model=model, algorithm="dfs", max_workers=1, **base))
        for algorithm in args.algorithms:
            for workers in args.workers:
                configs.append(
                    ProfileConfig(
                        model=model,
                        algorithm=algorithm,
                        max_workers=workers,
                        **base,
                    )
                )
    return configs


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        raw.groupby(["dataset", "model", "algorithm", "max_workers"], dropna=False)[
            "runtime_sec"
        ]
        .agg(runtime_median="median", runtime_min="min", runs="size")
        .reset_index()
    )
    baseline = (
        grouped[grouped["algorithm"] == "dfs"]
        .set_index(["dataset", "model"])["runtime_median"]
        .to_dict()
    )
    grouped["dfs_sec"] = [
        baseline.get((dataset, model))
        for dataset, model in zip(grouped["dataset"], grouped["model"])
    ]
    grouped["speedup"] = grouped["dfs_sec"] / grouped["runtime_median"]
    return grouped.sort_values(["dataset", "model", "algorithm", "max_workers"])


_ALGO_STYLES = {
    "processes": ("o", "-", 1.0, 3, "Prozesse"),
    "threads": ("s", "--", 0.7, 2, "Threads"),
}


def plot_speedup(
    summary: pd.DataFrame,
    path: Path,
    *,
    subtitle: str | None = None,
    algorithm: str | None = None,
    title: str | None = None,
) -> None:
    if algorithm is not None:
        summary = summary[summary["algorithm"].isin(["dfs", algorithm])].copy()
    workers = sorted(
        {
            int(w)
            for w in summary.loc[summary["algorithm"] != "dfs", "max_workers"]
        }
    )
    parallel = [name for name in ("processes", "threads") if name in set(summary["algorithm"])]
    annotate = len(workers) <= 8
    fig, (ax_time, ax_speed) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    def _draw_model(ax, ycol: str) -> float:
        ymax = 0.0
        n_models = summary.loc[summary["algorithm"] != "dfs", "model"].nunique()
        for model in ("lr", "rf", "lgbm", "mlp", "tabpfn"):
            part = summary[summary["model"] == model]
            if part.empty:
                continue
            color = MODEL_COLORS[model]
            label = MODEL_LABELS[model]
            for algo in parallel:
                marker, ls, alpha, zorder, algo_label = _ALGO_STYLES[algo]
                sub = part[part["algorithm"] == algo].sort_values("max_workers")
                if sub.empty:
                    continue
                y = sub[ycol].astype(float)
                if len(parallel) == 1:
                    line_label = label if n_models > 1 else algo_label
                else:
                    prefix = f"{label} — " if n_models > 1 else ""
                    line_label = f"{prefix}{algo_label}"
                ax.plot(
                    sub["max_workers"],
                    y,
                    marker + ls,
                    color=color,
                    lw=2.1,
                    markersize=6,
                    alpha=alpha,
                    label=line_label,
                    zorder=zorder,
                )
                ymax = max(ymax, float(y.max()))
                if annotate:
                    for x_val, y_val in zip(sub["max_workers"], y):
                        ax.annotate(
                            f"{y_val:.2f}",
                            (x_val, y_val),
                            textcoords="offset points",
                            xytext=(0, 7),
                            ha="center",
                            fontsize=7.5,
                            color=color,
                        )
        return ymax

    ymax_t = _draw_model(ax_time, "runtime_median")
    ymax_s = _draw_model(ax_speed, "speedup")
    dfs = summary[summary["algorithm"] == "dfs"]
    if len(dfs) == 1:
        dfs_sec = float(dfs["runtime_median"].iloc[0])
        ax_time.axhline(dfs_sec, color="#1A1A1A", lw=1.3, ls=":", zorder=1)
        ax_time.text(
            workers[0] if workers else 1,
            dfs_sec,
            f"  DFS ({dfs_sec:.2f} s)",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#1A1A1A",
        )
    ax_speed.axhline(1.0, color="#1A1A1A", lw=1.3, ls=":", zorder=1)
    ax_speed.text(
        workers[0] if workers else 1,
        1.0,
        "  sequenzielle Suche",
        va="bottom",
        ha="left",
        fontsize=8,
        color="#1A1A1A",
    )
    for ax in (ax_time, ax_speed):
        if workers:
            ax.set_xticks(workers)
        ax.set_xlabel("Anzahl Worker", fontsize=11)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8.5)
        ax.set_xlim((workers[0] - 0.3, workers[-1] + 0.3) if workers else (0, 1))
    ax_time.set_ylabel("Laufzeit (s)", fontsize=11)
    ax_speed.set_ylabel("Beschleunigung gegenüber DFS", fontsize=11)
    ax_time.set_ylim(0, ymax_t * 1.22 if ymax_t else 1)
    ax_speed.set_ylim(0, max(1.2, ymax_s) * 1.18 if ymax_s else 1.2)
    ax_time.legend(loc="upper right", fontsize=8, framealpha=0.95)
    dataset = str(summary["dataset"].iloc[0]) if len(summary) else ""
    if title is None:
        title = f"Worker-Sweep — {dataset}"
        if algorithm in _ALGO_STYLES:
            title = f"{title} — {_ALGO_STYLES[algorithm][4]}"
        if subtitle:
            title = f"{title}\n{subtitle}"
    elif subtitle:
        title = f"{title}\n{subtitle}"
    fig.suptitle(title, fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_speedup_side_by_side(
    summary: pd.DataFrame,
    path: Path,
    *,
    subtitle: str | None = None,
    title: str | None = None,
) -> None:
    """Speedup for processes and threads as two panels, each with its own y-scale."""
    workers = sorted(
        {
            int(w)
            for w in summary.loc[summary["algorithm"] != "dfs", "max_workers"]
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    n_models = summary.loc[summary["algorithm"] != "dfs", "model"].nunique()

    for ax, algo in zip(axes, ("processes", "threads")):
        marker, ls, alpha, zorder, algo_label = _ALGO_STYLES[algo]
        ymax = 0.0
        for model in ("lr", "rf", "lgbm", "mlp", "tabpfn"):
            sub = summary[
                (summary["model"] == model) & (summary["algorithm"] == algo)
            ].sort_values("max_workers")
            if sub.empty:
                continue
            y = sub["speedup"].astype(float)
            ax.plot(
                sub["max_workers"],
                y,
                marker + ls,
                color=MODEL_COLORS[model],
                lw=2.1,
                markersize=6,
                alpha=alpha,
                label=MODEL_LABELS[model] if n_models > 1 else algo_label,
                zorder=zorder,
            )
            ymax = max(ymax, float(y.max()))
        ax.axhline(1.0, color="#1A1A1A", lw=1.3, ls=":", zorder=1)
        ax.text(
            workers[0] if workers else 1,
            1.0,
            "  sequenzielle Suche",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#1A1A1A",
        )
        if workers:
            ax.set_xticks(workers)
            ax.set_xlim(workers[0] - 0.3, workers[-1] + 0.3)
        ax.set_xlabel("Anzahl Worker", fontsize=11)
        ax.set_title(algo_label, fontsize=12)
        ax.set_ylim(0, max(1.2, ymax) * 1.18 if ymax else 1.2)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8.5)

    axes[0].set_ylabel("Beschleunigung gegenüber DFS", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, max(1, len(handles))),
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
    )
    dataset = str(summary["dataset"].iloc[0]) if len(summary) else ""
    if title is None:
        title = f"Worker-Sweep — {dataset} — Beschleunigung"
        if subtitle:
            title = f"{title}\n{subtitle}"
    elif subtitle:
        title = f"{title}\n{subtitle}"
    fig.suptitle(title, fontsize=13, y=1.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _speedup_workers(summary: pd.DataFrame) -> list[int]:
    return sorted(
        {int(w) for w in summary.loc[summary["algorithm"] != "dfs", "max_workers"]}
    )


def _finish_speedup_ax(ax, workers: list[int], ymax: float, title: str) -> None:
    ax.axhline(1.0, color="#1A1A1A", lw=1.3, ls=":", zorder=1)
    ax.text(
        workers[0] if workers else 1,
        1.0,
        "  sequenzielle Suche",
        va="bottom",
        ha="left",
        fontsize=8,
        color="#1A1A1A",
    )
    if workers:
        ax.set_xticks(workers)
        ax.set_xlim(workers[0] - 0.3, workers[-1] + 0.3)
    ax.set_xlabel("Anzahl Worker", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(1.2, ymax) * 1.18 if ymax else 1.2)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8.5)


def plot_speedup_three_panel(
    offline: pd.DataFrame,
    tabpfn: pd.DataFrame,
    path: Path,
    *,
    subtitle: str | None = None,
) -> None:
    """Processes, threads (offline models) and TabPFN speedup, each with its own y-scale."""
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    n_models = offline.loc[offline["algorithm"] != "dfs", "model"].nunique()

    for ax, algo in zip(axes[:2], ("processes", "threads")):
        marker, ls, alpha, zorder, algo_label = _ALGO_STYLES[algo]
        ymax = 0.0
        workers = _speedup_workers(offline[offline["algorithm"].isin(["dfs", algo])])
        for model in ("lr", "rf", "lgbm", "mlp"):
            sub = offline[
                (offline["model"] == model) & (offline["algorithm"] == algo)
            ].sort_values("max_workers")
            if sub.empty:
                continue
            y = sub["speedup"].astype(float)
            ax.plot(
                sub["max_workers"],
                y,
                marker + ls,
                color=MODEL_COLORS[model],
                lw=2.1,
                markersize=6,
                alpha=alpha,
                label=MODEL_LABELS[model] if n_models > 1 else algo_label,
                zorder=zorder,
            )
            ymax = max(ymax, float(y.max()))
        _finish_speedup_ax(ax, workers, ymax, algo_label)

    ax_t = axes[2]
    ymax_t = 0.0
    workers_t = _speedup_workers(tabpfn)
    for algo in ("processes", "threads"):
        marker, ls, alpha, zorder, algo_label = _ALGO_STYLES[algo]
        sub = tabpfn[tabpfn["algorithm"] == algo].sort_values("max_workers")
        if sub.empty:
            continue
        y = sub["speedup"].astype(float)
        ax_t.plot(
            sub["max_workers"],
            y,
            marker + ls,
            color=MODEL_COLORS["tabpfn"],
            lw=2.1,
            markersize=6,
            alpha=alpha,
            label=algo_label,
            zorder=zorder,
        )
        ymax_t = max(ymax_t, float(y.max()))
    _finish_speedup_ax(ax_t, workers_t, ymax_t, "TabPFN")
    ax_t.legend(loc="upper right", fontsize=8, framealpha=0.95)

    axes[0].set_ylabel("Beschleunigung gegenüber DFS", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, max(1, len(handles))),
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.38, 0.02),
        bbox_transform=fig.transFigure,
    )
    dataset = str(offline["dataset"].iloc[0]) if len(offline) else ""
    title = f"Worker-Sweep — {dataset} — Beschleunigung"
    if subtitle:
        title = f"{title}\n{subtitle}"
    fig.suptitle(title, fontsize=13, y=1.04)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PhishingWebsites")
    parser.add_argument("--models", nargs="+", default=list(OFFLINE_MODELS))
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["processes", "threads"],
        choices=["threads", "processes"],
    )
    parser.add_argument("--workers", nargs="+", type=int, default=list(DEFAULT_WORKERS))
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--tag", default="worker_sweep")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cells already present in the raw CSV (resume after a crash).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 1500 rows, depth 1, lr, workers 1 2 4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.models = ["lr"]
        args.depth = 1
        args.row_limit = 1500
        args.workers = [1, 2, 4]
        args.tag = "worker_sweep_quick"
        args.repeats = 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / f"{args.tag}_raw.csv"
    summary_path = RESULTS_DIR / f"{args.tag}_summary.csv"
    png_path = RESULTS_DIR / f"{args.tag}.png"
    done = _load_done(raw_path) if args.skip_existing else set()
    warmed: set[tuple[str, str]] = set()
    configs = _configs(args)
    n_cells = len(configs) * args.repeats
    started = time.perf_counter()
    index = 0
    for cfg in configs:
        for repeat in range(args.repeats):
            index += 1
            workers = 1 if cfg.algorithm == "dfs" else cfg.max_workers
            probe = {
                "dataset": cfg.dataset,
                "model": cfg.model,
                "algorithm": cfg.algorithm,
                "max_workers": workers,
                "seed": cfg.seed,
                "repeat": repeat,
            }
            label = (
                f"{cfg.dataset}/{cfg.model}/{cfg.algorithm}_w{workers} "
                f"repeat={repeat}"
            )
            if _cell_key(probe) in done:
                print(f"[{index}/{n_cells}] skip {label}", flush=True)
                continue
            print(f"[{index}/{n_cells}] {label}", flush=True)
            pair = (cfg.model, cfg.algorithm)
            if pair not in warmed:
                time_once(cfg)
                warmed.add(pair)
            row = time_once(cfg)
            row["repeat"] = repeat
            _append_row(raw_path, row)
            done.add(_cell_key(row))
            print(f"    {row['runtime_sec']:.3f}s", flush=True)

    if not raw_path.is_file():
        print("keine Messungen", file=sys.stderr)
        return 1
    raw = pd.read_csv(raw_path)
    summary = summarize(raw)
    summary.to_csv(summary_path, index=False)
    depth = int(raw["depth"].iloc[0])
    n_rows = int(raw["n_rows"].iloc[0])
    row_limit = raw["row_limit"].iloc[0]
    limit_note = (
        f"{int(row_limit)} Zeilen (row_limit)"
        if pd.notna(row_limit) and str(row_limit) not in {"", "nan"}
        else f"{n_rows} Zeilen"
    )
    subtitle = f"Kurztest, Tiefe {depth}, {limit_note}" if args.quick else f"Tiefe {depth}, {n_rows} Zeilen"
    plot_speedup(summary, png_path, subtitle=subtitle)
    split_files: list[str] = []
    for algo in ("processes", "threads"):
        if algo not in set(summary["algorithm"]):
            continue
        split_path = RESULTS_DIR / f"{args.tag}_{algo}.png"
        plot_speedup(summary, split_path, subtitle=subtitle, algorithm=algo)
        split_files.append(str(split_path))
    if {"processes", "threads"} <= set(summary["algorithm"]):
        speedup_path = RESULTS_DIR / f"{args.tag}_speedup.png"
        plot_speedup_side_by_side(summary, speedup_path, subtitle=subtitle)
        split_files.append(str(speedup_path))
        tabpfn_summary_path = RESULTS_DIR / "worker_sweep_tabpfn_summary.csv"
        if tabpfn_summary_path.is_file():
            tabpfn_summary = pd.read_csv(tabpfn_summary_path)
            tabpfn_raw_path = RESULTS_DIR / "worker_sweep_tabpfn_raw.csv"
            tabpfn_note = "TabPFN lokal"
            if tabpfn_raw_path.is_file():
                tabpfn_raw = pd.read_csv(tabpfn_raw_path)
                tabpfn_note = (
                    f"Tiefe {int(tabpfn_raw['depth'].iloc[0])}, "
                    f"{int(tabpfn_raw['n_rows'].iloc[0])} Zeilen"
                )
            three_path = RESULTS_DIR / f"{args.tag}_speedup_three.png"
            plot_speedup_three_panel(
                summary,
                tabpfn_summary,
                three_path,
                subtitle=f"Prozesse/Threads: {subtitle}  ·  TabPFN: {tabpfn_note}",
            )
            split_files.append(str(three_path))
    meta = {
        "dataset": args.dataset,
        "models": args.models,
        "algorithms": ["dfs", *args.algorithms],
        "workers": args.workers,
        "depth": args.depth,
        "repeats": args.repeats,
        "elapsed_sec": time.perf_counter() - started,
        "n_rows_raw": int(len(raw)),
        "files": [str(p) for p in (raw_path, summary_path, png_path)] + split_files,
    }
    (RESULTS_DIR / f"{args.tag}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("\n=== Median runtime and speedup over sequential DFS")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\n{time.perf_counter() - started:.0f}s -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
