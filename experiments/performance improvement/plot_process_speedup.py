"""
Thesis figures for chapter 4 (section on search scalability).

The measurements are the worker sweeps on Phishing Websites:

* ``worker_sweep_summary.csv`` — depth 2, full data, offline models
  (figure ``speedup_prozesse_threads``)
* ``worker_sweep_tabpfn_summary.csv`` — local GPU TabPFN, depth 2,
  train/test cap 1024, one estimator (figure ``speedup_tabpfn``)

Both sweeps together give ``speedup_prozesse``, the thesis figure. It shows the
process pool only. Each curve is normalised against its own sequential DFS
baseline, so the two workload sizes share one panel without being put on a
common absolute scale.

An older plot of ``process_scale_summary.csv`` (depth 3) is superseded.
Writes the three PNGs next to the CSVs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from worker_sweep import (
    MODEL_COLORS,
    MODEL_LABELS,
    _finish_speedup_ax,
    _speedup_workers,
    plot_speedup,
    plot_speedup_side_by_side,
)

RESULTS = Path(__file__).resolve().parent / "results"


def plot_process_speedup(summary: pd.DataFrame, path: Path) -> None:
    """Process speedup of all models in a single panel. The LaTeX caption
    carries the setup, so the figure itself stays untitled."""
    workers = _speedup_workers(summary)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ymax = 0.0
    for model in ("lr", "rf", "lgbm", "mlp", "tabpfn"):
        sub = summary[
            (summary["model"] == model) & (summary["algorithm"] == "processes")
        ].sort_values("max_workers")
        if sub.empty:
            continue
        y = sub["speedup"].astype(float)
        ax.plot(
            sub["max_workers"],
            y,
            "o-",
            color=MODEL_COLORS[model],
            lw=2.1,
            markersize=6,
            label=MODEL_LABELS[model],
            zorder=3,
        )
        ymax = max(ymax, float(y.max()))
    _finish_speedup_ax(ax, workers, ymax, "")
    ax.set_ylabel("Beschleunigung gegenüber der sequenziellen Suche", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    offline = pd.read_csv(RESULTS / "worker_sweep_summary.csv")
    plot_speedup_side_by_side(
        offline,
        RESULTS / "speedup_prozesse_threads.png",
        title="Beschleunigung gegenüber der sequenziellen Suche",
        subtitle="Phishing Websites, Tiefe 2, n = 11055",
    )
    print(f"geschrieben: {RESULTS / 'speedup_prozesse_threads.png'}")

    tabpfn_path = RESULTS / "worker_sweep_tabpfn_summary.csv"
    if tabpfn_path.is_file():
        tabpfn = pd.read_csv(tabpfn_path)
        plot_speedup(
            tabpfn,
            RESULTS / "speedup_tabpfn.png",
            title="TabPFN: Laufzeit und Beschleunigung",
            subtitle="TabPFN lokal, Tiefe 2, train/test-cap 1024, 1 estimator",
        )
        print(f"geschrieben: {RESULTS / 'speedup_tabpfn.png'}")

        plot_process_speedup(
            pd.concat([offline, tabpfn], ignore_index=True),
            RESULTS / "speedup_prozesse.png",
        )
        print(f"geschrieben: {RESULTS / 'speedup_prozesse.png'}")
    else:
        print(f"fehlt: {tabpfn_path} (TabPFN-Abbildung übersprungen)")


if __name__ == "__main__":
    main()
