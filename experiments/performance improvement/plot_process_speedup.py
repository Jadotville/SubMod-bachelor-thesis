"""
Thesis figures for chapter 4 (section on search scalability).

The measurements are the worker sweeps on Phishing Websites:

* ``worker_sweep_summary.csv`` — depth 2, full data, offline models
  (figure ``speedup_prozesse_threads``)
* ``worker_sweep_tabpfn_summary.csv`` — local GPU TabPFN, depth 2,
  train/test cap 1024, one estimator (figure ``speedup_tabpfn``)

An older plot of ``process_scale_summary.csv`` (depth 3) is superseded.
Writes ``speedup_prozesse_threads.png`` and ``speedup_tabpfn.png`` next to the CSVs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from worker_sweep import plot_speedup, plot_speedup_side_by_side

RESULTS = Path(__file__).resolve().parent / "results"


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
    else:
        print(f"fehlt: {tabpfn_path} (TabPFN-Abbildung übersprungen)")


if __name__ == "__main__":
    main()
