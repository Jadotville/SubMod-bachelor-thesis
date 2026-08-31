"""
Read a finished batch and answer the two questions it was run for.

1. **What was found?** One row per dataset and model. ``n_interesting`` follows
   the thesis criterion: evaluable rows with raw Q > 0 and Q_ga >= 0.
2. **What did it cost?** Search time and total time per cell, plus the
   per-candidate cost, which is the only runtime figure comparable across
   datasets of different size.

    python summarize_runs.py --results-dir results/offline
    python summarize_runs.py --results-dir results/offline --top 20   # best subgroups
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import dataset_role  # noqa: E402


def _read_meta(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def collect(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (findings, subgroups) over every completed cell in ``results_dir``."""
    findings: list[dict[str, Any]] = []
    subgroup_frames: list[pd.DataFrame] = []

    # Smoke runs nest their cells one level deeper (`<dir>/smoke/<cell>/`), so both
    # depths are searched and the same command works either way.
    candidates = sorted(
        {*results_dir.glob("*/meta.json"), *results_dir.glob("*/*/meta.json")}
    )
    for meta_path in candidates:
        meta = _read_meta(meta_path)
        if not meta or "runtime_sec" not in meta:
            continue
        run = meta_path.parent
        dataset = meta.get("dataset", run.name.partition("__")[0])
        model = meta.get("model", run.name.rpartition("__")[2])

        # `shape` is stored as [rows, columns] after sampling and row limiting.
        shape = meta.get("shape") or [None, None]
        row: dict[str, Any] = {
            "dataset": dataset,
            "rolle": dataset_role(dataset),
            "model": model,
            "n_rows": shape[0],
            "n_features": meta.get("n_features"),
            "depth": meta.get("depth"),
            "min_support": meta.get("min_support"),
            "n_selectors": meta.get("search_space_size"),
            "n_subgroups": meta.get("n_subgroups"),
            "global_auc_train": meta.get("global_train_score"),
            "global_auc_test": meta.get("global_test_score"),
            "n_interesting": None,
            "runtime_search_sec": meta.get("runtime_sd_sec"),
            "runtime_total_sec": meta.get("runtime_sec"),
            "algorithm": meta.get("algorithm"),
        }
        if row["n_subgroups"] and row["runtime_search_sec"]:
            # Size-independent cost: a big dataset is slower per candidate, a deep
            # search only has *more* candidates. Comparing totals conflates the two.
            row["ms_per_candidate"] = (
                1000.0 * row["runtime_search_sec"] / row["n_subgroups"]
            )

        csv_path = run / "subgroups.csv"
        if csv_path.is_file():
            frame = pd.read_csv(csv_path)
            # Row 0 is the whole-dataset reference row, not a subgroup.
            frame = frame.iloc[1:].copy()
            frame.insert(0, "dataset", dataset)
            frame.insert(1, "model", model)
            if "quality_raw" in frame:
                raw = pd.to_numeric(frame["quality_raw"], errors="coerce")
                row["max_quality_raw"] = raw.max()
                if "quality_ga" in frame:
                    ga = pd.to_numeric(frame["quality_ga"], errors="coerce")
                    thesis_interesting = (raw > 0) & (ga >= 0)
                    if "status" in frame:
                        ok = frame["status"].astype(str) == "ok"
                        thesis_interesting = thesis_interesting & ok
                    row["n_interesting"] = int(thesis_interesting.sum())
                    frame["interesting"] = thesis_interesting
                    row["n_q_raw_ge_002"] = int((raw >= 0.02).sum())
            subgroup_frames.append(frame)
        if row["n_interesting"] is None:
            row["n_interesting"] = meta.get("n_interesting")

        findings.append(row)

    order = {"candidate": 0, "baseline": 1, "control": 2, "illustration": 3}
    frame = pd.DataFrame(findings)
    if not frame.empty:
        frame = frame.sort_values(
            ["rolle", "dataset", "model"],
            key=lambda s: s.map(order) if s.name == "rolle" else s,
        )
    subgroups = (
        pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
    )
    return frame, subgroups


def runtime_table(findings: pd.DataFrame) -> pd.DataFrame:
    """Runtime per model, aggregated over datasets."""
    if findings.empty:
        return pd.DataFrame()
    return (
        findings.groupby("model")
        .agg(
            cells=("dataset", "size"),
            search_sec_sum=("runtime_search_sec", "sum"),
            search_sec_median=("runtime_search_sec", "median"),
            total_sec_sum=("runtime_total_sec", "sum"),
            ms_per_candidate_median=("ms_per_candidate", "median"),
        )
        .sort_values("total_sec_sum", ascending=False)
        .reset_index()
    )


def top_subgroups(
    subgroups: pd.DataFrame, n: int, sort_by: str = "quality_raw"
) -> pd.DataFrame:
    """
    The strongest subgroups overall, with logically redundant refinements collapsed.

    A conjunction can add a condition that excludes nothing — on `covertype`,
    `Soil_Type10==1 AND Soil_Type37==0` covers exactly the same rows as
    `Soil_Type10==1`, because the soil indicators are mutually exclusive. Those
    refinements are not wrong (generalization awareness correctly scores them at
    `quality_ga` 0), but there are dozens of them per real finding and they crowd out
    everything else. Rows agreeing on dataset, model, cover size and quality are
    therefore treated as one, keeping the shortest description.
    """
    if subgroups.empty or "quality_raw" not in subgroups:
        return pd.DataFrame()
    columns = [
        c
        for c in (
            "dataset", "model", "subgroup", "size_sg", "quality_raw", "quality",
            "quality_ga", "interesting",
        )
        if c in subgroups
    ]
    frame = subgroups[columns].copy()
    for column in ("quality_raw", "quality_ga", "quality"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["_len"] = frame["subgroup"].astype(str).str.len()
    keys = [c for c in ("dataset", "model", "size_sg") if c in frame]
    frame["_q"] = frame["quality_raw"].round(9)
    frame = (
        frame.sort_values("_len")
        .drop_duplicates(subset=[*keys, "_q"], keep="first")
        .drop(columns=["_len", "_q"])
    )
    key = sort_by if sort_by in frame else "quality_raw"
    return frame.nlargest(n, key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--sort-by",
        choices=("quality_raw", "quality_ga", "quality"),
        default="quality_raw",
        help="Ranking for the top table. quality_ga discounts a subgroup by its best "
        "generalization and so favours findings that are not just inherited from a "
        "larger region.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write findings.csv, runtimes.csv and all_subgroups.csv",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"{args.results_dir} existiert nicht", file=sys.stderr)
        return 1

    findings, subgroups = collect(args.results_dir)
    if findings.empty:
        print(f"Keine abgeschlossenen Läufe unter {args.results_dir}")
        return 1

    runtimes = runtime_table(findings)
    top = top_subgroups(subgroups, args.top, args.sort_by)
    fmt = lambda v: f"{v:9.4f}"  # noqa: E731

    show = [
        "dataset", "rolle", "model", "n_rows", "n_subgroups", "global_auc_test",
        "max_quality_raw", "n_interesting", "n_q_raw_ge_002",
        "runtime_search_sec", "runtime_total_sec",
    ]
    print("=== Befunde je Datensatz und Modell")
    print(findings[[c for c in show if c in findings]].to_string(index=False, float_format=fmt))

    print("\n=== Laufzeit je Modell (Sekunden)")
    print(runtimes.to_string(index=False, float_format=fmt))

    if not top.empty:
        print(f"\n=== Stärkste {len(top)} Subgruppen (sortiert nach {args.sort_by})")
        print(top.to_string(index=False, float_format=fmt))

    incomplete = findings["n_interesting"].isna().sum()
    if incomplete:
        print(
            f"\nHinweis: {incomplete} Zelle(n) ohne n_interesting "
            "(keine quality_raw/quality_ga in subgroups.csv)."
        )

    if args.csv:
        findings.to_csv(args.results_dir / "findings.csv", index=False)
        runtimes.to_csv(args.results_dir / "runtimes.csv", index=False)
        if not subgroups.empty:
            subgroups.to_csv(args.results_dir / "all_subgroups.csv", index=False)
        print(f"\nGeschrieben: findings.csv, runtimes.csv, all_subgroups.csv in {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
