"""
Redundancy in the top-15: how often a high-ranked description is a strict
refinement of another entry in the same list.

Chapter 5 ranks by Q_gew (column ``quality``), then asks how many of those
fifteen descriptions are a proper conjunct-superset of another. The same count
after keeping only rows with Q_ga ≥ 0 measures what the sign filter actually
removes. No new search.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SUBGROUPSIZE = HERE / "subgroupsize"
if str(SUBGROUPSIZE) not in sys.path:
    sys.path.insert(0, str(SUBGROUPSIZE))

import plot_quality_vs_size as qvs  # noqa: E402

TOP_K = 15
RANK_COL = "quality"


def conjuncts(description: str) -> frozenset[str]:
    text = str(description).strip()
    if not text or text == "Dataset":
        return frozenset()
    return frozenset(part.strip() for part in text.split(" AND ") if part.strip())


def n_strict_refinements(descriptions: list[str]) -> int:
    """How many entries are a strict refinement of some other entry."""
    sets = [conjuncts(d) for d in descriptions]
    n = 0
    for i, child in enumerate(sets):
        if not child:
            continue
        if any(parent < child for j, parent in enumerate(sets) if j != i and parent):
            n += 1
    return n


def per_cell(frame: pd.DataFrame, *, top_k: int = TOP_K) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, model), part in frame.groupby(["dataset", "model"], sort=False):
        ranked = part.sort_values(RANK_COL, ascending=False)
        top = ranked.head(top_k)
        descs = top["subgroup"].astype(str).tolist()
        n_ref = n_strict_refinements(descs)

        kept = ranked
        if "quality_ga" in ranked.columns:
            ga = pd.to_numeric(ranked["quality_ga"], errors="coerce")
            kept = ranked.loc[ga >= 0].head(top_k)
        descs_ga = kept["subgroup"].astype(str).tolist()
        n_ref_ga = n_strict_refinements(descs_ga)

        rows.append(
            {
                "rolle": part["rolle"].iloc[0] if "rolle" in part else qvs.dataset_role(dataset),
                "dataset": dataset,
                "model": model,
                "n": len(part),
                "n_top": len(top),
                "n_refinements": n_ref,
                "n_top_ga": len(kept),
                "n_refinements_ga": n_ref_ga,
                "rank1": descs[0] if descs else "",
                "rank1_ga": descs_ga[0] if descs_ga else "",
            }
        )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], list(qvs.MODEL_ORDER), ordered=True)
    return out.sort_values(["rolle", "dataset", "model"]).reset_index(drop=True)


def covertype_lgbm_example(frame: pd.DataFrame, *, top_k: int = 8) -> pd.DataFrame:
    cell = frame[(frame["dataset"] == "covertype") & (frame["model"] == "lgbm")]
    if cell.empty:
        return pd.DataFrame()
    cols = [c for c in ("subgroup", "size_sg", "quality_raw", "quality", "quality_ga") if c in cell]
    return cell.nlargest(top_k, RANK_COL)[cols].reset_index(drop=True)


def summarize(table: pd.DataFrame) -> dict:
    return {
        "n_cells": int(len(table)),
        "mean_refinements": float(table["n_refinements"].mean()),
        "cells_with_refinement": int((table["n_refinements"] > 0).sum()),
        "mean_refinements_ga": float(table["n_refinements_ga"].mean()),
        "cells_with_refinement_ga": int((table["n_refinements_ga"] > 0).sum()),
        "top_k": TOP_K,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=None)
    args = parser.parse_args()
    frame = qvs.load_subgroups(args.results_csv) if args.results_csv else qvs.load_subgroups()
    table = per_cell(frame)
    out_dir = HERE / "redundancy"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "top15_refinements.csv", index=False)
    stats = summarize(table)
    (out_dir / "summary.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    example = covertype_lgbm_example(frame)
    if not example.empty:
        example.to_csv(out_dir / "covertype_lgbm_top8.csv", index=False)

    print(
        f"Ohne Q_ga-Filter: im Mittel {stats['mean_refinements']:.1f} der {TOP_K} "
        f"nach Q_gew; {stats['cells_with_refinement']} von {stats['n_cells']} Zellen "
        "mit mindestens einer Verfeinerung."
    )
    print(
        f"Mit Q_ga ≥ 0, erneut Top-{TOP_K}: im Mittel {stats['mean_refinements_ga']:.1f}; "
        f"{stats['cells_with_refinement_ga']} von {stats['n_cells']} Zellen."
    )
    if not example.empty:
        print("\nCovertype / LightGBM, Top-8 nach Q_gew:")
        print(example.to_string(index=False))
    print(f"\ngeschrieben: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
