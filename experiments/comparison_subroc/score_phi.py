"""
SubROC-φ on the candidates of the own Local-Retraining-Gain search.

This is the comparison in the thesis (chapters 4 and 5): φ is computed from
quantities the adaptability run already stores. It is not a second SubROC
search. Offline cells come from ``results/offline/``; TabPFN cells from
``results/tabpfn/`` (train/test cap 1024). Missing directories are skipped.

    φ(S) = AUC_{D_test}(f_train) − AUC_{S_test}(f_train)

The Dataset row of ``subgroups.csv`` holds the first term; each subgroup row
holds the second as ``global_test_score``. Weighted φ uses |S_test|^α with
β = 0, matching the protocol.

    python score_phi.py
    python score_phi.py --adapt-results-dir ../experiments_real_world_data/results/offline
    python score_phi.py --adapt-results-dir ../experiments_real_world_data/results/tabpfn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RW_DIR = ROOT.parent / "experiments_real_world_data"
DEFAULT_ADAPT_DIRS = (
    RW_DIR / "results" / "offline",
    RW_DIR / "results" / "tabpfn",
)
DEFAULT_OUT = ROOT / "results" / "phi_on_adapt"

DEFAULT_SIZE_WEIGHT = 0.7
TOP_K = 50
# Global AUC = 1; both qualities sit at 0 and a Spearman is not a rank relation.
EXCLUDE_FROM_POOL = {
    ("mushroom", "lgbm"),
    ("mushroom", "rf"),
    ("mushroom", "tabpfn"),
}


def subgroups_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subgroup"] = out["subgroup"].astype(str)
    return out[out["subgroup"].ne("Dataset") & out["subgroup"].ne("")].copy()


def add_phi(
    adapt: pd.DataFrame,
    *,
    auc_dataset: float | None = None,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
) -> pd.DataFrame:
    """Attach φ and φ_gew to adaptability subgroup rows."""
    if adapt.empty:
        return adapt
    dataset_rows = adapt[adapt["subgroup"].astype(str).eq("Dataset")]
    if auc_dataset is None:
        if dataset_rows.empty or "global_test_score" not in dataset_rows:
            raise ValueError("need Dataset row with global_test_score or auc_dataset")
        auc_dataset = float(dataset_rows["global_test_score"].iloc[0])

    sg = subgroups_only(adapt)
    if sg.empty:
        return sg
    sg = sg.copy()
    sg["auc_dataset"] = auc_dataset
    sg["phi"] = auc_dataset - pd.to_numeric(sg["global_test_score"], errors="coerce")
    size_test = pd.to_numeric(sg["size_sg_test"], errors="coerce").clip(lower=0)
    sg["phi_gew"] = sg["phi"] * np.power(size_test, size_weight)
    return sg


def cell_metrics(sg: pd.DataFrame, top_k: int = TOP_K) -> dict[str, float]:
    """Spearman, top-k overlap, median-split 2×2 — as in chapter 4."""
    q = pd.to_numeric(sg["quality"], errors="coerce")
    phi_w = pd.to_numeric(sg["phi_gew"], errors="coerce")
    raw = pd.to_numeric(sg["quality_raw"], errors="coerce") if "quality_raw" in sg else q
    phi = pd.to_numeric(sg["phi"], errors="coerce")
    mask = q.notna() & phi_w.notna()
    q, phi_w = q[mask], phi_w[mask]
    raw, phi = raw[mask], phi[mask]
    n = int(len(q))
    if n < 3:
        return {"n": n}
    if phi_w.nunique() < 2 or q.nunique() < 2:
        spearman = float("nan")
    else:
        spearman = float(q.corr(phi_w, method="spearman"))
    if phi.nunique() < 2 or raw.nunique() < 2:
        spearman_raw = float("nan")
    else:
        spearman_raw = float(raw.corr(phi, method="spearman"))
    k = min(top_k, n)
    top_q = set(sg.loc[q.nlargest(k).index, "subgroup"].astype(str))
    top_phi = set(sg.loc[phi_w.nlargest(k).index, "subgroup"].astype(str))
    overlap = len(top_q & top_phi) / k if k else float("nan")

    q_hi = q >= q.median()
    p_hi = phi_w >= phi_w.median()
    return {
        "n": n,
        "spearman_quality_phi_gew": spearman,
        "spearman_raw": spearman_raw,
        "top50_overlap": overlap,
        "frac_both_high": float((q_hi & p_hi).mean()),
        "frac_only_subroc": float((~q_hi & p_hi).mean()),
        "frac_only_lrg": float((q_hi & ~p_hi).mean()),
        "frac_both_low": float((~q_hi & ~p_hi).mean()),
    }


def discover_adapt_cells(adapt_dirs: list[Path]) -> list[tuple[str, str, Path]]:
    cells: list[tuple[str, str, Path]] = []
    seen: set[tuple[str, str]] = set()
    for adapt_dir in adapt_dirs:
        if not adapt_dir.is_dir():
            continue
        for meta_path in sorted(adapt_dir.glob("*/meta.json")):
            name = meta_path.parent.name
            if "__" not in name:
                continue
            dataset, model = name.split("__", 1)
            key = (dataset, model)
            if key in seen:
                print(f"skip duplicate {dataset}__{model} in {adapt_dir}", file=sys.stderr)
                continue
            seen.add(key)
            cells.append((dataset, model, meta_path.parent))
    return cells


def _in_pool(summary: pd.DataFrame) -> pd.DataFrame:
    keys = list(zip(summary["dataset"], summary["model"]))
    return summary.loc[[k not in EXCLUDE_FROM_POOL for k in keys]].copy()


def print_pool_report(keep: pd.DataFrame, title: str) -> None:
    if keep.empty:
        return
    rho = pd.to_numeric(keep["spearman_quality_phi_gew"], errors="coerce")
    rho_raw = pd.to_numeric(keep["spearman_raw"], errors="coerce")
    print(f"\n=== {title}")
    print(f"  Zellen          : {len(keep)}")
    print(f"  Median ρ        : {rho.median():.3f}   (Mittel {rho.mean():.3f})")
    print(f"  |ρ| < 0,3       : {(rho.abs() < 0.3).mean():.0%}")
    print(f"  ρ < 0           : {(rho < 0).mean():.0%}")
    print(f"  Top-50-Überlapp.: {keep['top50_overlap'].mean():.1%}")
    print(f"  Median ρ roh    : {rho_raw.median():.3f}")
    print(
        f"  beide hoch      : {keep['frac_both_high'].mean():.1%}   "
        f"beide niedrig {keep['frac_both_low'].mean():.1%}"
    )
    print(
        f"  nur SubROC      : {keep['frac_only_subroc'].mean():.1%}   "
        f"nur LRG {keep['frac_only_lrg'].mean():.1%}"
    )
    print("  Median ρ je Modell:")
    for model, part in keep.groupby("model", sort=False):
        print(f"    {model:6s}  {part['spearman_quality_phi_gew'].median():+.3f}")


def print_thesis_report(summary: pd.DataFrame) -> None:
    """Offline 50-cell pool plus TabPFN (capped protocol), mushroom trees/TabPFN dropped."""
    if summary.empty:
        return
    keep = _in_pool(summary)
    offline = keep[keep["model"].ne("tabpfn")].copy()
    tabpfn = keep[keep["model"].eq("tabpfn")].copy()
    print_pool_report(
        offline,
        "Kapitel 5, Offline (50 Zellen, ohne Mushroom/lgbm und Mushroom/rf)",
    )
    print_pool_report(
        tabpfn,
        "TabPFN (ohne Mushroom; Train/Test je 1024, ein Estimator)",
    )
    if not tabpfn.empty:
        print_pool_report(
            keep,
            "Offline + TabPFN (Mushroom/lgbm, Mushroom/rf, Mushroom/tabpfn ausgelassen)",
        )


def process_all(
    adapt_dirs: list[Path],
    out_dir: Path,
    *,
    size_weight: float = DEFAULT_SIZE_WEIGHT,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset, model, cell in discover_adapt_cells(adapt_dirs):
        csv_path = cell / "subgroups.csv"
        meta_path = cell / "meta.json"
        if not csv_path.is_file():
            continue
        adapt = pd.read_csv(csv_path)
        auc = None
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            auc = meta.get("global_test_score")
        sg = add_phi(adapt, auc_dataset=auc, size_weight=size_weight)
        cell_out = out_dir / f"{dataset}__{model}"
        cell_out.mkdir(parents=True, exist_ok=True)
        sg.to_csv(cell_out / "phi.csv", index=False, encoding="utf-8")
        metrics = cell_metrics(sg)
        metrics.update({"dataset": dataset, "model": model})
        rows.append(metrics)
        print(
            f"{dataset}__{model}: n={metrics.get('n')} "
            f"ρ={metrics.get('spearman_quality_phi_gew', float('nan')):.3f} "
            f"top50={metrics.get('top50_overlap', float('nan')):.3f}"
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Wrote {summary_path}")
    print_thesis_report(summary)
    return {"n_cells": len(rows), "summary": str(summary_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapt-results-dir",
        type=Path,
        nargs="*",
        default=None,
        help="Search result directories (default: results/offline and results/tabpfn).",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--size-weight", type=float, default=DEFAULT_SIZE_WEIGHT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adapt_dirs = list(args.adapt_results_dir or DEFAULT_ADAPT_DIRS)
    missing = [str(d) for d in adapt_dirs if not d.is_dir()]
    present = [d for d in adapt_dirs if d.is_dir()]
    for path in missing:
        print(f"Adapt results dir missing (skipped): {path}", file=sys.stderr)
    if not present:
        print("No adapt results directories found.", file=sys.stderr)
        return 1
    process_all(
        present,
        args.out_dir,
        size_weight=args.size_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
