"""
Rebuild the chapter-5 characterization from the finished offline batch.

Needs ``experiments_real_world_data/results/offline/all_subgroups.csv`` and
``findings.csv`` (written by ``summarize_runs.py --csv``). Reweights already
scored candidates; it does not search again. TabPFN is not part of this sweep.

    python run.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUBGROUPSIZE = HERE / "subgroupsize"
if str(SUBGROUPSIZE) not in sys.path:
    sys.path.insert(0, str(SUBGROUPSIZE))
if str(HERE / "alpha_sweep") not in sys.path:
    sys.path.insert(0, str(HERE / "alpha_sweep"))


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import plot_alpha_sweep as alpha_mod  # noqa: E402
import plot_quality_vs_size as qvs  # noqa: E402
import redundancy  # noqa: E402

beta0_mod = _load("beta_sweep_alpha0", HERE / "beta_sweep" / "plot_beta_sweep.py")
beta07_mod = _load(
    "beta_sweep_alpha07", HERE / "beta_sweep_alpha_0p7" / "plot_beta_sweep.py"
)

POLICY_ALPHA = 0.7
POLICY_BETA = 0.0
PLATEAU = (0.6, 0.8)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def thesis_block(frame: pd.DataFrame) -> dict:
    """Numbers that chapter 5 quotes, so a later re-run can be checked."""
    n = int(len(frame))
    n_pos = int((frame["quality_raw"] > 0).sum())
    n_zero = int((frame["quality_raw"] == 0).sum())
    n_neg = int((frame["quality_raw"] < 0).sum())
    n_cells = int(frame.groupby(["dataset", "model"]).ngroups)
    n_datasets = int(frame["dataset"].nunique())
    share_big = float(
        frame["dataset"].isin(["road-safety", "covertype"]).mean()
    )
    covertype_n = int((frame["dataset"] == "covertype").sum())
    covertype_per_model = (
        frame[frame["dataset"] == "covertype"]
        .groupby("model")
        .size()
        .to_dict()
    )

    alpha_table = alpha_mod.sweep_spearman(frame)
    at0 = alpha_table[alpha_table["alpha"] == 0.0].set_index("model")["spearman"]
    at07 = alpha_table[np.isclose(alpha_table["alpha"], POLICY_ALPHA)].set_index(
        "model"
    )["spearman"]
    plateau = alpha_table[alpha_table["alpha"].between(*PLATEAU, inclusive="both")]
    plateau_max_abs = float(plateau.groupby("alpha")["spearman"].apply(lambda s: s.abs().max()).max())
    at07_max_abs = float(at07.abs().max())
    crossings = alpha_mod.zero_crossings(alpha_table)

    beta0 = beta0_mod.sweep_spearman(frame)
    beta0_at0 = beta0[beta0["beta"] == 0.0].set_index("model")["spearman"]
    beta07 = beta07_mod.sweep_spearman(frame)
    beta07_at0 = beta07[beta07["beta"] == 0.0].set_index("model")["spearman"]
    extra_025 = beta07_mod.sweep_spearman(frame, betas=(0.25,))
    beta07_at025 = extra_025.set_index("model")["spearman"]
    mean_abs_0 = float(beta07_at0.abs().mean())
    mean_abs_025 = float(beta07_at025.abs().mean())
    max_abs_0 = float(beta07_at0.abs().max())
    at01 = beta07[np.isclose(beta07["beta"], 0.1)].set_index("model")["spearman"]
    max_abs_01 = float(at01.abs().max())

    red = redundancy.summarize(redundancy.per_cell(frame))

    return {
        "n_candidates": n,
        "n_q_raw_positive": n_pos,
        "n_q_raw_zero": n_zero,
        "n_q_raw_negative": n_neg,
        "n_datasets": n_datasets,
        "n_cells": n_cells,
        "share_road_safety_covertype": share_big,
        "covertype_n": covertype_n,
        "covertype_n_per_model": {str(k): int(v) for k, v in covertype_per_model.items()},
        "spearman_unweighted": {m: float(at0[m]) for m in qvs.PLOT_MODELS if m in at0},
        "spearman_alpha_0p7": {m: float(at07[m]) for m in qvs.PLOT_MODELS if m in at07},
        "plateau_0p6_0p8_max_abs_rho": plateau_max_abs,
        "alpha_0p7_max_abs_rho": at07_max_abs,
        "zero_crossings": crossings.to_dict(orient="records"),
        "spearman_balance_no_size": {
            m: float(beta0_at0[m]) for m in qvs.PLOT_MODELS if m in beta0_at0
        },
        "spearman_balance_alpha_0p7": {
            m: float(beta07_at0[m]) for m in qvs.PLOT_MODELS if m in beta07_at0
        },
        "mean_abs_rho_beta_0": mean_abs_0,
        "mean_abs_rho_beta_0p25": mean_abs_025,
        "max_abs_rho_beta_0": max_abs_0,
        "max_abs_rho_beta_0p1": max_abs_01,
        "redundancy": red,
        "source": str(qvs.DEFAULT_RESULTS),
    }


def main() -> int:
    frame = qvs.load_subgroups()
    print(f"Quelle: {qvs.DEFAULT_RESULTS}")
    print(
        f"{len(frame):,} Kandidaten, "
        f"{int((frame.quality_raw > 0).sum()):,} mit Q_roh>0, "
        f"{frame.dataset.nunique()} Datensätze × "
        f"{frame.model.nunique()} Modelle"
    )

    written = qvs.write_plots(frame, SUBGROUPSIZE)
    print("Subgruppengröße:")
    for key, path in written.items():
        print(f"  {key}: {path}")

    print("\n=== α-Sweep")
    alpha_mod.main()
    print("\n=== β-Sweep (α = 0)")
    beta0_mod.main()
    print("\n=== β-Sweep (α = 0,7)")
    beta07_mod.main()
    print("\n=== Redundanz Top-15")
    redundancy.main()

    block = thesis_block(frame)
    out = HERE / "thesis_numbers.json"
    out.write_text(json.dumps(block, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nKapitel-5-Zahlen: {out}")
    print(
        f"  Kandidaten {block['n_candidates']:,}  "
        f"Q_roh>0 {block['n_q_raw_positive']:,}  "
        f"Zellen {block['n_cells']}"
    )
    print("  Spearman α=0:", {k: _fmt(v) for k, v in block["spearman_unweighted"].items()})
    print("  Spearman α=0,7:", {k: _fmt(v) for k, v in block["spearman_alpha_0p7"].items()})
    print(
        f"  Plateau max |ρ| {block['plateau_0p6_0p8_max_abs_rho']:.2f}  "
        f"bei α=0,7 {block['alpha_0p7_max_abs_rho']:.2f}"
    )
    print(
        "  Balance ohne Größe:",
        {k: _fmt(v) for k, v in block["spearman_balance_no_size"].items()},
    )
    print(
        "  Balance α=0,7 β=0:",
        {k: _fmt(v) for k, v in block["spearman_balance_alpha_0p7"].items()},
    )
    print(
        f"  mittlerer |ρ| β=0 {block['mean_abs_rho_beta_0']:.3f}  "
        f"β=0,25 {block['mean_abs_rho_beta_0p25']:.3f}"
    )
    red = block["redundancy"]
    print(
        f"  Top-15 Verfeinerungen {red['mean_refinements']:.1f} → "
        f"{red['mean_refinements_ga']:.1f} mit Q_ga≥0 "
        f"({red['cells_with_refinement']} → {red['cells_with_refinement_ga']} Zellen)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
