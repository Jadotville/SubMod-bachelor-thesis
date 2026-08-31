"""
Fetch every curated dataset once into ``data/<name>.csv``.

The runners only ever read local CSVs (``datasets._load_local_csv``), so this
script is the single place that touches the network. Adult, Bank Marketing,
Phishing and Credit-Card Default keep the published label; the loaders map it
to 0/1. The other writers write a binary ``target`` plus the feature columns.

Two conventions matter for the results to stay comparable:

* **Row cap.** Large sources are subsampled to ``--max-rows`` with a fixed seed.
  The search cost grows linearly in rows and the datasets here run from 100k to
  1.6M, while a 2 % ``min_support`` on 100k rows already gives subgroups of 2000 —
  far more than a local fit needs. Capping buys nothing statistically but saves
  hours.
* **No identifiers.** Columns that merely index a row or an entity are dropped.
  They carry no semantics, so selectors built on them are unreadable, and a
  booster will happily memorise them (this is exactly why
  ``Amazon_employee_access`` was retired: all nine of its columns are IDs and the
  linear and neural models landed at AUC 0.51).

    python download_data.py                    # the thirteen datasets of chapter 4
    python download_data.py --only Diabetes130US --force

OpenML-Cache und ACS-Rohdaten liegen unter ``data/.openml_cache`` und
``data/acs_raw`` (nicht im Repo). ACS braucht das Paket ``folktables``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_HOME = DATA_DIR / ".openml_cache"

DEFAULT_MAX_ROWS = 100_000
SEED = 42


def _openml(data_id: int) -> pd.DataFrame:
    bundle = fetch_openml(
        data_id=data_id, as_frame=True, parser="auto", data_home=str(DATA_HOME)
    )
    frame = bundle.data.copy()
    frame["__target_raw__"] = bundle.target.to_numpy()
    return frame


def _drop(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.drop(columns=[c for c in columns if c in frame.columns])


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def build_diabetes130us() -> pd.DataFrame:
    """
    Readmission within 30 days, 101 766 hospital encounters.

    The published label has three levels (``NO``, ``<30``, ``>30``). Only the
    ``<30`` split is a real task — it is the one hospitals are measured on and the
    one that is hard (AUC around 0.68), which is precisely the headroom the method
    needs. Collapsing ``>30`` into the negative class is the standard treatment.
    """
    frame = _openml(45069)
    raw = frame.pop("__target_raw__").astype(str)
    frame = _drop(frame, ["encounter_id", "patient_nbr", "readmitted"])
    # `weight` is missing for ~97 % of encounters; `payer_code` is administrative.
    frame = _drop(frame, ["weight", "payer_code"])
    frame["target"] = (raw == "<30").astype(int)
    return frame


def build_covertype() -> pd.DataFrame:
    """
    Forest cover type, reduced to the two dominant species.

    Cover type is driven by elevation and soil, but *differently* per wilderness
    area — a documented geographic heterogeneity, and the reason this dataset is
    interesting here rather than just large. Classes 1 (Spruce/Fir) and 2
    (Lodgepole Pine) hold about 85 % of the rows and are the pair that actually
    competes; keeping all seven classes would force an arbitrary one-vs-rest cut.
    """
    frame = _openml(1596)
    raw = frame.pop("__target_raw__").astype(str)
    keep = raw.isin(["1", "2"])
    frame = frame.loc[keep].reset_index(drop=True)
    frame["target"] = (raw.loc[keep] == "2").astype(int).to_numpy()
    return frame


def build_road_safety() -> pd.DataFrame:
    """
    UK road accidents, 363 243 casualty records: was the accident serious or fatal?

    The published OpenML target is `Sex_of_Driver`, which is not a task anyone cares
    about. `Accident_Severity` is, and it carries the property this thesis needs:
    what makes an accident severe depends strongly on *where* it happened — speed
    limit, road type, urban against rural, police force area — so the feature-label
    relation differs by region. Severity levels 1 (fatal) and 2 (serious) are pooled
    against 3 (slight), the standard split, giving a 16 % positive rate.
    """
    frame = _openml(42803)
    frame.pop("__target_raw__")
    severity = pd.to_numeric(frame.pop("Accident_Severity"), errors="coerce")

    frame = _drop(
        frame,
        [
            # `Casualty_Severity` is the same information recorded per casualty — a
            # direct leak that would put the global AUC near 1 and hide everything.
            "Casualty_Severity",
            # Identifiers.
            "Accident_Index",
            "Casualty_Reference",
            "Vehicle_Reference_df",
            "Vehicle_Reference_df_res",
            # Free-form or very high cardinality: one-hot would add hundreds of
            # near-empty columns and the selectors would be unreadable. Coarse
            # geography stays in via Police_Force, Urban_or_Rural_Area and the
            # easting/northing coordinates.
            "LSOA_of_Accident_Location",
            "Local_Authority_(District)",
            "Local_Authority_(Highway)",
            "Date",
            "Time",
            "1st_Road_Number",
            "2nd_Road_Number",
        ],
    )
    keep = severity.isin([1.0, 2.0, 3.0])
    frame = frame.loc[keep].reset_index(drop=True)
    frame["target"] = severity.loc[keep].isin([1.0, 2.0]).astype(int).to_numpy()
    return frame


def build_electricity() -> pd.DataFrame:
    """
    Australian electricity market, 45 312 half-hour records: does the price rise?

    Included for a mechanism the other datasets do not have: this is the standard
    concept-drift benchmark, where the price-demand relation shifts over the two
    years covered. Drift is heterogeneity along the time axis, so `date` and
    `period` should be exactly the descriptors a local model profits from.
    """
    return _openml_binary(151, "UP")


def _acs(task_name: str) -> pd.DataFrame:
    """
    American Community Survey via ``folktables`` (Ding et al. 2021).

    This is the reference benchmark for subpopulation shift in tabular data: the
    feature-label relation genuinely differs across states and demographics, which
    is the mechanism this thesis looks for. All 50 states are pulled so that the
    heterogeneity is present at all; the row cap is applied afterwards.
    """
    from folktables import ACSDataSource
    import folktables

    task = getattr(folktables, task_name)
    source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=str(DATA_DIR / "acs_raw")
    )
    data = source.get_data(download=True)
    features, labels, _ = task.df_to_numpy(data)
    frame = pd.DataFrame(features, columns=task.features)
    frame["target"] = pd.Series(labels).astype(int).to_numpy()
    # ACS codes categoricals as floats; make the discrete ones readable so that
    # selectors come out as equalities rather than interval cuts.
    for column in frame.columns:
        if column == "target":
            continue
        if frame[column].nunique() <= 30:
            frame[column] = frame[column].astype("Int64").astype(str)
    return frame


def build_acs_income() -> pd.DataFrame:
    """Income above $50k — the modern replacement for `adult`."""
    return _acs("ACSIncome")


def build_acs_public_coverage() -> pd.DataFrame:
    """
    Public health-insurance coverage among low-income, under-65 respondents.

    Harder than ACSIncome (AUC around 0.79 instead of 0.83) because eligibility
    rules differ by state, which is exactly the kind of region-specific structure a
    single global model has to average over.
    """
    return _acs("ACSPublicCoverage")


def build_acs_mobility() -> pd.DataFrame:
    """
    Did the respondent move address within the last year?

    The hardest of the ACS tasks (AUC around 0.72-0.75): migration depends on local
    housing and labour markets, so much of what drives it is not in the person-level
    features at all — which is precisely the situation where regional structure
    should be worth more to a local model than to a global one.
    """
    return _acs("ACSMobility")


def build_acs_travel_time() -> pd.DataFrame:
    """
    Is the commute longer than 20 minutes?

    Commuting time is dominated by geography and urban form, and the raw ACS
    features describe the person rather than the place. Same state-level
    heterogeneity as the other ACS tasks, at a comparable difficulty to ACSMobility.
    """
    return _acs("ACSTravelTime")


def _openml_binary(data_id: int, positive: object | None = None) -> pd.DataFrame:
    """A source that needs nothing beyond mapping its label to 0/1."""
    frame = _openml(data_id)
    raw = frame.pop("__target_raw__").astype(str)
    if positive is None:
        classes = sorted(raw.unique())
        if len(classes) != 2:
            raise ValueError(f"data_id {data_id}: target is not binary ({classes})")
        positive = classes[1]
    frame["target"] = (raw == str(positive)).astype(int)
    return frame


def _openml_source_target(data_id: int) -> pd.DataFrame:
    """Keep the published label; ``datasets.py`` maps it to 0/1 at load time."""
    frame = _openml(data_id)
    frame["target"] = frame.pop("__target_raw__")
    return frame


def build_adult() -> pd.DataFrame:
    return _openml_source_target(1590)


def build_bank_marketing() -> pd.DataFrame:
    return _openml_source_target(1461)


def build_default_credit() -> pd.DataFrame:
    return _openml_source_target(42477)


def build_phishing_websites() -> pd.DataFrame:
    return _openml_source_target(4534)


# Chapter 4: five candidates, six baselines, two controls. Not MagicTelescope.
WRITERS: dict[str, Callable[[], pd.DataFrame]] = {
    "covertype": build_covertype,
    "electricity": build_electricity,
    "Diabetes130US": build_diabetes130us,
    "road-safety": build_road_safety,
    "ACSPublicCoverage": build_acs_public_coverage,
    "ACSIncome": build_acs_income,
    "ACSMobility": build_acs_mobility,
    "ACSTravelTime": build_acs_travel_time,
    "adult": build_adult,
    "bank-marketing": build_bank_marketing,
    "default-of-credit-card-clients": build_default_credit,
    "PhishingWebsites": build_phishing_websites,
    "mushroom": lambda: _openml_binary(24, "e"),
}


def write(name: str, max_rows: int, force: bool) -> dict[str, object]:
    path = DATA_DIR / f"{name}.csv"
    if path.is_file() and not force:
        existing = pd.read_csv(path, nrows=1)
        return {
            "dataset": name,
            "status": "vorhanden",
            "n_rows": sum(1 for _ in open(path)) - 1,
            "n_columns": existing.shape[1],
        }

    started = time.perf_counter()
    frame = WRITERS[name]()
    n_full = len(frame)
    if max_rows and n_full > max_rows:
        frame = frame.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
    if "target" not in frame.columns:
        raise ValueError(f"{name}: writer produced no 'target' column")
    if frame["target"].nunique() != 2:
        raise ValueError(f"{name}: target is not binary")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    numeric = pd.to_numeric(frame["target"], errors="coerce")
    positive_rate = (
        round(float(numeric.mean()), 4) if numeric.notna().all() else None
    )
    return {
        "dataset": name,
        "status": "geschrieben",
        "n_rows_source": n_full,
        "n_rows": len(frame),
        "n_columns": frame.shape[1] - 1,
        "positive_rate": positive_rate,
        "sec": round(time.perf_counter() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None, choices=list(WRITERS))
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = []
    for name in args.only or list(WRITERS):
        print(f"--- {name}", flush=True)
        try:
            rows.append(write(name, args.max_rows, args.force))
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
            print(f"    FEHLER {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            rows.append({"dataset": name, "status": f"Fehler: {type(exc).__name__}"})
        print(f"    {rows[-1]}", flush=True)

    print("\n=== Ergebnis")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
