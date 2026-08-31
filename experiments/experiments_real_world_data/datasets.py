"""Load real-world datasets from local CSVs (preferred) or OpenML fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_HOME = DATA_DIR / ".openml_cache"


@dataclass(frozen=True)
class DatasetSpec:
    """One curated real-world dataset."""

    name: str
    openml_id: int
    sample_frac: float = 1.0
    # If set, binary map (str(y) == str(positive)) -> 1; else stable 2-class map.
    label_positive: object | None = None


def _fetch(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    bun = fetch_openml(
        data_id=data_id,
        as_frame=True,
        parser="auto",
        data_home=str(DATA_HOME),
    )
    return bun.data.copy(), bun.target.copy()


def _to_binary_frame(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    positive: object | None = None,
) -> pd.DataFrame:
    df = X.copy()
    y = y.copy()
    if positive is not None:
        df["target"] = (y.astype(str) == str(positive)).astype(int)
    else:
        classes = sorted(y.astype(str).unique())
        if len(classes) != 2:
            raise ValueError(f"Expected binary target, got {classes}")
        mapping = {classes[0]: 0, classes[1]: 1}
        df["target"] = y.astype(str).map(mapping).astype(int)
    df["prediction"] = 0
    return df.reset_index(drop=True)


def _load_local_csv(name: str) -> tuple[pd.DataFrame, pd.Series] | None:
    path = DATA_DIR / f"{name}.csv"
    if not path.is_file():
        return None
    raw = pd.read_csv(path)
    if "target" not in raw.columns:
        raise ValueError(f"{path} missing 'target' column")
    y = raw["target"]
    X = raw.drop(columns=["target"])
    if "prediction" in X.columns:
        X = X.drop(columns=["prediction"])
    return X, y


def _load_xy(name: str, openml_id: int) -> tuple[pd.DataFrame, pd.Series]:
    local = _load_local_csv(name)
    if local is not None:
        return local
    return _fetch(openml_id)


def load_adult() -> pd.DataFrame:
    X, y = _load_xy("adult", 1590)
    return _to_binary_frame(X, y, positive=">50K")


def load_bank_marketing() -> pd.DataFrame:
    X, y = _load_xy("bank-marketing", 1461)
    # OpenML/local CSV: class "2" = subscribed
    return _to_binary_frame(X, y, positive="2")


def load_default_credit() -> pd.DataFrame:
    X, y = _load_xy("default-of-credit-card-clients", 42477)
    return _to_binary_frame(X, y)  # already 0/1


def load_phishing_websites() -> pd.DataFrame:
    X, y = _load_xy("PhishingWebsites", 4534)
    # UCI: 1 = phishing, -1 = legitimate
    return _to_binary_frame(X, y, positive="1")


def _load_prepared(name: str) -> pd.DataFrame:
    """
    Load a CSV that ``download_data.py`` already put into its final shape.

    These sources need per-dataset work before they are usable — collapsing a
    three-level readmission label, restricting cover type to the two competing
    species, pulling ACS from the Census API — so that work lives in
    ``download_data.py`` and is done once, not on every run.
    """
    path = DATA_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Run: python download_data.py --only {name}"
        )
    df = pd.read_csv(path, low_memory=False)
    if "target" not in df.columns:
        raise ValueError(f"{path} missing 'target' column")
    if "prediction" not in df.columns:
        df["prediction"] = 0
    return df.reset_index(drop=True)


DATASET_SPECS: dict[str, DatasetSpec] = {
    # --- Candidates with headroom: hard tasks and documented heterogeneity ------
    "Diabetes130US": DatasetSpec("Diabetes130US", 45069),
    "ACSIncome": DatasetSpec("ACSIncome", -1),
    "ACSPublicCoverage": DatasetSpec("ACSPublicCoverage", -1),
    "ACSMobility": DatasetSpec("ACSMobility", -1),
    "ACSTravelTime": DatasetSpec("ACSTravelTime", -1),
    "covertype": DatasetSpec("covertype", 1596),
    "road-safety": DatasetSpec("road-safety", 42803),
    "electricity": DatasetSpec("electricity", 151),
    # --- Established baselines -------------------------------------------------
    "adult": DatasetSpec("adult", 1590, label_positive=">50K"),
    "bank-marketing": DatasetSpec("bank-marketing", 1461, label_positive="2"),
    "default-of-credit-card-clients": DatasetSpec(
        "default-of-credit-card-clients", 42477
    ),
    # --- Negative controls -----------------------------------------------------
    "PhishingWebsites": DatasetSpec("PhishingWebsites", 4534, label_positive="1"),
    "mushroom": DatasetSpec("mushroom", 24, label_positive="e"),
}

LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "Diabetes130US": lambda: _load_prepared("Diabetes130US"),
    "ACSIncome": lambda: _load_prepared("ACSIncome"),
    "ACSPublicCoverage": lambda: _load_prepared("ACSPublicCoverage"),
    "ACSMobility": lambda: _load_prepared("ACSMobility"),
    "ACSTravelTime": lambda: _load_prepared("ACSTravelTime"),
    "covertype": lambda: _load_prepared("covertype"),
    "road-safety": lambda: _load_prepared("road-safety"),
    "electricity": lambda: _load_prepared("electricity"),
    # The CSVs below predate download_data.py and still hold the source labels
    # (">50K", "yes", "-1"), so their loaders keep applying the positive-class map.
    "adult": load_adult,
    "bank-marketing": load_bank_marketing,
    "default-of-credit-card-clients": load_default_credit,
    "PhishingWebsites": load_phishing_websites,
    # These two are written by download_data.py with the label already binarised;
    # mapping again would compare "0"/"1" against "e" and silently produce a
    # constant target.
    "mushroom": lambda: _load_prepared("mushroom"),
}

# The roles below are not guesses: every dataset was screened at depth 1 with all
# four offline models. The numbers quoted are the raw test-AUC gap of the best
# single selector.

#: Datasets where a *strong* global model still leaves room, each carrying a
#: different source of heterogeneity:
#:
#: * `covertype` — **geographic**. Tree species depends on elevation and soil, but
#:   differently per wilderness area. The strongest case: at depth 2 all four models
#:   find something (LightGBM 2925 subgroups above the reference effect at a healthy
#:   AUC 0.920, forest 2035, logistic regression 1284, MLP only 54), which is also
#:   the model separation this thesis wants to show.
#: * `electricity` — **temporal**. The standard concept-drift benchmark: the
#:   price-demand relation shifts over the two years covered, so drift acts as
#:   heterogeneity along the time axis. Effects survive the strong learners
#:   (LightGBM 0.085 at AUC 0.945, forest 0.076).
#: * `Diabetes130US` — **only in conjunctions**. See the note below; this one is a
#:   methodological finding in its own right.
#: * `road-safety` — **geographic**, modest but present across models (0.041 for
#:   logistic regression, 0.023 forest, 0.022 MLP). Severity depends on speed limit,
#:   road type and urban against rural.
#: * `ACSPublicCoverage` — real, but confined to logistic regression (0.066 against
#:   0.007 for the forest), so it documents misspecification rather than
#:   heterogeneity that survives a strong learner.
CANDIDATE_DATASETS = [
    "covertype",
    "electricity",
    "Diabetes130US",
    "road-safety",
    "ACSPublicCoverage",
]

#: Moderate headroom, kept for comparability with the literature rather than because
#: a large effect is expected. The three ACS tasks land here because only logistic
#: regression finds anything on them (0.041-0.069, against roughly 0 for the forest
#: and LightGBM), which makes them evidence about the linear model, not about the
#: data.
BASELINE_DATASETS = [
    "ACSIncome",
    "ACSMobility",
    "ACSTravelTime",
    "adult",
    "bank-marketing",
    "default-of-credit-card-clients",
]

#: Datasets where the correct outcome is *nothing*, because the global model is
#: already at AUC 0.98-1.00 and nothing can be won anywhere. A method that stays
#: quiet here is behaving correctly.
CONTROL_DATASETS = [
    "PhishingWebsites",
    "mushroom",
]

# A correction worth keeping visible: `Diabetes130US` was first classified as a
# control on the strength of a depth-1 screen, where the best single selector
# reached only 0.0135. At depth 2 it turns into one of the clearer positive cases
# (forest: 197 subgroups beyond the null band, 103 above the reference effect,
# max 0.100). Its heterogeneity lives entirely in conjunctions, so a depth-1 screen
# cannot see it. Two consequences: screening decides suitability at the depth the
# real runs will use, and "no single selector works" is not evidence of homogeneity.

#: Retired, deliberately: `Amazon_employee_access` consists of nine identifier
#: columns, which left the linear and neural models at AUC 0.51 and produced
#: unreadable selectors; `churn` has only 5000 rows, so every subgroup sits near
#: `min_support` and the measured effects are dominated by variance.
RETIRED_DATASETS = ["Amazon_employee_access", "churn"]


# --------------------------------------------------------------------------- #
# Nominal columns stored as integers
# --------------------------------------------------------------------------- #
# Survey and registry data encodes nominal variables as numeric codes: in
# `road-safety`, `Weather_Conditions` 1 means fine, 2 rain, 7 fog. Their dtype is
# integer, so `preprocess._split_columns` files them as numeric and they get scaled
# instead of one-hot encoded — which makes the linear and neural models fit a linear
# function of an arbitrary code number. Measured on `road-safety`, that costs
# logistic regression 0.048 AUC and the MLP 0.035, while the tree models are
# unaffected (they recover nominal structure by splitting repeatedly).
#
# The columns have to be listed rather than detected, because a low-cardinality
# integer is just as often genuinely ordinal — `time_in_hospital` (1-14 days),
# `AGEP` (age), `education-num` (years of schooling) and the ACS payment-delay
# columns all carry real order that one-hot would throw away.
#
# Casting happens in `prepare_dataset`, not in `download_data.py`: a CSV has no dtype
# to preserve, so `astype(str)` before writing is undone by the next `read_csv`.

#: ACS columns with genuine order or a continuous scale. Everything else in an ACS
#: task is a nominal code from the Census codebook (MAR, CIT, RELP, RAC1P, ...), so
#: the allowlist is shorter and less error-prone than its complement.
_ACS_NUMERIC = frozenset(
    {"AGEP", "SCHL", "WKHP", "JWMNP", "PINCP", "POVPIP", "target", "prediction"}
)

NOMINAL_COLUMNS: dict[str, list[str]] = {
    "road-safety": [
        "Vehicle_Type", "Towing_and_Articulation", "Vehicle_Manoeuvre",
        "Vehicle_Location-Restricted_Lane", "Junction_Location",
        "Skidding_and_Overturning", "Hit_Object_in_Carriageway",
        "Vehicle_Leaving_Carriageway", "Hit_Object_off_Carriageway",
        "1st_Point_of_Impact", "Was_Vehicle_Left_Hand_Drive?",
        "Journey_Purpose_of_Driver", "Propulsion_Code", "Driver_Home_Area_Type",
        "Police_Force", "Day_of_Week", "1st_Road_Class", "Road_Type",
        "Junction_Detail", "Junction_Control", "2nd_Road_Class",
        "Pedestrian_Crossing-Human_Control",
        "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions",
        "Weather_Conditions", "Road_Surface_Conditions",
        "Special_Conditions_at_Site", "Carriageway_Hazards", "Urban_or_Rural_Area",
        "Did_Police_Officer_Attend_Scene_of_Accident", "Casualty_Class",
        "Sex_of_Casualty", "Pedestrian_Location", "Pedestrian_Movement",
        "Car_Passenger", "Bus_or_Coach_Passenger",
        "Pedestrian_Road_Maintenance_Worker", "Casualty_Type",
        "Casualty_Home_Area_Type",
    ],
    # Administrative routing codes. The counts around them (`time_in_hospital`,
    # `num_procedures`, `number_diagnoses`, ...) stay numeric.
    "Diabetes130US": [
        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id",
    ],
    # x2 sex, x3 education, x4 marital status. x6-x11 encode months of payment delay
    # and are ordinal, so they stay numeric.
    "default-of-credit-card-clients": ["x2", "x3", "x4"],
    # Day of week.
    "electricity": ["day"],
}


def nominal_columns(name: str, df: pd.DataFrame) -> list[str]:
    """Which columns of ``name`` should be treated as nominal rather than numeric."""
    if name.startswith("ACS"):
        return [c for c in df.columns if c not in _ACS_NUMERIC]
    return [c for c in NOMINAL_COLUMNS.get(name, []) if c in df.columns]


#: Explicit level for a missing nominal code. Registry data records "unknown" as a
#: code of its own (STATS19 uses -1), so an absent value carries information and is
#: better kept as a level than imputed away — it also becomes a usable selector.
#: A literal `None` would additionally break `ps.EqualitySelector`, which rejects it.
MISSING_LEVEL = "missing"


def _apply_nominal(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Cast declared nominal codes to strings so the encoder one-hots them."""
    columns = nominal_columns(name, df)
    if not columns:
        return df
    out = df.copy()
    for column in columns:
        series = out[column]
        if pd.api.types.is_numeric_dtype(series):
            # Via Int64 so that a code reads as "2" and not "2.0"; a float column
            # that happens to hold whole numbers is the normal case after read_csv.
            if series.dropna().mod(1).eq(0).all():
                series = series.astype("Int64")
        out[column] = (
            series.astype(str)
            .where(series.notna(), MISSING_LEVEL)
            .astype(str)
        )
    return out


def available_local_datasets() -> list[str]:
    """Datasets with a CSV in data/."""
    return [n for n in DATASET_SPECS if (DATA_DIR / f"{n}.csv").is_file()]


AVAILABLE_DATASETS = available_local_datasets()

#: Default working set for a full batch, in the order results should be read.
MAIN_DATASETS = [
    n
    for n in CANDIDATE_DATASETS + BASELINE_DATASETS + CONTROL_DATASETS
    if n in AVAILABLE_DATASETS
] or CANDIDATE_DATASETS

SMOKE_DATASETS = [n for n in ("PhishingWebsites", "adult") if n in AVAILABLE_DATASETS]


def dataset_role(name: str) -> str:
    """Which of the three roles a dataset plays in the evaluation."""
    if name in CANDIDATE_DATASETS:
        return "candidate"
    if name in BASELINE_DATASETS:
        return "baseline"
    if name in CONTROL_DATASETS:
        return "control"
    return "unknown"


def prepare_dataset(
    name: str,
    seed: int = 42,
    sample_frac: float | None = None,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, DatasetSpec]:
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[name]
    df = _apply_nominal(name, LOADERS[name]())
    frac = spec.sample_frac if sample_frac is None else sample_frac
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    if row_limit is not None and len(df) > row_limit:
        df = df.sample(n=row_limit, random_state=seed).reset_index(drop=True)
    if "prediction" not in df.columns:
        df["prediction"] = 0
    return df.reset_index(drop=True), spec


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"target", "prediction"}
    return [c for c in df.columns if c not in exclude]


def cap_frame(
    df: pd.DataFrame,
    n: int | None,
    seed: int,
    label: str = "target",
) -> pd.DataFrame:
    """Keep at most ``n`` rows, stratified on ``label`` when that split is valid."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    y = df[label]
    counts = y.value_counts()
    if y.nunique() >= 2 and int(counts.min()) >= 2 and n >= int(y.nunique()):
        try:
            kept, _ = train_test_split(
                df,
                train_size=n,
                random_state=seed,
                shuffle=True,
                stratify=y,
            )
            return kept.reset_index(drop=True)
        except ValueError:
            pass
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def stratified_split_fn(
    seed: int = 42,
    test_size: float = 0.5,
    train_cap: int | None = None,
    test_cap: int | None = None,
) -> Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Build a stratified 50/50 train/test split callable for the QF.

    ``train_cap`` / ``test_cap`` subsample each half after the split. Used for
    TabPFN so fit and predict stay inside a fixed context, not the full 50k.
    """

    def split_fn(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(
            data,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=data["target"],
        )
        train = cap_frame(train, train_cap, seed)
        test = cap_frame(test, test_cap, seed + 1)
        return train.reset_index(drop=True), test.reset_index(drop=True).copy()

    return split_fn
