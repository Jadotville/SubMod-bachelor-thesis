"""
Datensätze der synthetischen Experimente.

``run.py`` ruft nur ``make_a0`` … ``make_g1`` auf. Sweeps setzen dieselben
Keyword-Argumente (z. B. ``make_c1(beta0=…)``, ``make_c5(n_noise=60)``).
Hilfsfunktionen darunter implementieren die jeweiligen DGPs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DatasetSpec:
    experiment_id: str
    name: str
    df: pd.DataFrame
    label_column: str = "target"
    feature_columns: list[str] = field(default_factory=list)
    search_columns: list[str] = field(default_factory=list)
    ground_truth_subgroups: list[str] = field(default_factory=list)
    description: str = ""


# Ohne dieses Rauschen sitzen RF/MLP an der AUC-Decke; Q kann dann nicht
# positiv werden. Kalibriert in Experiment A1.
DEFAULT_LABEL_NOISE = 0.15


def _noise_columns(n: int, n_noise: int, rng: np.random.Generator) -> pd.DataFrame:
    arr = rng.normal(0, 1, (n, n_noise))
    return pd.DataFrame(
        arr, columns=[f"noise_{i}" for i in range(n_noise)]
    )


def _apply_label_noise(
    target: np.ndarray, label_noise: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Flip each label independently with probability ``label_noise``.

    Symmetric flipping is deliberate: it maps P(y=1|x) to a strictly monotone
    transform of itself, so the Bayes-optimal *ranking* is unchanged and the
    noise cannot by itself manufacture adaptability. It only lowers the
    achievable AUC and thereby removes the ceiling effect.
    """
    if not label_noise:
        return target
    if not 0.0 <= label_noise < 0.5:
        raise ValueError(f"label_noise must be in [0, 0.5), got {label_noise}")
    flip = rng.random(len(target)) < label_noise
    out = np.asarray(target).copy()
    out[flip] = ~out[flip].astype(bool) if out.dtype == bool else 1 - out[flip]
    return out


def _stratum_scores(b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """16 fest definierte Regeln, eine Zeile pro Stratum (C2, D1, D4, E3)."""
    s0 = (d + e) / 2.0
    s1 = np.where(
        b + e > 0,
        (c * c - d * d) / 2.0,
        (d * c - e * b) / 2.0,
    )
    s2 = np.where(
        b + e > c + d,
        (b + c - e) / 3.0,
        (b / 2.0) * e - (c * d) / 2.0,
    )
    s3 = np.where(
        b + e > c + d + e,
        (b + e) / 2.0,
        (c * e - b * d) / 2.0,
    )
    s4 = b - c
    s5 = b * c - 0.5 * d
    s6 = (b + c) * (d - e) / 2.0
    s7 = b * b - c * c + 0.25 * e
    s8 = np.abs(b) - np.abs(c) + 0.5 * d
    s9 = np.where(d > 0, b + e, -c + e)
    s10 = b * d - c * e
    s11 = (b + d) / 2.0 - c * e
    s12 = np.where(b * c > 0, d + e, d - e)
    s13 = b * b * c - d
    s14 = np.tanh(b + c) - 0.5 * (d + e)  # bounded nonlinearity
    s15 = (b - e) * (c + d) / 2.0
    return np.stack(
        [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15],
        axis=0,
    )


def _piecewise_y(df: pd.DataFrame) -> pd.Series:
    """Score y anhand von Stratum ``a`` (unterstützt a∈{0,…,15})."""
    a = df["a"].to_numpy(dtype=int)
    if a.min() < 0 or a.max() > 15:
        raise ValueError("piecewise rules are defined only for a∈{0,…,15}")
    scores = _stratum_scores(
        df["b"].to_numpy(dtype=float),
        df["c"].to_numpy(dtype=float),
        df["d"].to_numpy(dtype=float),
        df["e"].to_numpy(dtype=float),
    )
    y = scores[a, np.arange(len(df))]
    return pd.Series(y, index=df.index)


def _base_abc(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.integers(0, 2, n),
            "b": rng.uniform(-1, 1, n),
            "c": rng.uniform(-1, 1, n),
        }
    )


def _base_piecewise(
    n: int, seed: int, n_strata: int = 4, label_noise: float = 0.0
) -> pd.DataFrame:
    if n_strata < 1 or n_strata > 16:
        raise ValueError("n_strata must be in 1..16")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.integers(0, n_strata, n),
            "b": rng.uniform(-1, 1, n),
            "c": rng.uniform(-1, 1, n),
            "d": rng.uniform(-1, 1, n),
            "e": rng.uniform(-1, 1, n),
        }
    )
    df["y"] = _piecewise_y(df)
    df["target"] = _apply_label_noise((df["y"] > 0).to_numpy(), label_noise, rng)
    return df


def with_indicator_interactions(spec: DatasetSpec) -> DatasetSpec:
    """Indikator × Merkmal für B3: die LR kann dann die Regionsregel darstellen."""
    indicators = [c for c in spec.search_columns if c in spec.feature_columns]
    if not indicators:
        return spec
    df = spec.df.copy()
    feats = list(spec.feature_columns)
    continuous = [
        f
        for f in spec.feature_columns
        if f not in indicators and df[f].dtype.kind == "f"
    ]
    for ind in indicators:
        for level in sorted(df[ind].unique()):
            mask = (df[ind] == level).astype(float)
            for feat in continuous:
                col = f"{ind}{level}_x_{feat}"
                df[col] = mask * df[feat]
                feats.append(col)
    return DatasetSpec(
        experiment_id=spec.experiment_id,
        name=f"{spec.name}__sat",
        df=df,
        label_column=spec.label_column,
        feature_columns=feats,
        search_columns=list(spec.search_columns),
        ground_truth_subgroups=list(spec.ground_truth_subgroups),
        description=spec.description + " [+Indikator×Merkmal-Interaktionen]",
    )


def with_noise_features(
    spec: DatasetSpec, n_noise: int, *, seed: int = 42, prefix: str = "dilute"
) -> DatasetSpec:
    """Hängt irrelevante Standardnormalspalten an (C-Verdünnung, E3, F2)."""
    if n_noise <= 0:
        return spec
    rng = np.random.default_rng(seed + 9176)
    cols = pd.DataFrame(
        rng.normal(0, 1, (len(spec.df), n_noise)),
        columns=[f"{prefix}_{i}" for i in range(n_noise)],
        index=spec.df.index,
    )
    return DatasetSpec(
        experiment_id=spec.experiment_id,
        name=f"{spec.name}__dil{n_noise}",
        df=pd.concat([spec.df, cols], axis=1),
        label_column=spec.label_column,
        feature_columns=list(spec.feature_columns) + list(cols.columns),
        search_columns=list(spec.search_columns),
        ground_truth_subgroups=list(spec.ground_truth_subgroups),
        description=spec.description + f" [+{n_noise} irrelevante Spalten]",
    )


def without_indicator_in_features(
    spec: DatasetSpec, indicator: str | None = None
) -> DatasetSpec:
    """Drop subgroup indicator from model features (keep in search_columns)."""
    ind = indicator
    if ind is None:
        for cand in ("a", "region", "group"):
            if cand in spec.feature_columns and cand in (spec.search_columns or []):
                ind = cand
                break
    if ind is None or ind not in spec.feature_columns:
        return spec
    return DatasetSpec(
        experiment_id=spec.experiment_id,
        name=spec.name,
        df=spec.df,
        label_column=spec.label_column,
        feature_columns=[c for c in spec.feature_columns if c != ind],
        search_columns=list(spec.search_columns),
        ground_truth_subgroups=list(spec.ground_truth_subgroups),
        description=spec.description + f" [ind∉F: drop {ind}]",
    )


def make_1_1_1(
    n: int = 10_000,
    seed: int = 42,
    effect_scale: float = 1.0,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """Vorzeichenwechsel: a==0: y=-b+c, a==1: y=b+c. Träger von B3 und dem Demo-Notebook."""
    df = _base_abc(n, seed)
    df["y"] = np.where(
        df["a"] == 0,
        effect_scale * (-df["b"] + df["c"]),
        effect_scale * (df["b"] + df["c"]),
    )
    df["target"] = _apply_label_noise(
        (df["y"] > 0).to_numpy(), label_noise, np.random.default_rng(seed + 1001)
    )
    f = ["a", "b", "c"]
    return DatasetSpec(
        "sign_flip",
        "opposite_signal",
        df,
        feature_columns=f,
        search_columns=["a"],
        ground_truth_subgroups=["a==0", "a==1"],
        description="Gegensätzliches Signal (Vorzeichenwechsel).",
    )


def make_1_1_2(
    n: int = 10_000,
    seed: int = 42,
    beta0: float = 4.0,
    beta1: float = 1.0,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """Heterogene Steigung (C1): a==0: y=beta0*x1+x2, a==1: y=beta1*x1+x2."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.integers(0, 2, n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    df["y"] = np.where(
        df["a"] == 0, beta0 * df["x1"] + df["x2"], beta1 * df["x1"] + df["x2"]
    )
    df["target"] = _apply_label_noise((df["y"] > 0).to_numpy(), label_noise, rng)
    f = ["a", "x1", "x2"]
    return DatasetSpec(
        "heterogeneous_slope",
        "shifted_signal",
        df,
        feature_columns=f,
        search_columns=["a"],
        ground_truth_subgroups=["a==0", "a==1"],
        description=(
            f"Heterogene Steigung: beta0={beta0}, beta1={beta1} "
            "(Vorzeichenwechsel wenn beta0/beta1<0)."
        ),
    )


def make_1_1_4(
    n: int = 10_000,
    seed: int = 42,
    n_noise: int = 0,
    n_strata: int = 4,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """Stückweise Regeln (C2, D1, D4): a∈{0,…,n_strata-1}, feste Regel pro Stratum."""
    allowed = {2, 4, 8, 16}
    if n_strata not in allowed:
        raise ValueError(f"n_strata must be one of {sorted(allowed)}, got {n_strata}")
    df = _base_piecewise(n, seed, n_strata=n_strata, label_noise=label_noise)
    f = ["a", "b", "c", "d", "e"]
    if n_noise > 0:
        noise = _noise_columns(n, n_noise, np.random.default_rng(seed + 99))
        df = pd.concat([df, noise], axis=1)
        f = f + [f"noise_{i}" for i in range(n_noise)]
    return DatasetSpec(
        "piecewise",
        "piecewise_complex",
        df,
        feature_columns=f,
        search_columns=["a"],
        ground_truth_subgroups=[f"a=={i}" for i in range(n_strata)],
        description=f"Abschnittsweise verschiedenes Signal (n_strata={n_strata}).",
    )


def make_1_3_1(
    n: int = 10_000,
    seed: int = 42,
    n_noise: int = 50,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """Stückweiser DGP plus irrelevante Spalten (E3)."""
    spec = make_1_1_4(n=n, seed=seed, n_noise=n_noise, label_noise=label_noise)
    return DatasetSpec(
        "noise_features",
        "noise_features",
        spec.df,
        feature_columns=spec.feature_columns,
        search_columns=spec.search_columns,
        ground_truth_subgroups=spec.ground_truth_subgroups,
        description="Piecewise DGP mit vielen irrelevanten Noise-Features.",
    )


def make_2_1_feature_shift(
    n: int = 10_000, seed: int = 42, label_noise: float = 0.0
) -> DatasetSpec:
    """B2: x1 verschoben in region==1, gleiche Regel y=sign(x1)."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 2, n)
    x1 = np.zeros(n)
    x1[region == 0] = rng.normal(0, 1, (region == 0).sum())
    x1[region == 1] = rng.normal(3, 1, (region == 1).sum())
    target = _apply_label_noise((x1 > 0).astype(int), label_noise, rng)
    df = pd.DataFrame({"region": region, "x1": x1, "target": target})
    f = ["region", "x1"]
    return DatasetSpec(
        "feature_shift",
        "feature_shift",
        df,
        feature_columns=f,
        search_columns=["region"],
        # region==1 is nearly single-class (x1~N(3,1)); use region==0 as
        # the meaningful negative-control subgroup for adaptability Q.
        ground_truth_subgroups=["region==0"],
        description="Feature-Shift: x1~N(3,1) in region==1, sonst N(0,1); y=sign(x1).",
    )


def make_2_2_subgroup_size(
    n: int = 10_000,
    seed: int = 42,
    minority_fraction: float = 0.1,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """E1: Minderheit a==0 mit gegensätzlichem Signal."""
    rng = np.random.default_rng(seed)
    n_min = int(n * minority_fraction)
    a = np.zeros(n, dtype=int)
    a[:n_min] = 0
    a[n_min:] = 1
    rng.shuffle(a)
    b = rng.uniform(-1, 1, n)
    c = rng.uniform(-1, 1, n)
    y = np.where(a == 0, -b + c, b + c)
    target = _apply_label_noise(y > 0, label_noise, rng)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "y": y, "target": target})
    f = ["a", "b", "c"]
    return DatasetSpec(
        "subgroup_size",
        "subgroup_size",
        df,
        feature_columns=f,
        search_columns=["a"],
        ground_truth_subgroups=["a==0"],
        description=f"Minderheit a==0 ({minority_fraction:.0%}) mit gegensätzlichem Signal.",
    )


def make_2_3_class_imbalance(n: int = 10_000, seed: int = 42) -> DatasetSpec:
    """E2: extreme Klassenschiefe in region==0, gleiche Regel."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    logit = x1.copy()
    target = (logit + rng.normal(0, 0.5, n) > 0).astype(int)
    # region==0: 95% Klasse 0 erzwingen (Label-Shift / Imbalance)
    mask0 = region == 0
    n0 = mask0.sum()
    n_pos = max(1, int(0.05 * n0))
    idx0 = np.where(mask0)[0]
    pos_idx = rng.choice(idx0, size=n_pos, replace=False)
    target[pos_idx] = 1
    target[np.setdiff1d(idx0, pos_idx)] = 0
    df = pd.DataFrame({"region": region, "x1": x1, "target": target})
    f = ["region", "x1"]
    return DatasetSpec(
        "class_imbalance",
        "class_imbalance",
        df,
        feature_columns=f,
        search_columns=["region"],
        ground_truth_subgroups=["region==0", "region==1"],
        description="region==0: ~95% Klasse 0; prüft Class-Balance-Artefakte.",
    )


def make_3_1_homogeneous(
    n: int = 10_000, seed: int = 42, label_noise: float = 0.0
) -> DatasetSpec:
    """Homogener DGP (A0, A1, F1): überall dieselbe Regel."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = x1 + x2
    target = _apply_label_noise(y > 0, label_noise, rng)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y, "target": target})
    f = ["x1", "x2"]
    return DatasetSpec(
        "homogeneous",
        "homogeneous",
        df,
        feature_columns=f,
        search_columns=["x1"],
        ground_truth_subgroups=[],
        description="Homogener DGP — erwartet Quality ≈ 0.",
    )


def make_3_2_overfitting(
    n: int = 2_000,
    seed: int = 42,
    n_noise: int = 40,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """F2: kleine Stichprobe, viele irrelevante Merkmale."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    target = _apply_label_noise((x1 > 0).astype(int), label_noise, rng)
    df = pd.DataFrame({"group": rng.integers(0, 2, n), "x1": x1, "target": target})
    noise = _noise_columns(n, n_noise, rng)
    df = pd.concat([df, noise], axis=1)
    f = ["group", "x1"] + [f"noise_{i}" for i in range(n_noise)]
    return DatasetSpec(
        "overfitting",
        "overfitting_control",
        df,
        feature_columns=f,
        search_columns=["group"],
        ground_truth_subgroups=["group==0", "group==1"],
        description="Wenig Daten, viele Features — Overfitting-Check (train vs test).",
    )


def make_I_1f_label_shift(
    n: int = 10_000,
    seed: int = 42,
    prior0: float = 0.5,
    prior1: float = 0.1,
    mu0: float = -1.0,
    mu1: float = 1.0,
) -> DatasetSpec:
    """Prior-Shift (B1): gleiches P(x|y), anderer Klassenanteil je Region."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 2, n)
    target = np.zeros(n, dtype=int)
    for r, prior in ((0, prior0), (1, prior1)):
        mask = region == r
        target[mask] = rng.random(mask.sum()) < prior
    x1 = np.where(target == 1, rng.normal(mu1, 1.0, n), rng.normal(mu0, 1.0, n))
    df = pd.DataFrame({"region": region, "x1": x1, "target": target})
    return DatasetSpec(
        "label_shift",
        "label_shift",
        df,
        feature_columns=["region", "x1"],
        search_columns=["region"],
        ground_truth_subgroups=["region==0", "region==1"],
        description=(
            f"Label-Shift: P(y=1|r=0)={prior0}, P(y=1|r=1)={prior1}; "
            "same Gaussian class-conditionals."
        ),
    )


def make_I_2c_heterogeneous_noise(
    n: int = 10_000,
    seed: int = 42,
    flip_out: float = 0.4,
    flip_in: float = 0.0,
    gt_fraction: float = 0.25,
    noise_mode: str = "feature_dependent",
    label_noise: float = 0.0,
) -> DatasetSpec:
    """C4: überall y=sign(x1+x2); außerhalb der Region drei Label-Modi.

    ``uniform`` / ``feature_dependent`` ändern die Rangordnung nicht.
    ``concept_shift`` setzt außerhalb sign(x1-x2).
    """
    mode = noise_mode.strip().lower()
    if mode not in {"uniform", "feature_dependent", "concept_shift"}:
        raise ValueError(
            "noise_mode must be uniform|feature_dependent|concept_shift, "
            f"got {noise_mode!r}"
        )
    rng = np.random.default_rng(seed)
    region = np.zeros(n, dtype=int)
    n_gt = int(n * gt_fraction)
    region[:n_gt] = 1
    rng.shuffle(region)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = (x1 + x2 > 0).astype(int)

    alt_rule = (x1 - x2 > 0).astype(int)
    for region_mask, p in ((region == 1, flip_in), (region == 0, flip_out)):
        if not p:
            continue
        # All modes flip the same *number* of labels, so the curves are
        # comparable at equal p and only the placement of the noise differs.
        n_flip = int(int(region_mask.sum()) * p)
        if mode == "uniform":
            eligible = region_mask
        elif mode == "feature_dependent":
            eligible = region_mask & (x2 > 0)
        else:  # concept_shift: disagreements with the alternative rule
            eligible = region_mask & (target != alt_rule)
        idx = np.where(eligible)[0]
        n_flip = min(n_flip, len(idx))
        if n_flip:
            flip_idx = rng.choice(idx, size=n_flip, replace=False)
            target[flip_idx] = 1 - target[flip_idx]

    target = _apply_label_noise(target, label_noise, rng)
    df = pd.DataFrame({"region": region, "x1": x1, "x2": x2, "target": target})
    return DatasetSpec(
        "heterogeneous_noise",
        f"heterogeneous_noise_{mode}",
        df,
        feature_columns=["region", "x1", "x2"],
        search_columns=["region"],
        ground_truth_subgroups=["region==1"],
        description=(
            f"y=sign(x1+x2); {mode} label noise p={flip_out:.0%} outside region==1, "
            f"p={flip_in:.0%} inside (gt_fraction={gt_fraction}), "
            f"Bayes noise {label_noise:.0%} everywhere."
        ),
    )


def make_I_3c_true_shortcut(
    n: int = 10_000,
    seed: int = 42,
    shortcut_strength: float = 0.9,
    minority_fraction: float = 0.2,
    minority_shortcut_corr: float = -1.0,
    signal_noise: float = 0.5,
    n_noise: int = 5,
    label_noise: float = 0.0,
    interaction_order: int = 2,
) -> DatasetSpec:
    """C3: wahre Regel ist das Produkt von ``interaction_order`` Normalen.

    Der Shortcut kopiert das Vorzeichen der Mehrheit und kehrt in der
    Minderheit die Korrelation um (``minority_shortcut_corr``).
    """
    rng = np.random.default_rng(seed)
    region = np.zeros(n, dtype=int)
    n_min = int(n * minority_fraction)
    region[:n_min] = 1
    rng.shuffle(region)

    if interaction_order < 1:
        raise ValueError("interaction_order must be >= 1")
    xs = [rng.normal(0, 1, n) for _ in range(interaction_order)]
    latent = np.prod(xs, axis=0)
    target = _apply_label_noise(
        (latent + rng.normal(0, signal_noise, n) > 0).astype(int), label_noise, rng
    )

    signed = 2.0 * (latent > 0).astype(float) - 1.0
    corr = np.where(region == 0, 1.0, minority_shortcut_corr)
    shortcut = shortcut_strength * corr * signed + rng.normal(
        0, max(1.0 - shortcut_strength, 0.05), n
    )

    x_names = [f"x{i + 1}" for i in range(interaction_order)]
    df = pd.DataFrame(
        {
            "region": region,
            **dict(zip(x_names, xs)),
            "shortcut": shortcut,
            "target": target,
        }
    )
    if n_noise > 0:
        df = pd.concat([df, _noise_columns(n, n_noise, rng)], axis=1)
    f = ["region"] + x_names + ["shortcut"] + [f"noise_{i}" for i in range(n_noise)]
    return DatasetSpec(
        "shortcut",
        "true_shortcut",
        df,
        feature_columns=f,
        search_columns=["region"],
        ground_truth_subgroups=["region==1"],
        description=(
            f"y~sign({'*'.join(x_names)} + noise); shortcut "
            f"(strength={shortcut_strength}) valid on the majority, correlation "
            f"{minority_shortcut_corr:+.1f} in region==1 ({minority_fraction:.0%})."
        ),
    )


def make_I_1g_hidden_stratification(
    n: int = 10_000,
    seed: int = 42,
    hidden_fraction: float = 0.15,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """C5: Mehrheit folgt x1+0.5 x2, die kleine sichtbare Gruppe folgt x3."""
    rng = np.random.default_rng(seed)
    stratum = np.zeros(n, dtype=int)
    n_hidden = int(n * hidden_fraction)
    stratum[:n_hidden] = 1
    rng.shuffle(stratum)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    score = np.where(stratum == 1, x3, x1 + 0.5 * x2)
    target = _apply_label_noise((score > 0).astype(int), label_noise, rng)

    df = pd.DataFrame(
        {"group": stratum, "x1": x1, "x2": x2, "x3": x3, "target": target}
    )
    return DatasetSpec(
        "hidden_stratification",
        "hidden_stratification",
        df,
        feature_columns=["group", "x1", "x2", "x3"],
        search_columns=["group"],
        ground_truth_subgroups=["group==1"],
        description=(
            f"Hidden stratum group==1 ({hidden_fraction:.0%}) follows x3; "
            "majority follows x1+0.5*x2."
        ),
    )


def make_I_4c_misspec_shift_2d(
    n: int = 10_000,
    seed: int = 42,
    angle_scale: float = 1.0,
    label_noise: float = 0.0,
) -> DatasetSpec:
    """C6: überall y=sign(x1*x2), die Kovariatenverteilung rotiert pro Region."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 2, n)
    x1 = np.where(
        region == 0,
        rng.normal(0.0, 1.0, n),
        rng.normal(angle_scale, 1.0, n),
    )
    x2 = np.where(
        region == 0,
        rng.normal(angle_scale, 1.0, n),
        rng.normal(0.0, 1.0, n),
    )
    target = _apply_label_noise((x1 * x2 > 0).astype(int), label_noise, rng)
    df = pd.DataFrame({"region": region, "x1": x1, "x2": x2, "target": target})
    return DatasetSpec(
        "misspec_shift",
        "misspec_shift_2d",
        df,
        feature_columns=["region", "x1", "x2"],
        search_columns=["region"],
        ground_truth_subgroups=["region==0", "region==1"],
        description=(
            f"y=sign(x1*x2) everywhere; p(x) rotated per region "
            f"(angle_scale={angle_scale})."
        ),
    )


def with_lr_interaction_features(
    spec: DatasetSpec,
    *,
    include_interactions: bool,
    indicator: str = "a",
    base_features: tuple[str, ...] = ("b", "c"),
) -> DatasetSpec:
    """B3: LR-Features additiv oder mit Indikator×Merkmal-Interaktionen."""
    df = spec.df.copy()
    feats = [indicator, *base_features]
    if include_interactions:
        for feat in base_features:
            col = f"{indicator}_x_{feat}"
            df[col] = df[indicator] * df[feat]
            feats.append(col)
    return DatasetSpec(
        experiment_id=spec.experiment_id,
        name="with_interaction" if include_interactions else "no_interaction",
        df=df,
        label_column=spec.label_column,
        feature_columns=feats,
        search_columns=list(spec.search_columns),
        ground_truth_subgroups=list(spec.ground_truth_subgroups),
        description=(
            "LR with a·x interactions" if include_interactions else "LR additive [a,b,c]"
        ),
    )


def _as_experiment(spec: DatasetSpec, exp_id: str, name: str | None = None) -> DatasetSpec:
    spec.experiment_id = exp_id
    if name is not None:
        spec.name = name
    return spec


def _maybe_dilute(spec: DatasetSpec, n_noise: int, seed: int) -> DatasetSpec:
    if n_noise:
        return with_noise_features(spec, n_noise, seed=seed)
    return spec


def make_a0(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    *,
    carrier: str = "homogeneous",
    n_noise: int = 0,
) -> DatasetSpec:
    """A0: Träger ohne Mechanismus. Q wird auf Zufallsregionen gemessen."""
    if carrier == "piecewise":
        spec = make_1_1_4(n=n, seed=seed, label_noise=label_noise)
    elif carrier == "homogeneous":
        spec = make_3_1_homogeneous(n=n, seed=seed, label_noise=label_noise)
    elif carrier == "homogeneous_diluted":
        spec = make_3_1_homogeneous(n=n, seed=seed, label_noise=label_noise)
        spec = with_noise_features(spec, n_noise or 60, seed=seed)
    else:
        raise ValueError(f"unknown A0 carrier: {carrier!r}")
    return _as_experiment(spec, "A0", f"a0_{carrier}")


def make_a1(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
) -> DatasetSpec:
    """A1: homogener DGP, Sweep über Labelrauschen."""
    return _as_experiment(
        make_3_1_homogeneous(n=n, seed=seed, label_noise=label_noise), "A1"
    )


def make_b1(
    n: int = 4_000,
    seed: int = 42,
    prior0: float = 0.5,
    prior1: float = 0.15,
    **kwargs,
) -> DatasetSpec:
    """B1: reiner Prior-Shift (gleiche Regel, anderer Klassenanteil)."""
    kwargs.pop("label_noise", None)
    return _as_experiment(
        make_I_1f_label_shift(n=n, seed=seed, prior0=prior0, prior1=prior1, **kwargs),
        "B1",
    )


def make_b2(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
) -> DatasetSpec:
    """B2: Feature-Shift, Regel bleibt in der Hypothesenklasse."""
    return _as_experiment(
        make_2_1_feature_shift(n=n, seed=seed, label_noise=label_noise), "B2"
    )


def make_b3(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    *,
    include_interactions: bool = False,
    carrier: str = "sign_flip",
    saturated: bool = False,
) -> DatasetSpec:
    """B3: LR mit/ohne Interaktionsterme, optional auf einem C-Träger."""
    carriers = {
        "sign_flip": lambda: make_1_1_1(n=n, seed=seed, label_noise=label_noise),
        "C1_slope": lambda: make_1_1_2(
            n=n, seed=seed, beta0=-2.0, beta1=1.0, label_noise=label_noise
        ),
        "C2_piecewise": lambda: make_1_1_4(
            n=n, seed=seed, n_strata=4, label_noise=label_noise
        ),
        "C3_shortcut": lambda: make_I_3c_true_shortcut(
            n=n, seed=seed, label_noise=label_noise
        ),
        "C4_concept_shift": lambda: make_I_2c_heterogeneous_noise(
            n=n,
            seed=seed,
            flip_out=0.5,
            noise_mode="concept_shift",
            label_noise=label_noise,
        ),
        "C5_hidden": lambda: make_I_1g_hidden_stratification(
            n=n, seed=seed, label_noise=label_noise
        ),
        "C6_misspec_shift": lambda: make_I_4c_misspec_shift_2d(
            n=n, seed=seed, angle_scale=2.0, label_noise=label_noise
        ),
    }
    if carrier not in carriers:
        raise ValueError(f"unknown B3 carrier: {carrier!r}")
    spec = carriers[carrier]()
    if carrier == "sign_flip":
        spec = with_lr_interaction_features(
            spec, include_interactions=include_interactions
        )
    elif saturated:
        spec = with_indicator_interactions(spec)
    return _as_experiment(spec, "B3", f"b3_{carrier}")


def make_c1(
    n: int = 4_000,
    seed: int = 42,
    beta0: float = -2.0,
    beta1: float = 1.0,
    label_noise: float = DEFAULT_LABEL_NOISE,
    n_noise: int = 0,
) -> DatasetSpec:
    """C1: heterogene Steigung in zwei Regionen."""
    spec = make_1_1_2(
        n=n, seed=seed, beta0=beta0, beta1=beta1, label_noise=label_noise
    )
    return _as_experiment(_maybe_dilute(spec, n_noise, seed), "C1")


def make_c2(
    n: int = 4_000,
    seed: int = 42,
    n_strata: int = 4,
    label_noise: float = DEFAULT_LABEL_NOISE,
    per_stratum: int | None = None,
    n_noise: int = 0,
) -> DatasetSpec:
    """C2: stückweise Regeln. ``per_stratum`` hält die Stratumgröße fest."""
    if per_stratum is not None:
        n = per_stratum * n_strata
    spec = make_1_1_4(
        n=n, seed=seed, n_strata=n_strata, label_noise=label_noise
    )
    return _as_experiment(_maybe_dilute(spec, n_noise, seed), "C2")


def make_c3(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    minority_fraction: float = 0.2,
    shortcut_strength: float = 0.9,
    minority_shortcut_corr: float = -1.0,
    interaction_order: int = 2,
    n_noise: int = 5,
) -> DatasetSpec:
    """C3: Shortcut mit Korrelationsumkehr in der Minderheit.

    ``n_noise`` ist das Rauschen des Shortcut-DGPs (Standard 5), nicht die
    C-Verdünnungsachse.
    """
    spec = make_I_3c_true_shortcut(
        n=n,
        seed=seed,
        label_noise=label_noise,
        minority_fraction=minority_fraction,
        shortcut_strength=shortcut_strength,
        minority_shortcut_corr=minority_shortcut_corr,
        interaction_order=interaction_order,
        n_noise=n_noise,
    )
    return _as_experiment(spec, "C3")


def make_c4(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    noise_mode: str = "concept_shift",
    flip_out: float = 0.5,
    gt_fraction: float = 0.25,
    n_noise: int = 0,
) -> DatasetSpec:
    """C4: heterogenes Rauschen bzw. Konzeptwechsel."""
    spec = make_I_2c_heterogeneous_noise(
        n=n,
        seed=seed,
        flip_out=flip_out,
        noise_mode=noise_mode,
        gt_fraction=gt_fraction,
        label_noise=label_noise,
    )
    return _as_experiment(_maybe_dilute(spec, n_noise, seed), "C4")


def make_c5(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    hidden_fraction: float = 0.2,
    n_noise: int = 0,
) -> DatasetSpec:
    """C5: Hidden Stratification."""
    spec = make_I_1g_hidden_stratification(
        n=n, seed=seed, hidden_fraction=hidden_fraction, label_noise=label_noise
    )
    return _as_experiment(_maybe_dilute(spec, n_noise, seed), "C5")


def make_c6(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    angle_scale: float = 2.0,
    n_noise: int = 0,
) -> DatasetSpec:
    """C6: Misspezifikation plus Feature-Shift in 2D."""
    spec = make_I_4c_misspec_shift_2d(
        n=n, seed=seed, angle_scale=angle_scale, label_noise=label_noise
    )
    return _as_experiment(_maybe_dilute(spec, n_noise, seed), "C6")


def make_d1(
    n: int = 4_000,
    seed: int = 42,
    n_strata: int = 4,
    label_noise: float = DEFAULT_LABEL_NOISE,
) -> DatasetSpec:
    """D1: stückweise Mischung für das Kapazitätsgitter."""
    return _as_experiment(
        make_1_1_4(n=n, seed=seed, n_strata=n_strata, label_noise=label_noise),
        "D1",
    )


def make_d2(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    n_strata: int = 4,
) -> DatasetSpec:
    """D2: gleicher Träger wie D1, Sweep über Regularisierung."""
    return _as_experiment(
        make_1_1_4(n=n, seed=seed, n_strata=n_strata, label_noise=label_noise),
        "D2",
    )


def make_d3(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    n_strata: int = 4,
) -> DatasetSpec:
    """D3: fester Datensatz, nur Modell-Seed variiert."""
    return _as_experiment(
        make_1_1_4(n=n, seed=seed, n_strata=n_strata, label_noise=label_noise),
        "D3",
    )


def make_d4(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    *,
    variant: str = "jagged",
    n_strata: int = 8,
    beta0: float = -1.0,
    beta1: float = 1.0,
) -> DatasetSpec:
    """D4: zackige stückweise vs. glatte lineare Zielregel."""
    if variant == "jagged":
        spec = make_1_1_4(
            n=n, seed=seed, n_strata=n_strata, label_noise=label_noise
        )
    elif variant == "smooth":
        spec = make_1_1_2(
            n=n, seed=seed, beta0=beta0, beta1=beta1, label_noise=label_noise
        )
    else:
        raise ValueError(f"unknown D4 variant: {variant!r}")
    return _as_experiment(spec, "D4", f"d4_{variant}")


def make_e1(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    minority_fraction: float = 0.2,
) -> DatasetSpec:
    """E1: fester Mechanismus, Sweep über den Minderheitenanteil."""
    return _as_experiment(
        make_2_2_subgroup_size(
            n=n,
            seed=seed,
            minority_fraction=minority_fraction,
            label_noise=label_noise,
        ),
        "E1",
    )


def make_e2(
    n: int = 4_000,
    seed: int = 42,
    **kwargs,
) -> DatasetSpec:
    """E2: extreme Klassenunbalance ohne Regelwechsel."""
    kwargs.pop("label_noise", None)
    return _as_experiment(make_2_3_class_imbalance(n=n, seed=seed, **kwargs), "E2")


def make_e3(
    n: int = 4_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    n_noise: int = 50,
) -> DatasetSpec:
    """E3: Zahl irrelevanter Merkmale."""
    return _as_experiment(
        make_1_3_1(n=n, seed=seed, n_noise=n_noise, label_noise=label_noise),
        "E3",
    )


def make_f1(
    n: int = 3_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
) -> DatasetSpec:
    """F1: homogen, volle Suche."""
    return _as_experiment(
        make_3_1_homogeneous(n=n, seed=seed, label_noise=label_noise), "F1"
    )


def make_f2(
    n: int = 1_500,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    n_noise: int = 40,
) -> DatasetSpec:
    """F2: kleine Stichprobe, viele Rauschspalten."""
    return _as_experiment(
        make_3_2_overfitting(
            n=n, seed=seed, n_noise=n_noise, label_noise=label_noise
        ),
        "F2",
    )


def make_g1(
    n: int = 3_000,
    seed: int = 42,
    label_noise: float = DEFAULT_LABEL_NOISE,
    *,
    carrier: str = "piecewise",
    n_noise: int = 0,
) -> DatasetSpec:
    """G1: Träger für die Mini-Suche."""
    builders = {
        "piecewise": lambda: make_1_1_4(n=n, seed=seed, label_noise=label_noise),
        "hidden": lambda: make_I_1g_hidden_stratification(
            n=n, seed=seed, hidden_fraction=0.2, label_noise=label_noise
        ),
        "concept_shift": lambda: make_I_2c_heterogeneous_noise(
            n=n,
            seed=seed,
            flip_out=0.5,
            noise_mode="concept_shift",
            label_noise=label_noise,
        ),
    }
    if carrier not in builders:
        raise ValueError(f"unknown G1 carrier: {carrier!r}")
    spec = _maybe_dilute(builders[carrier](), n_noise, seed)
    return _as_experiment(spec, "G1", f"g1_{carrier}")
