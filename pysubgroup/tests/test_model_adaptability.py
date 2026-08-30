"""Unit tests for the model-adaptability quality function and result API.

Pins the method of chapters 3–4: Q_roh, Q_gew, Q_ga, scorability, and the
success criterion. Process-search equivalence lives in
``test_process_model_adaptability_dfs.py``.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import pysubgroup as ps
from pysubgroup.model_adaptability_target import (
    _SizeStatsWithCovers,
    _label_balance_fraction,
    meets_success_criterion,
    scorability_status,
)


def _opposite_signal_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Inline opposite-signal DGP (a flips the sign of b)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.integers(0, 2, n),
            "b": rng.uniform(-1, 1, n),
            "c": rng.uniform(-1, 1, n),
        }
    )
    df["y"] = np.where(df["a"] == 0, -df["b"] + df["c"], df["b"] + df["c"])
    df["target"] = (df["y"] > 0).astype(int)
    return df.drop(columns=["y"])


def _lr_qf(**kwargs):
    """Logistic-regression QF with the chapter-4 protocol defaults."""
    defaults = dict(
        model_builder_global=lambda: LogisticRegression(max_iter=500, random_state=0),
        training_global=lambda m, X, y: m.fit(X, y),
        random_state=0,
    )
    defaults.update(kwargs)
    return ps.LocalSoftClassifierPerformanceQF(**defaults)


def _run_discovery(
    df,
    *,
    depth=1,
    min_support=20,
    result_set_size=5,
    generalization_awareness=True,
    algorithm=None,
    qf=None,
):
    qf = qf or _lr_qf()
    target = ps.ModelAdaptabilityTarget(
        label_column="target",
        feature_columns=["a", "b", "c"],
    )
    # Integer search columns need per-attribute nominal selectors (not select_dtypes).
    search_space = ps.create_nominal_selectors_for_attribute(df, "a")
    task = ps.ModelAdaptabilityDiscoveryTask(
        data=df,
        target=target,
        search_space=search_space,
        qf=qf,
        depth=depth,
        result_set_size=result_set_size,
        constraints=[ps.MinSupportConstraint(min_support)],
        generalization_awareness=generalization_awareness,
    )
    if algorithm is None:
        algorithm = ps.ParallelModelAdaptabilityDFS(max_workers=2)
    return task.execute(algorithm=algorithm)


class TestModelAdaptability(unittest.TestCase):
    def test_opposite_signal_smoke(self):
        df = _opposite_signal_df(n=500, seed=1)
        result = _run_discovery(df, depth=1, min_support=30, result_set_size=5)
        table = result.to_dataframe()
        subgroups = table.iloc[1:]
        self.assertGreater(len(subgroups), 0)
        top = subgroups.head(2)
        descriptions = [str(s) for s in top["subgroup"]]
        self.assertTrue(
            any("a" in d for d in descriptions),
            f"expected ground-truth-like a-selector in top-2, got {descriptions}",
        )
        self.assertTrue((top["quality"] > 0).any())

    def test_size_sg_semantics(self):
        df = _opposite_signal_df(n=300, seed=2)
        result = _run_discovery(df, depth=1, min_support=20, result_set_size=3)
        table = result.to_dataframe()
        for _, row in table.iterrows():
            self.assertEqual(
                int(row["size_sg"]),
                int(row["size_sg_train"]) + int(row["size_sg_test"]),
            )

    def test_min_support_before_fit_prunes(self):
        df = _opposite_signal_df(n=200, seed=3)
        result = _run_discovery(
            df, depth=1, min_support=len(df) + 10, result_set_size=5
        )
        table = result.to_dataframe()
        subgroups = [
            row for _, row in table.iterrows() if str(row["subgroup"]) != "Dataset"
        ]
        self.assertEqual(subgroups, [])

    def test_cover_reuse_from_size_stats(self):
        df = _opposite_signal_df(n=250, seed=4)
        qf = _lr_qf()
        target = ps.ModelAdaptabilityTarget(
            label_column="target", feature_columns=["a", "b", "c"]
        )
        train = qf.resolve_search_data(df, target)
        qf.calculate_constant_statistics(train, target)

        sg = ps.create_subgroup_with_representation(
            train, [ps.EqualitySelector("a", 0)]
        )
        size_stats = qf.calculate_size_statistics(sg, target, train)
        self.assertIsInstance(size_stats, _SizeStatsWithCovers)
        self.assertFalse(qf._statistics_complete(size_stats))

        calls = {"n": 0}
        original = qf._covers_and_balance

        def counting_covers(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        qf._covers_and_balance = counting_covers
        full_stats = qf.calculate_statistics(sg, target, train, size_stats)
        self.assertEqual(calls["n"], 0)
        self.assertEqual(full_stats.size_sg, size_stats.size_sg)
        self.assertEqual(full_stats.size_sg_train, size_stats.size_sg_train)
        self.assertEqual(full_stats.size_sg_test, size_stats.size_sg_test)
        self.assertEqual(full_stats.class_balance, size_stats.class_balance)
        self.assertEqual(
            full_stats.size_sg, full_stats.size_sg_train + full_stats.size_sg_test
        )

    def test_stratified_default_split(self):
        rng = np.random.default_rng(5)
        n0, n1 = 80, 320
        df = pd.DataFrame(
            {
                "a": rng.integers(0, 2, n0 + n1),
                "b": rng.normal(size=n0 + n1),
                "target": np.array([0] * n0 + [1] * n1),
            }
        )
        qf = _lr_qf(random_state=5)
        target = ps.ModelAdaptabilityTarget(
            label_column="target", feature_columns=["a", "b"]
        )
        train = qf.resolve_search_data(df, target)
        test = qf.test_data
        train_rate = float(train["target"].mean())
        test_rate = float(test["target"].mean())
        full_rate = float(df["target"].mean())
        self.assertAlmostEqual(train_rate, full_rate, delta=0.05)
        self.assertAlmostEqual(test_rate, full_rate, delta=0.05)

    def test_generalization_awareness_is_noop_at_depth_one(self):
        """At depth 1 the only parent is the empty description (baseline 0).

        Q_ga then equals Q_roh. Ranking still uses the size-weighted Q_gew.
        """
        df = _opposite_signal_df(n=500, seed=3)
        table = _run_discovery(
            df, depth=1, generalization_awareness=True
        ).to_dataframe()
        subgroups = table.iloc[1:]
        self.assertGreater(len(subgroups), 0)
        np.testing.assert_allclose(
            subgroups["quality_ga"].to_numpy(dtype=float),
            subgroups["quality_raw"].to_numpy(dtype=float),
            atol=1e-12,
        )
        # Protocol α = 0.7 shrinks every proper subgroup, so the two scales differ.
        self.assertTrue(
            np.any(
                np.abs(
                    subgroups["quality"].to_numpy(dtype=float)
                    - subgroups["quality_raw"].to_numpy(dtype=float)
                )
                > 1e-9
            )
        )

    def test_protocol_weights_define_quality(self):
        """``quality`` is Q_gew with α = 0.7 and β = 0 (chapter 4)."""
        df = _opposite_signal_df(n=500, seed=4)
        table = _run_discovery(df).to_dataframe()
        dataset_size = float(table.iloc[0]["size_sg"])
        subgroups = table.iloc[1:]
        self.assertGreater(len(subgroups), 0)

        for _, row in subgroups.iterrows():
            raw = float(row["quality_raw"])
            expected = raw * (float(row["size_sg"]) / dataset_size) ** 0.7
            self.assertAlmostEqual(float(row["quality"]), expected, places=6)
            self.assertLess(abs(float(row["quality"])), abs(raw))

    def test_size_and_balance_weights_scale_quality(self):
        """Both factors of Q_gew are multiplicative when β is nonzero."""
        df = _opposite_signal_df(n=500, seed=4)
        table = _run_discovery(
            df,
            qf=_lr_qf(subgroup_size_weight=0.75, subgroup_class_balance_weight=0.25),
        ).to_dataframe()
        dataset_size = float(table.iloc[0]["size_sg"])
        subgroups = table.iloc[1:]
        self.assertGreater(len(subgroups), 0)

        for _, row in subgroups.iterrows():
            raw = float(row["quality_raw"])
            expected = (
                raw
                * (float(row["size_sg"]) / dataset_size) ** 0.75
                * float(row["class_balance"]) ** 0.25
            )
            self.assertAlmostEqual(float(row["quality"]), expected, places=6)
            self.assertLess(abs(float(row["quality"])), abs(raw))

    def test_generalization_awareness_column(self):
        df = _opposite_signal_df(n=300, seed=6)
        result_on = _run_discovery(df, generalization_awareness=True)
        result_off = _run_discovery(df, generalization_awareness=False)
        self.assertIn("quality_ga", result_on.to_dataframe().columns)
        self.assertNotIn("quality_ga", result_off.to_dataframe().columns)

    def test_to_dataframe_ranks_by_weighted_quality(self):
        """Displayed ranking is ``quality`` (Q_gew), not ``quality_ga``."""
        df = _opposite_signal_df(n=400, seed=7)
        table = _run_discovery(
            df, depth=1, result_set_size=0, generalization_awareness=True
        ).to_dataframe()
        subgroups = table.iloc[1:]
        self.assertGreater(len(subgroups), 1)
        qualities = subgroups["quality"].to_numpy(dtype=float)
        self.assertTrue(np.all(qualities[:-1] >= qualities[1:] - 1e-12))

    def test_protocol_defaults(self):
        qf = ps.LocalSoftClassifierPerformanceQF(
            model_builder_global=lambda: LogisticRegression(max_iter=200),
            training_global=lambda m, X, y: m.fit(X, y),
        )
        self.assertEqual(qf.subgroup_size_weight, 0.7)
        self.assertEqual(qf.subgroup_class_balance_weight, 0.0)
        self.assertEqual(qf.test_size, 0.5)
        self.assertEqual(qf.min_size_train, 5)
        self.assertEqual(qf.min_size_test, 5)

    def test_dataframe_reports_three_qualities_and_success_criterion(self):
        df = _opposite_signal_df(n=500, seed=8)
        result = _run_discovery(
            df, depth=1, min_support=30, result_set_size=0, generalization_awareness=True
        )
        table = result.to_dataframe()
        self.assertIn("quality_raw", table.columns)
        self.assertIn("quality", table.columns)
        self.assertIn("quality_ga", table.columns)
        self.assertIn("interesting", table.columns)

        dataset = table.iloc[0]
        self.assertEqual(str(dataset["subgroup"]), "Dataset")
        self.assertFalse(bool(dataset["interesting"]))
        subgroups = table.iloc[1:]
        self.assertGreater(len(subgroups), 0)
        np.testing.assert_allclose(
            subgroups["quality_raw"].to_numpy(dtype=float),
            subgroups["local_test_score"].to_numpy(dtype=float)
            - subgroups["global_test_score"].to_numpy(dtype=float),
            atol=1e-12,
        )
        expected_interesting = (subgroups["quality_raw"] > 0) & (
            subgroups["quality_ga"] >= 0
        )
        pd.testing.assert_series_equal(
            subgroups["interesting"].astype(bool),
            expected_interesting.astype(bool),
            check_names=False,
        )

        interesting_only = result.to_dataframe(interesting_only=True)
        self.assertEqual(str(interesting_only.iloc[0]["subgroup"]), "Dataset")
        rest = interesting_only.iloc[1:]
        if len(rest):
            self.assertTrue(rest["interesting"].all())
            self.assertTrue((rest["quality_raw"] > 0).all())
            self.assertTrue((rest["quality_ga"] >= 0).all())
            qualities = rest["quality"].to_numpy(dtype=float)
            self.assertTrue(np.all(qualities[:-1] >= qualities[1:] - 1e-12))

    def test_success_criterion_without_generalization_awareness(self):
        df = _opposite_signal_df(n=400, seed=8)
        table = _run_discovery(
            df, depth=1, result_set_size=0, generalization_awareness=False
        ).to_dataframe()
        self.assertNotIn("quality_ga", table.columns)
        self.assertFalse(bool(table.iloc[0]["interesting"]))
        subgroups = table.iloc[1:]
        self.assertGreater(len(subgroups), 0)
        pd.testing.assert_series_equal(
            subgroups["interesting"].astype(bool),
            (subgroups["quality_raw"] > 0).astype(bool),
            check_names=False,
        )

    def test_dataset_row_is_not_an_observed_parent(self):
        """The empty description is Q_ga baseline 0, not a recorded parent."""
        df = _opposite_signal_df(n=300, seed=9)
        result = _run_discovery(df, depth=1, result_set_size=0)
        qf = result.task.qf
        self.assertNotIn(frozenset(), qf.observed_raw_qualities)

    def test_execute_without_algorithm_returns_a_result(self):
        df = _opposite_signal_df(n=240, seed=9)
        qf = _lr_qf()
        target = ps.ModelAdaptabilityTarget(
            label_column="target", feature_columns=["a", "b", "c"]
        )
        task = ps.ModelAdaptabilityDiscoveryTask(
            data=df,
            target=target,
            search_space=ps.create_nominal_selectors_for_attribute(df, "a"),
            qf=qf,
            depth=1,
            result_set_size=5,
            constraints=[ps.MinSupportConstraint(20)],
        )
        result = task.execute()
        self.assertIsInstance(result, ps.ModelAdaptabilityDiscoveryResult)
        table = result.to_dataframe()
        self.assertEqual(str(table.iloc[0]["subgroup"]), "Dataset")
        self.assertGreater(len(table), 1)


class TestUnscorableCandidates(unittest.TestCase):
    """A candidate that cannot be scored must be reported, not silently dropped."""

    @staticmethod
    def _df_with_degenerate_region(n: int = 400, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"b": rng.normal(size=n), "c": rng.normal(size=n)})
        df["target"] = (df["b"] > 0).astype(int)
        df["a"] = 0
        df.loc[df.index[:20], "a"] = 1
        df.loc[df.index[:20], "target"] = 0
        df.loc[df.index[:1], "target"] = 1
        return df

    def _prepared_qf(self, df, **kwargs):
        qf = _lr_qf(**kwargs)
        target = ps.ModelAdaptabilityTarget(
            label_column="target", feature_columns=["b", "c"]
        )
        train = qf.resolve_search_data(df, target)
        qf.calculate_constant_statistics(train, target)
        return qf, target, train

    def test_quality_is_minus_inf_not_nan(self):
        """Failed scores are -inf, so the candidate stays in the result as a miss."""
        df = self._df_with_degenerate_region()
        qf, target, train = self._prepared_qf(df)
        sg = ps.Conjunction([ps.EqualitySelector("a", 1)])

        stats = qf.calculate_statistics(sg, target, train)
        quality = qf.evaluate(sg, target, train, stats)

        self.assertNotEqual(stats.status, "ok")
        self.assertFalse(np.isnan(quality))
        self.assertEqual(quality, -np.inf)
        self.assertGreater(sum(qf.failure_counts.values()), 0)
        self.assertFalse(meets_success_criterion(quality, status=stats.status))

    def test_failure_reason_is_recorded(self):
        df = self._df_with_degenerate_region()
        qf, target, train = self._prepared_qf(df)
        sg = ps.Conjunction([ps.EqualitySelector("a", 1)])
        stats = qf.calculate_statistics(sg, target, train)
        self.assertIn(stats.status, qf.failure_counts)
        self.assertEqual(qf.failure_counts[stats.status], 1)

    def test_unscorable_candidate_is_not_fitted_twice(self):
        df = self._df_with_degenerate_region()
        fits = {"n": 0}

        def counting_fit(model, X, y):
            fits["n"] += 1
            model.fit(X, y)

        qf, target, train = self._prepared_qf(df, training_global=counting_fit)
        sg = ps.Conjunction([ps.EqualitySelector("a", 1)])

        fits["n"] = 0
        stats = qf.calculate_statistics(sg, target, train)
        after_statistics = fits["n"]
        qf.evaluate(sg, target, train, stats)
        self.assertEqual(fits["n"], after_statistics)

    def test_min_size_per_split_is_enforced(self):
        """Min-support bounds |S|; scorability bounds each split (chapter 4: 5/5)."""
        rng = np.random.default_rng(7)
        n = 300
        df = pd.DataFrame({"b": rng.normal(size=n), "c": rng.normal(size=n)})
        df["target"] = (df["b"] > 0).astype(int)
        df["a"] = 0
        df.loc[df.index[:14], "a"] = 1

        qf, target, train = self._prepared_qf(df, min_size_train=10, min_size_test=10)
        sg = ps.Conjunction([ps.EqualitySelector("a", 1)])
        stats = qf.calculate_statistics(sg, target, train)
        self.assertIn(
            stats.status,
            {qf.STATUS_TOO_SMALL_TRAIN, qf.STATUS_TOO_SMALL_TEST},
        )
        qf2, target2, train2 = self._prepared_qf(df, min_size_train=1, min_size_test=1)
        stats2 = qf2.calculate_statistics(sg, target2, train2)
        self.assertGreater(stats2.size_sg_train, 0)

    def test_failure_report_reaches_the_result(self):
        df = self._df_with_degenerate_region(n=500, seed=2)
        result = _run_discovery(df, depth=1, min_support=10, result_set_size=0)
        self.assertTrue(result.evaluation_failures)
        self.assertNotEqual(result.failure_report(), "no evaluation failures")
        failed = result.to_dataframe()
        failed = failed.iloc[1:]
        if len(failed):
            bad = failed["quality_raw"].isna() | (failed["quality"] == -np.inf)
            if bad.any():
                self.assertFalse(failed.loc[bad, "interesting"].any())


class TestGeneralizationAwarenessIsUnbiased(unittest.TestCase):
    """``quality_ga`` must not depend on how many results are reported."""

    @staticmethod
    def _nested_effect_df(n: int = 1200, seed: int = 11) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"b": rng.normal(size=n), "c": rng.normal(size=n)})
        for name in ("a", "d", "e", "f"):
            df[name] = rng.integers(0, 3, n)
        lin = df["b"] + 0.5 * df["c"]
        lin = np.where(df["a"] == 1, -3 * df["b"], lin)
        lin = np.where((df["d"] == 0) & (df["e"] == 1), -3 * df["c"], lin)
        df["target"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        return df

    def _run(self, df, result_set_size):
        qf = _lr_qf()
        target = ps.ModelAdaptabilityTarget(
            label_column="target",
            feature_columns=["b", "c", "a", "d", "e", "f"],
        )
        search_space = []
        for attr in ("a", "d", "e", "f"):
            search_space += ps.create_nominal_selectors_for_attribute(df, attr)
        task = ps.ModelAdaptabilityDiscoveryTask(
            data=df,
            target=target,
            search_space=search_space,
            qf=qf,
            depth=2,
            result_set_size=result_set_size,
            constraints=[ps.MinSupportConstraint(20)],
            generalization_awareness=True,
        )
        result = task.execute(algorithm=ps.ParallelModelAdaptabilityDFS(max_workers=4))
        return result.to_dataframe()

    def test_quality_ga_matches_untrimmed_run(self):
        """Q_ga uses every evaluated candidate, not only the reported top-k."""
        df = self._nested_effect_df()
        full = self._run(df, result_set_size=0)
        trimmed = self._run(df, result_set_size=10)

        reference = {
            str(row["subgroup"]): float(row["quality_ga"])
            for _, row in full.iterrows()
        }
        compared = 0
        for _, row in trimmed.iloc[1:].iterrows():
            key = str(row["subgroup"])
            self.assertIn(key, reference)
            self.assertAlmostEqual(float(row["quality_ga"]), reference[key], places=9)
            compared += 1
        self.assertGreater(compared, 0)

    def test_quality_ga_is_formed_on_the_raw_scale(self):
        """Q_ga = Q_roh − max({0} ∪ {Q_roh(H)}); it is not Q_gew minus a parent."""
        df = self._nested_effect_df()
        table = self._run(df, result_set_size=0)
        subgroups = table.iloc[1:]
        raw = subgroups["quality_raw"].to_numpy(dtype=float)
        ga = subgroups["quality_ga"].to_numpy(dtype=float)
        self.assertTrue(np.all(ga <= raw + 1e-12))

        refinements = subgroups[subgroups["subgroup"].astype(str).str.contains(" AND ")]
        self.assertGreater(len(refinements), 0)
        self.assertTrue(
            (
                refinements["quality_ga"] < refinements["quality_raw"] - 1e-9
            ).any(),
            "no refinement was penalised against a generalization on the raw scale",
        )


class TestQualityFunctionHygiene(unittest.TestCase):
    def test_reusing_a_qf_retrains_the_global_model(self):
        df_a = _opposite_signal_df(n=300, seed=21)
        df_b = df_a.copy()
        df_b["target"] = 1 - df_b["target"]

        qf = _lr_qf()
        target = ps.ModelAdaptabilityTarget(
            label_column="target", feature_columns=["a", "b", "c"]
        )
        train_a = qf.resolve_search_data(df_a, target)
        qf.calculate_constant_statistics(train_a, target)
        self.assertTrue(qf.has_constant_statistics)

        qf.resolve_search_data(df_b, target)
        self.assertFalse(qf.has_constant_statistics)
        self.assertEqual(qf.failure_counts, {})
        self.assertEqual(qf.observed_raw_qualities, {})

    def test_non_monotone_constraints_are_enforced(self):
        class RejectEverything:
            is_monotone = False

            def is_satisfied(self, subgroup, statistics=None, data=None):
                return False

        df = _opposite_signal_df(n=300, seed=22)
        qf = _lr_qf()
        target = ps.ModelAdaptabilityTarget(
            label_column="target", feature_columns=["a", "b", "c"]
        )
        task = ps.ModelAdaptabilityDiscoveryTask(
            data=df,
            target=target,
            search_space=ps.create_nominal_selectors_for_attribute(df, "a"),
            qf=qf,
            depth=1,
            result_set_size=0,
            constraints=[ps.MinSupportConstraint(20), RejectEverything()],
        )
        result = task.execute(algorithm=ps.ParallelModelAdaptabilityDFS(max_workers=2))
        subgroups = [
            row
            for _, row in result.to_dataframe().iterrows()
            if str(row["subgroup"]) != "Dataset"
        ]
        self.assertEqual(subgroups, [])

    def test_singleton_minority_class_in_train_is_scorable(self):
        self.assertIsNone(
            scorability_status(
                size_train=5,
                size_test=5,
                n_classes_train=2,
                min_class_count_train=1,
                n_classes_test=2,
            )
        )
        self.assertEqual(
            scorability_status(
                size_train=5,
                size_test=5,
                n_classes_train=1,
                min_class_count_train=5,
                n_classes_test=2,
            ),
            "single_class_train",
        )

    def test_label_balance_fraction(self):
        self.assertAlmostEqual(
            _label_balance_fraction(pd.Series([0, 0, 1, 1])), 1.0, places=12
        )
        self.assertAlmostEqual(
            _label_balance_fraction(pd.Series([0, 0, 0, 1])), 1 / 3, places=12
        )
        self.assertAlmostEqual(
            _label_balance_fraction(pd.Series([1, 1, 1, 0])), 1 / 3, places=12
        )
        self.assertEqual(_label_balance_fraction(pd.Series([1, 1, 1])), 0)
        self.assertAlmostEqual(
            _label_balance_fraction(pd.Series([0, 1, 1, 1], index=[0, 0, 1, 1])),
            1 / 3,
            places=12,
        )

    def test_success_criterion(self):
        self.assertTrue(meets_success_criterion(0.02, 0.01))
        self.assertTrue(meets_success_criterion(0.02, 0.0))
        self.assertFalse(meets_success_criterion(0.02, -0.001))
        self.assertFalse(meets_success_criterion(0.0, 0.0))
        self.assertFalse(meets_success_criterion(-0.01, 0.0))
        self.assertTrue(meets_success_criterion(0.02, None))
        self.assertFalse(meets_success_criterion(0.02, 0.01, status="too_small_test"))


if __name__ == "__main__":
    unittest.main()
