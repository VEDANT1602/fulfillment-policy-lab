from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fulfillment_policy.data import FEATURES, generate, load, subset
from fulfillment_policy.evaluation import estimate, weights, cluster_interval, UnsupportedPolicy
from fulfillment_policy.model import policy, grouped_validation_predictions


class EstimatorContracts(unittest.TestCase):
    def setUp(self):
        self.p = np.tile([0.5, 0.5], (100, 1))
        self.pi = np.tile([0, 1.0], (100, 1))
        self.a = np.tile([0, 1], 50)
        self.r = self.a * 0.8
        self.q = np.tile([0, 0.8], (100, 1))

    def test_balanced_known_answer_estimators(self):
        result, _ = estimate(self.p, self.pi, self.a, self.r, self.q)
        for key in ["ips", "snips", "clipped_ips", "dr"]:
            self.assertAlmostEqual(result[key], 0.8)

    def test_exact_reward_model_cancels_sampling_noise(self):
        a = np.zeros(100, dtype=int)
        a[:10] = 1
        result, scores = estimate(self.p, self.pi, a, a * 0.8, self.q, min_ess=1)
        np.testing.assert_allclose(scores, 0.8)
        self.assertNotAlmostEqual(result["ips"], 0.8)

    def test_zero_support_checked_for_unobserved_actions(self):
        p = np.tile([1.0, 0.0], (100, 1))
        with self.assertRaises(UnsupportedPolicy):
            weights(p, self.pi, np.zeros(100, dtype=int))

    def test_weight_cap_checked_before_observed_match(self):
        p = np.tile([0.999, 0.001], (100, 1))
        a = np.zeros(100, dtype=int)
        a[0] = 1
        with self.assertRaisesRegex(UnsupportedPolicy, "maximum possible"):
            weights(p, self.pi, a, min_ess=0)

    def test_no_matching_events_cannot_create_nan(self):
        with self.assertRaises(UnsupportedPolicy):
            estimate(self.p, self.pi, np.zeros(100, dtype=int), self.r, self.q, min_ess=0)

    def test_low_ess_refused(self):
        a = np.zeros(100, dtype=int)
        a[0] = 1
        with self.assertRaisesRegex(UnsupportedPolicy, "effective sample"):
            weights(self.p, self.pi, a)

    def test_clipping_reports_bias_sensitive_change(self):
        p = np.tile([0.9, 0.1], (100, 1))
        a = np.zeros(100, dtype=int)
        a[:10] = 1
        result, _ = estimate(p, self.pi, a, a * 0.8, self.q, min_ess=1, clip=5)
        self.assertAlmostEqual(result["ips"], 0.8)
        self.assertAlmostEqual(result["clipped_ips"], 0.4)
        self.assertAlmostEqual(result["diagnostics"]["clipped_fraction"], 0.1)

    def test_nonfinite_inputs_rejected(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value):
                q = self.q.copy()
                q[0, 0] = value
                with self.assertRaises(ValueError):
                    estimate(self.p, self.pi, self.a, self.r, q)
                with self.assertRaises(ValueError):
                    estimate(self.p, self.pi, self.a, self.r, self.q, clip=value)
                p = self.p.copy()
                p[0, 0] = value
                with self.assertRaises(ValueError):
                    weights(p, self.pi, self.a)

    def test_invalid_probability_mass_rejected(self):
        for row in [[-0.01, 1.01], [0.5, 0.500001]]:
            p = self.p.copy()
            p[0] = row
            with self.assertRaises(ValueError):
                weights(p, self.pi, self.a)

    def test_action_indices_not_silently_cast(self):
        with self.assertRaises(ValueError):
            weights(self.p, self.pi, self.a.astype(float))
        bad = self.a.copy()
        bad[0] = 2
        with self.assertRaises(ValueError):
            weights(self.p, self.pi, bad)

    def test_cluster_replicates_whole_groups(self):
        groups = np.repeat(np.arange(20), 10)
        scores = np.repeat(np.arange(20, dtype=float), 10)
        ci = cluster_interval(scores, groups)
        # Duplicating each row leaves equal-sized cluster bootstrap unchanged.
        self.assertEqual(ci, cluster_interval(np.repeat(scores, 2), np.repeat(groups, 2)))
        self.assertLess(ci[0], scores.mean())
        self.assertGreater(ci[1], scores.mean())

    def test_cluster_guard_rejects_too_few_groups(self):
        with self.assertRaises(ValueError):
            cluster_interval(np.ones(20), np.zeros(20))
        with self.assertRaises(ValueError):
            cluster_interval(np.ones(20), np.arange(20).reshape(-1, 1))

    def test_missing_group_cannot_become_empty_bootstrap_cluster(self):
        for missing in [float("nan"), float("inf"), -float("inf"), None, ""]:
            with self.subTest(missing=missing):
                groups = np.arange(20).astype(object)
                groups[0] = missing
                with self.assertRaises(ValueError):
                    cluster_interval(np.ones(20), groups)
        labels = np.array([f"zone-day-{i}" for i in range(20)])
        self.assertEqual(cluster_interval(np.ones(20), labels), [1.0, 1.0])


class ExperimentContracts(unittest.TestCase):
    def test_generated_log_preserves_randomization_support(self):
        rows, oracle = generate(days=8, zones=2, events=5)
        self.assertEqual(len(rows), 80)
        self.assertEqual([r["event_id"] for r in rows], [r["event_id"] for r in oracle])
        for row in rows:
            self.assertGreater(row[f"p{row['action']}"], 0)
            self.assertEqual(row["p3"], 0)
        self.assertFalse(any("expected" in name for name in FEATURES))

    def test_crossfit_audit_has_no_group_leakage(self):
        audit = json.loads((ROOT / "results/crossfit_audit.json").read_text())
        covered = []
        for fold in audit:
            self.assertFalse(set(fold["fit_groups"]) & set(fold["heldout_groups"]))
            covered.extend(fold["heldout_groups"])
        self.assertEqual(len(covered), 60)
        self.assertEqual(len(set(covered)), 60)
        self.assertTrue(all(int(g[1:3]) < 45 for g in covered))

    def test_split_groups_and_feature_contract(self):
        data = load(ROOT / "data/logged_offers.csv")
        groups = {s: set(data["group"][data["split"] == s]) for s in ["train", "validation", "test"]}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(data["x"].shape, (9600, 6))

    def test_selection_is_validation_only_and_not_replaced_after_test(self):
        selection = json.loads((ROOT / "results/selection.json").read_text())
        metrics = json.loads((ROOT / "results/metrics.json").read_text())
        expected = max(selection["candidate_policies"], key=lambda name: selection["results"][name]["dr"])
        self.assertEqual(selection["selected_policy"], expected)
        self.assertEqual(metrics["selected_policy"], expected)
        self.assertFalse(selection["test_outcomes_used"])
        self.assertFalse(selection["oracle_used"])
        self.assertIsNone(metrics["unsupported_policy"]["estimate"])

    def test_all_artifact_and_source_digests_match(self):
        manifest = json.loads((ROOT / "results/manifest.json").read_text())
        for values in manifest.values():
            for path, expected in values.items():
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)


if __name__ == "__main__":
    unittest.main()
