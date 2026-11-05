"""Generate data, select a policy on validation only, then evaluate frozen test."""
from pathlib import Path
import csv
import hashlib
import json
import pickle
import platform
import sys
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fulfillment_policy.data import FEATURES, generate, write_csv, load, subset
from fulfillment_policy.model import POLICIES, fit, predict_all, grouped_validation_predictions, policy
from fulfillment_policy.evaluation import estimate, cluster_interval, UnsupportedPolicy


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    rows, oracle = generate()
    write_csv(ROOT / "data/logged_offers.csv", rows)
    write_csv(ROOT / "data/oracle_diagnostic_only.csv", oracle)
    data = load(ROOT / "data/logged_offers.csv")
    train = subset(data, data["split"] == "train")
    val = subset(data, data["split"] == "validation")
    test = subset(data, data["split"] == "test")
    assert not (set(train["group"]) & set(val["group"]) | set(val["group"]) & set(test["group"]) | set(train["group"]) & set(test["group"]))
    qval, audit = grouped_validation_predictions(train, val)
    validation = {}
    for name in POLICIES:
        result, scores = estimate(val["p"], policy(val["x"], name), val["a"], val["r"], qval)
        result["dr_cluster_ci_95"] = cluster_interval(scores, val["group"])
        validation[name] = result
    # Selection is defined before reading test outcomes. No oracle access here.
    selected = max(POLICIES, key=lambda name: validation[name]["dr"])
    selection = {"selected_policy": selected, "criterion": "highest validation DR point estimate among five predefined supported policies", "candidate_policies": POLICIES, "test_outcomes_used": False, "oracle_used": False, "results": validation}
    dump(ROOT / "results/selection.json", selection)
    model_data = subset(data, data["split"] != "test")
    model = fit(model_data)
    with (ROOT / "models/reward_model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=4)
    qtest = predict_all(model, test["x"])
    # All candidates are reported descriptively; selected policy stays frozen.
    test_results, all_scores = {}, {}
    for name in POLICIES:
        result, scores = estimate(test["p"], policy(test["x"], name), test["a"], test["r"], qtest)
        result["dr_cluster_ci_95"] = cluster_interval(scores, test["group"])
        test_results[name], all_scores[name] = result, scores
    baseline = "no_incentive"
    paired = all_scores[selected] - all_scores[baseline]
    try:
        estimate(test["p"], policy(test["x"], "unsupported_premium"), test["a"], test["r"], qtest)
        raise AssertionError("unsupported policy was accepted")
    except UnsupportedPolicy as exc:
        unsupported = {"status": "refused", "reason": str(exc), "estimate": None}
    # Isolated oracle access occurs only after selection/test estimation.
    oracle_test = np.array([[float(row[f"expected_reward_{a}"]) for a in range(4)] for row, is_test in zip(oracle, data["split"] == "test") if is_test])
    oracle_values = {name: float(np.mean(np.sum(policy(test["x"], name) * oracle_test, axis=1))) for name in POLICIES}
    scores_rows = []
    for i, event_id in enumerate(test["ids"]):
        scores_rows.append({"event_id": event_id, "group": test["group"][i], **{f"dr_score_{name}": float(all_scores[name][i]) for name in POLICIES}})
    write_csv(ROOT / "results/test_scores.csv", scores_rows)
    result = {
        "experiment": "independent synthetic fulfillment offer simulation; not a workplace result",
        "seed": 1602, "events": len(data["a"]),
        "split_events": {s: int(np.sum(data["split"] == s)) for s in ["train", "validation", "test"]},
        "split_groups": {s: len(np.unique(data["group"][data["split"] == s])) for s in ["train", "validation", "test"]},
        "selected_policy": selected, "selection_frozen_before_test": True,
        "reward": "completion indicator minus fixed normalized incentive cost; arbitrary utility units, not dollars",
        "features": FEATURES, "reward_model": "HistGradientBoostingRegressor: 80 iterations, 15 leaves, L2=3, minimum leaf=25",
        "crossfit": "four validation group folds with historical train groups always available; final nuisance model fits train+validation only",
        "bootstrap": {"unit": "zone-day", "repetitions": 1000, "seed": 42, "type": "paired percentile cluster bootstrap with fixed fitted nuisance; not selection-adjusted"},
        "test": test_results,
        "selected_vs_no_incentive": {"dr_difference": float(paired.mean()), "dr_difference_cluster_ci_95": cluster_interval(paired, test["group"])},
        "unsupported_policy": unsupported,
        "oracle_diagnostic_only": {"expected_utility": oracle_values, "selected_policy_abs_dr_error": abs(test_results[selected]["dr"] - oracle_values[selected]), "used_for_training_or_selection": False},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
    }
    dump(ROOT / "results/metrics.json", result)
    dump(ROOT / "results/crossfit_audit.json", audit)
    frozen = ["data/logged_offers.csv", "data/oracle_diagnostic_only.csv", "models/reward_model.pkl", "results/selection.json", "results/test_scores.csv", "results/metrics.json", "results/crossfit_audit.json"]
    manifest = {"sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in frozen}, "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]}}
    dump(ROOT / "results/manifest.json", manifest)
    print(json.dumps({"selected": selected, "test": test_results[selected], "difference": result["selected_vs_no_incentive"], "unsupported": unsupported}, indent=2))


if __name__ == "__main__":
    main()
