"""Reward nuisance fitting: no oracle columns, group-disjoint validation predictions."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


def design(x, action):
    return np.column_stack([x, np.eye(4)[action]])


def fit(data):
    model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=3.0, random_state=1602)
    return model.fit(design(data["x"], data["a"]), data["r"])


def predict_all(model, x):
    return np.column_stack([model.predict(design(x, np.full(len(x), a, dtype=int))) for a in range(4)])


def grouped_validation_predictions(train, validation, n_splits=4):
    q = np.empty((len(validation["x"]), 4))
    audit = []
    for fold, (fit_idx, held_idx) in enumerate(GroupKFold(n_splits).split(validation["x"], groups=validation["group"])):
        fit_groups = set(validation["group"][fit_idx]) | set(train["group"])
        held_groups = set(validation["group"][held_idx])
        if fit_groups & held_groups:
            raise ValueError("cross-fit group leakage")
        fit_data = {k: np.concatenate([train[k], validation[k][fit_idx]]) for k in train}
        model = fit(fit_data)
        q[held_idx] = predict_all(model, validation["x"][held_idx])
        audit.append({"fold": fold, "fit_groups": sorted(fit_groups), "heldout_groups": sorted(held_groups), "fit_events": len(fit_data["a"]), "heldout_events": len(held_idx)})
    return q, audit


def policy(x, name):
    n = len(x)
    backlog, distance = x[:, 2], x[:, 0]
    if name == "no_incentive":
        a = np.zeros(n, dtype=int)
    elif name == "modest_offer":
        a = np.ones(n, dtype=int)
    elif name == "high_offer":
        a = np.full(n, 2, dtype=int)
    elif name == "load_sensitive":
        a = np.where(backlog > 0.65, 2, np.where((backlog > 0.30) | (distance > 8), 1, 0))
    elif name == "distance_sensitive":
        a = np.where(distance > 8, 2, np.where(distance > 4, 1, 0))
    elif name == "unsupported_premium":
        a = np.full(n, 3, dtype=int)
    else:
        raise ValueError(f"unknown policy: {name}")
    return np.eye(4)[a]


POLICIES = ["no_incentive", "modest_offer", "high_offer", "load_sensitive", "distance_sensitive"]
