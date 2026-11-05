"""Support-gated estimators. Gates refuse estimates; they do not repair overlap."""
from numbers import Real
import numpy as np


class UnsupportedPolicy(ValueError):
    pass


def _finite_matrix(value, shape, name):
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return arr


def weights(logging, target, actions, min_ess=30, max_weight=20):
    p = np.asarray(logging, dtype=float)
    if p.ndim != 2 or len(p) == 0:
        raise ValueError("logging must be a nonempty matrix")
    pi = _finite_matrix(target, p.shape, "target")
    p = _finite_matrix(p, p.shape, "logging")
    if (p < 0).any() or (pi < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-10, rtol=0) or not np.allclose(pi.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError("probabilities must be nonnegative and rows sum to one")
    a = np.asarray(actions)
    if a.shape != (len(p),) or not np.issubdtype(a.dtype, np.integer) or (a < 0).any() or (a >= p.shape[1]).any():
        raise ValueError("actions must be valid integer indices")
    if not np.isfinite(min_ess) or isinstance(min_ess, bool) or min_ess < 0:
        raise ValueError("min_ess must be finite and nonnegative")
    if not np.isfinite(max_weight) or isinstance(max_weight, bool) or max_weight <= 0:
        raise ValueError("max_weight must be finite and positive")
    if ((pi > 0) & (p == 0)).any():
        raise UnsupportedPolicy("target assigns mass to an action with zero logging probability")
    observed = p[np.arange(len(p)), a]
    if (observed <= 0).any():
        raise ValueError("logged action has zero probability")
    w = pi[np.arange(len(p)), a] / observed
    if not np.any(w):
        raise UnsupportedPolicy("no logged actions match the target policy")
    # Check all target-supported context/action pairs, not merely sampled actions.
    possible_ratio = np.divide(pi, p, out=np.zeros_like(pi), where=p > 0)
    largest = float(possible_ratio.max())
    ess = float(w.sum() ** 2 / (w @ w)) if np.any(w) else 0.0
    if largest > max_weight:
        raise UnsupportedPolicy(f"maximum possible importance ratio {largest:.3f} exceeds {max_weight}")
    if ess < min_ess:
        raise UnsupportedPolicy(f"effective sample size {ess:.1f} below {min_ess}")
    return w, {"ess": ess, "max_possible_weight": largest, "matched_events": int(np.count_nonzero(w))}


def estimate(logging, target, actions, reward, qhat, clip=5.0, min_ess=30, max_weight=20):
    if isinstance(clip, bool) or not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    w, diagnostics = weights(logging, target, actions, min_ess, max_weight)
    n = len(w)
    r = _finite_matrix(reward, (n,), "reward")
    q = _finite_matrix(qhat, np.asarray(logging).shape, "qhat")
    pi = np.asarray(target)
    correction = w * (r - q[np.arange(n), actions])
    direct = (pi * q).sum(1)
    scores = direct + correction
    return {
        "ips": float(np.mean(w * r)),
        "snips": float(np.sum(w * r) / w.sum()),
        "clipped_ips": float(np.mean(np.minimum(w, clip) * r)),
        "dr": float(scores.mean()),
        "diagnostics": {**diagnostics, "weight_sum_per_event": float(w.mean()), "clip": clip, "clipped_fraction": float(np.mean(w > clip))},
    }, scores


def cluster_interval(scores, groups, seed=42, repetitions=1000):
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=object)
    if scores.ndim != 1 or groups.ndim != 1 or len(scores) != len(groups) or not np.isfinite(scores).all():
        raise ValueError("scores/groups must be aligned and finite")
    string_ids = all(isinstance(g, str) and bool(g.strip()) for g in groups)
    numeric_ids = all(isinstance(g, Real) and np.isfinite(g) for g in groups)
    if not (string_ids or numeric_ids):
        raise ValueError("group ids must be nonempty strings or finite numeric ids without missing values")
    unique = np.unique(groups)
    if len(unique) < 10:
        raise ValueError("at least ten independent groups required")
    if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)) or repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions required")
    sums = np.array([scores[groups == g].sum() for g in unique])
    counts = np.array([np.sum(groups == g) for g in unique])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    means = sums[picks].sum(1) / counts[picks].sum(1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]
