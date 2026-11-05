"""Original synthetic offer log. Oracle outcomes stay outside the feature matrix."""
from pathlib import Path
import csv
import numpy as np

FEATURES = ["distance_km", "basket_items", "backlog", "hour_sin", "hour_cos", "store_friction"]
COST = np.array([0.0, 0.035, 0.09, 0.20])


def generate(seed=1602, days=60, zones=4, events=40):
    rng = np.random.default_rng(seed)
    rows, oracle = [], []
    for day in range(days):
        for zone in range(zones):
            group_shock = rng.normal(0, 0.32)
            split = "train" if day < days // 2 else "validation" if day < days * 3 // 4 else "test"
            for j in range(events):
                distance = rng.uniform(0.5, 12)
                items = int(rng.integers(3, 50))
                backlog = rng.uniform(0, 1)
                hour = rng.uniform(7, 22)
                friction = rng.uniform(0, 1)
                x = [distance, items, backlog, np.sin(hour * np.pi / 12), np.cos(hour * np.pi / 12), friction]
                # Randomized logging conditional on recorded context. No fourth-action exploration.
                p = np.array([0.35, 0.45, 0.20, 0.0]) if backlog > 0.55 else np.array([0.70, 0.23, 0.07, 0.0])
                action = int(rng.choice(4, p=p))
                base = 1.9 - 0.10 * distance - 0.015 * items - 1.1 * backlog - 0.5 * friction + group_shock
                responsiveness = 0.35 + 1.35 * backlog + 0.06 * distance
                logits = base + np.array([0, 0.65, 1.0, 1.3]) * responsiveness
                completion_prob = 1 / (1 + np.exp(-logits))
                complete = int(rng.random() < completion_prob[action])
                row = dict(event_id=f"d{day:02d}-z{zone}-{j:02d}", group=f"d{day:02d}-z{zone}", day=day, zone=zone, split=split)
                row.update(dict(zip(FEATURES, x)))
                row.update(action=action, reward=complete - COST[action], completed=complete)
                row.update({f"p{a}": v for a, v in enumerate(p)})
                rows.append(row)
                oracle.append(dict(event_id=row["event_id"], **{f"expected_reward_{a}": completion_prob[a] - COST[a] for a in range(4)}))
    return rows, oracle


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "x": np.array([[float(r[c]) for c in FEATURES] for r in rows]),
        "p": np.array([[float(r[f"p{a}"]) for a in range(4)] for r in rows]),
        "a": np.array([int(r["action"]) for r in rows]),
        "r": np.array([float(r["reward"]) for r in rows]),
        "group": np.array([r["group"] for r in rows]),
        "split": np.array([r["split"] for r in rows]),
        "ids": np.array([r["event_id"] for r in rows]),
    }


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}
