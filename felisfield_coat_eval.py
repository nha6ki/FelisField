"""Evaluate the existing FelisField embedding without opening any photographs."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

K = 15


def load_map(folder):
    rows = json.loads((folder / "points.json").read_text())
    coords = np.array([[r[k] for k in ("x", "y", "z")] for r in rows])
    sources = np.array(["commons" if r["source"] == "commons_initial20" else r["source"] for r in rows])
    return rows, coords, sources


def source_metrics(coords, sources):
    indices = cKDTree(coords).query(coords, k=K + 1)[1]
    same = 0
    for i, near in enumerate(indices):
        same += np.count_nonzero(sources[near[near != i][:K]] == sources[i])
    n = len(coords)
    counts = np.unique(sources, return_counts=True)[1]
    return {"same_source_3d": same / (n * K),
            "same_source_random_expected": float(counts @ (counts - 1)) / (n * (n - 1))}


def compare(coords, features, sources, cross_source):
    # The first 64 columns are sqrt(0.5) times the square-root histogram.
    # Euclidean distance on these columns equals the Hellinger distance.
    colors = features[:, :64]
    totals = np.zeros(3)
    rng = np.random.default_rng(42)
    for start in range(0, len(coords), 48):
        queries = np.arange(start, min(start + 48, len(coords)))
        distances = [cdist(a[queries], a, "sqeuclidean") for a in (coords, features)]
        for j, i in enumerate(queries):
            eligible = sources != sources[i] if cross_source else np.ones(len(coords), dtype=bool)
            eligible[i] = False
            pool = np.flatnonzero(eligible)
            neighbors = []
            for d in distances:
                d[j, ~eligible] = np.inf
                neighbors.append(np.argpartition(d[j], K - 1)[:K])
            neighbors.append(rng.choice(pool, K, replace=False))
            for k, idx in enumerate(neighbors):
                totals[k] += np.linalg.norm(colors[idx] - colors[i], axis=1).mean()
    return dict(zip(("map_3d", "features_114d", "random"), totals / len(coords)))


def main():
    folder = Path.home() / "projects/FelisField/results/coat_full50"
    rows, coords, sources = load_map(folder)
    features = np.load(folder / "combined_features.npy", allow_pickle=False)
    ids = [r["id"] for r in rows]
    if (ids != json.loads((folder / "row_ids.json").read_text())
            or len(set(ids)) != len(rows) or features.shape != (len(rows), 114)
            or not np.isfinite(features).all() or not np.isfinite(coords).all()
            or (features[:, :64] < 0).any()
            or not np.allclose((features[:, :64] ** 2).sum(axis=1), .5, atol=1e-5)):
        raise ValueError("Map rows and 50:50 features failed validation")
    if len(rows) - np.unique(sources, return_counts=True)[1].max() < K:
        raise ValueError("Too few cross-source candidates")
    report = {"photos": len(rows), "k": K, "seed": 42, **source_metrics(coords, sources)}
    for cross, name in [(False, "all_sources"), (True, "other_sources_only")]:
        print(f"Evaluating {name} ...", flush=True)
        report[name] = compare(coords, features, sources, cross)
    output = folder / "coat_neighbor_evaluation.json"
    result = json.dumps(report, indent=2, allow_nan=False)
    output.write_text(result + "\n")
    print(result, output, sep="\n")


if __name__ == "__main__":
    main()
