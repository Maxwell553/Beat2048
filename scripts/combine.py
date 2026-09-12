"""Combine episode-disjoint teacher and learner datasets."""

import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs", nargs="+", default=["data/teacher.npz", "data/dagger.npz"]
    )
    p.add_argument("--output", default="data/combined.npz")
    a = p.parse_args()
    parts = [np.load(path) for path in a.inputs]
    seen = set()
    for part in parts:
        ids = set(np.unique(part["episodes"]).tolist())
        if seen & ids:
            raise ValueError("Episode IDs overlap: use distinct collection seed ranges")
        seen.update(ids)
    np.savez_compressed(
        a.output,
        **{
            k: np.concatenate([p[k] for p in parts])
            for k in ["boards", "scores", "episodes"]
        },
    )
    print(f"Saved {sum(len(p['boards']) for p in parts)} states to {a.output}")


if __name__ == "__main__":
    main()
