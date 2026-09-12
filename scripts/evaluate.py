"""Fixed-seed, paired evaluation with Wilson binomial confidence intervals."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import math


def wilson(wins, total):
    if total == 0:
        return [0.0, 1.0]
    z = 1.95996398454
    p = wins / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [mid - half, mid + half]


def play(task):
    mode, checkpoint, seed, depth, cutoff, weight, record = task
    import torch

    torch.set_num_threads(1)
    from bot2048.env import Game2048
    from bot2048.model import Agent

    agent = Agent(
        checkpoint,
        "neural" if mode == "initial" else mode,
        seed + 314159,
        depth,
        cutoff,
        weight,
    )
    game = Game2048(seed)
    trajectory = []
    started = time.time()
    while not game.done and game.moves < 20000:
        action = agent.act(game.board)
        if record:
            trajectory.append(
                {
                    "board": game.board.tolist(),
                    "score": game.score,
                    "move": game.moves,
                    "action": action,
                }
            )
        game.step(action)
    if record:
        trajectory.append(
            {
                "board": game.board.tolist(),
                "score": game.score,
                "move": game.moves,
                "action": None,
            }
        )
    row = {
        "mode": mode,
        "seed": seed,
        "won": game.won,
        "score": game.score,
        "moves": game.moves,
        "max_tile": 2 ** int(game.board.max()),
        "seconds": time.time() - started,
        "truncated": not game.done,
    }
    if record:
        row["trajectory"] = trajectory
    return row


def summarize(rows):
    import numpy as np

    wins = sum(r["won"] for r in rows)
    n = len(rows)
    return {
        "games": n,
        "wins": wins,
        "win_rate": wins / n,
        "wilson_95": wilson(wins, n),
        "mean_score": float(np.mean([r["score"] for r in rows])),
        "mean_moves": float(np.mean([r["moves"] for r in rows])),
        "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
        "max_tile_histogram": {
            str(t): sum(r["max_tile"] == t for r in rows)
            for t in sorted(set(r["max_tile"] for r in rows))
        },
        "truncated": sum(r["truncated"] for r in rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modes",
        nargs="+",
        default=["random", "initial", "neural", "search", "hybrid"],
    )
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=200000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--cutoff", type=float, default=0.0001)
    p.add_argument("--neural-weight", type=float, default=0.02)
    p.add_argument("--checkpoint", default="models/final_policy.pt")
    p.add_argument("--output", default="results/evaluation.json")
    a = p.parse_args()
    from bot2048.search import Planner

    Planner(a.depth, a.cutoff)
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [
            pool.submit(
                play,
                (
                    m,
                    (
                        "models/initial_policy.pt"
                        if m == "initial"
                        else a.checkpoint if m in ("neural", "hybrid") else None
                    ),
                    a.seed + i,
                    a.depth,
                    a.cutoff,
                    a.neural_weight,
                    False,
                ),
            )
            for m in a.modes
            for i in range(a.games)
        ]
        for f in as_completed(futures):
            row = f.result()
            rows.append(row)
            if len(rows) % 10 == 0:
                print(
                    json.dumps(
                        {"completed": len(rows), "total": len(futures), "last": row}
                    ),
                    flush=True,
                )
    import hashlib, platform
    import numpy, torch

    provenance = {
        "checkpoint_sha256": (
            hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
            if Path(a.checkpoint).exists()
            else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "symmetry_ensemble": True,
        "immediate_win_check": True,
        "policy_seed_rule": "environment_seed + 314159",
    }
    result = {
        "provenance": provenance,
        "config": vars(a),
        "seconds": time.time() - started,
        "summary": {m: summarize([r for r in rows if r["mode"] == m]) for m in a.modes},
        "games": sorted(rows, key=lambda r: (r["mode"], r["seed"])),
    }
    Path(a.output).parent.mkdir(exist_ok=True, parents=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
