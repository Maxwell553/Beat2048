"""Parallel teacher self-play. Episode IDs define leakage-free train/validation splits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import numpy as np
from bot2048.env import Game2048
from bot2048.search import Planner


def episode(task):
    seed, depth, cutoff, explore, checkpoint = task
    planner = Planner(depth, cutoff)
    game = Game2048(seed)
    rng = np.random.default_rng(seed + 98765)
    boards = []
    scores = []
    learner = None
    if checkpoint:
        import torch

        torch.set_num_threads(1)
        from bot2048.model import Agent

        learner = Agent(checkpoint, mode="neural", ensemble=False)
    while not game.done and game.moves < 20000:
        s = planner.scores(game.board)
        boards.append(game.board.copy())
        scores.append(np.where(s < -1e90, -np.inf, s))
        legal = np.flatnonzero(s > -1e90)
        action = (
            int(rng.choice(legal))
            if rng.random() < explore
            else (learner.act(game.board) if learner else int(np.argmax(s)))
        )
        game.step(action)
    return (
        seed,
        np.asarray(boards, dtype=np.uint8),
        np.asarray(scores, dtype=np.float32),
        {
            "seed": seed,
            "won": game.won,
            "score": game.score,
            "moves": game.moves,
            "max_tile": 2 ** int(game.board.max()),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=96)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--explore", type=float, default=0.015)
    p.add_argument("--output", default="data/teacher.npz")
    p.add_argument(
        "--checkpoint", help="Collect learner rollouts labeled by search (DAgger)"
    )
    a = p.parse_args()
    Planner(a.depth, a.cutoff)  # Build before workers start.
    start = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [
            pool.submit(
                episode, (a.seed + i, a.depth, a.cutoff, a.explore, a.checkpoint)
            )
            for i in range(a.games)
        ]
        for f in as_completed(futures):
            record = f.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "total": a.games,
                        "elapsed_s": round(time.time() - start, 1),
                        **record[3],
                    }
                ),
                flush=True,
            )
    records.sort(key=lambda r: r[0])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        boards=np.concatenate([r[1] for r in records]),
        scores=np.concatenate([r[2] for r in records]),
        episodes=np.concatenate(
            [np.full(len(r[1]), r[0], dtype=np.int32) for r in records]
        ),
    )
    Path(a.output).with_suffix(".json").write_text(
        json.dumps(
            {
                "config": vars(a),
                "seconds": time.time() - start,
                "episodes": [r[3] for r in records],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
