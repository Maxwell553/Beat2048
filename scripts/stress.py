"""Arbitrary-start audit, deliberately including a mathematically lost board."""

import argparse
import json
from pathlib import Path
import numpy as np
from bot2048.env import Game2048, legal_actions
from bot2048.model import Agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/final_policy.pt")
    p.add_argument("--depth", type=int, default=4)
    a = p.parse_args()
    import torch

    torch.set_num_threads(1)
    starts = []
    # Reachable boards from independent random play; no filtering by eventual bot outcome.
    for i in range(16):
        seed = 300000 + i
        g = Game2048(seed)
        rng = np.random.default_rng(seed + 314159)
        for _ in range(30 + 10 * i):
            if g.done:
                break
            g.step(int(rng.choice(legal_actions(g.board))))
        starts.append((f"random_prefix_{i}", seed, g.board.copy()))
    lost = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])
    starts.append(("already_lost_checkerboard", 399990, lost))
    near = np.zeros((4, 4), dtype=int)
    near[0, :2] = 10
    starts.append(("two_1024_tiles", 399991, near))
    won = np.zeros((4, 4), dtype=int)
    won[0, 0] = 11
    starts.append(("already_won", 399992, won))
    rows = []
    agent = Agent(a.checkpoint, depth=a.depth)
    for name, seed, board in starts:
        g = Game2048(seed, board)
        initial_done = g.done
        initial_won = g.won
        while not g.done and g.moves < 20000:
            g.step(agent.act(g.board))
        row = {
            "name": name,
            "seed": seed,
            "start_exponents": board.tolist(),
            "initial_done": initial_done,
            "initial_won": initial_won,
            "won": g.won,
            "moves": g.moves,
            "score": g.score,
            "max_tile": 2 ** int(g.board.max()),
            "truncated": not g.done,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path("results/stress.json").write_text(
        json.dumps({"config": vars(a), "games": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
