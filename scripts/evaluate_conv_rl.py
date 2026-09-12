"""Evaluate direct neural decisions in the independent Python 2048 environment."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from bot2048.deep_rl import ConvAgent
from bot2048.env import Game2048
from scripts.evaluate import wilson


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/rl/conv_policy.pt")
    p.add_argument("--extra-checkpoint", action="append", default=[])
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=400000)
    p.add_argument("--output", default="results/rl/conv_development.json")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--voting", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    agent = ConvAgent(
        a.checkpoint,
        ensemble=a.ensemble or a.voting,
        voting=a.voting,
        extra_checkpoints=a.extra_checkpoint,
    )
    rows = []
    started = time.time()
    checkpoint_hash = hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    for index in range(a.games):
        seed = a.seed + index
        game = Game2048(seed)
        while not game.done and game.moves < 20000:
            game.step(agent.act(game.board))
        row = {
            "seed": seed,
            "won": game.won,
            "score": game.score,
            "moves": game.moves,
            "max_tile": 2 ** int(game.board.max()),
            "truncated": not game.done,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    wins = sum(r["won"] for r in rows)
    result = {
        "config": vars(a),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_metadata": agent.metadata,
        "search_at_inference": False,
        "games": rows,
        "summary": {
            "games": len(rows),
            "wins": wins,
            "wilson_95": wilson(wins, len(rows)),
            "mean_score": float(np.mean([r["score"] for r in rows])),
            "seconds": time.time() - started,
        },
    }
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
