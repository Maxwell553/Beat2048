"""Evaluate the CNN/sparse neural tie-break policy in independent Python 2048."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent
from bot2048.env import Game2048, legal_actions
from bot2048.rl import RLAgent
from scripts.evaluate import wilson
from scripts.evaluate_neural_tiebreak import actions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn", required=True)
    p.add_argument("--sparse", nargs="+", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=400000)
    p.add_argument("--device", default="mps")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    rows = []
    started = time.time()
    for index in range(args.games):
        game = Game2048(args.seed + index)
        while not game.done and game.moves < 20000:
            legal = np.zeros((1, 4), dtype=np.uint8)
            legal[0, legal_actions(game.board)] = 1
            action = int(actions(cnn, sparse, game.board[None], legal)[0])
            game.step(action)
        rows.append(
            {
                "seed": args.seed + index,
                "won": game.won,
                "score": game.score,
                "moves": game.moves,
                "max_tile": 2 ** int(game.board.max()),
                "truncated": not game.done,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    for model in sparse:
        model.close()
    wins = sum(row["won"] for row in rows)
    result = {
        "config": vars(args),
        "cnn_sha256": hashlib.sha256(Path(args.cnn).read_bytes()).hexdigest(),
        "sparse_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in args.sparse
        },
        "environment": "independent Python standard 2048; two initial tiles; 90% 2 / 10% 4",
        "policy": "CNN neural symmetry vote; direct sparse neural tie-break only",
        "search_at_inference": False,
        "games": rows,
        "summary": {
            "games": len(rows),
            "wins": wins,
            "wilson_95": wilson(wins, len(rows)),
            "mean_score": float(np.mean([row["score"] for row in rows])),
            "seconds": time.time() - started,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
