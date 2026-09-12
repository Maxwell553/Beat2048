"""Evaluate the neural fusion policy in the independent Python environment."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent, FusionPolicy
from bot2048.env import Game2048, legal_actions
from bot2048.rl import RLAgent
from scripts.evaluate import wilson
from scripts.evaluate_fusion_policy import fusion_actions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn", required=True)
    p.add_argument("--sparse", nargs="+", required=True)
    p.add_argument("--expert", nargs="*", default=[])
    p.add_argument("--fusion", nargs="+", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=400000)
    p.add_argument("--device", default="mps")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    experts = [ConvAgent(path, args.device) for path in args.expert]
    fusion = []
    for path in args.fusion:
        artifact = torch.load(path, map_location=args.device, weights_only=True)
        model = FusionPolicy(
            artifact["metadata"]["inputs"], artifact["metadata"]["hidden"]
        ).to(args.device)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        fusion.append(model)
    rows = []; started = time.time()
    for index in range(args.games):
        game = Game2048(args.seed + index)
        while not game.done and game.moves < 20000:
            legal = np.zeros((1, 4), dtype=np.uint8)
            legal[0, legal_actions(game.board)] = 1
            action = int(fusion_actions(
                cnn, sparse, fusion, game.board[None], legal, args.device,
                experts=experts,
            )[0])
            game.step(action)
        row = {"seed": args.seed + index, "won": game.won, "score": game.score, "moves": game.moves, "max_tile": 2 ** int(game.board.max()), "truncated": not game.done}
        rows.append(row); print(json.dumps(row), flush=True)
    for model in sparse: model.close()
    wins = sum(row["won"] for row in rows)
    result = {
        "config": vars(args),
        "hashes": {
            "cnn": hashlib.sha256(Path(args.cnn).read_bytes()).hexdigest(),
            "fusion": {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in args.fusion
            },
            "sparse": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.sparse},
            "experts": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.expert},
        },
        "environment": "independent Python standard 2048; two initial tiles; 90% 2 / 10% 4",
        "search_at_inference": False,
        "games": rows,
        "summary": {"games": len(rows), "wins": wins, "wilson_95": wilson(wins, len(rows)), "mean_score": float(np.mean([row["score"] for row in rows])), "seconds": time.time() - started},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
