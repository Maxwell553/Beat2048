"""Quickly screen a direct policy on one standard native game per environment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent
from scripts.collect_rl_dagger import policy_actions
from scripts.evaluate import wilson
from scripts.ppo_rl import BatchEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--extra-checkpoint", action="append", default=[])
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=90000000)
    p.add_argument("--device", default="mps")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument(
        "--aggregation",
        choices=["mean", "median", "probability", "vote", "rank"],
        default="mean",
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    agent = ConvAgent(a.checkpoint, a.device, extra_checkpoints=a.extra_checkpoint)
    env = BatchEnv(a.games, a.seed)
    finished = np.zeros(a.games, dtype=bool)
    wins = np.zeros(a.games, dtype=bool)
    moves = np.zeros(a.games, dtype=np.int32)
    started = time.time()
    while not finished.all():
        actions = policy_actions(
            agent, env.boards, env.legal, a.ensemble, a.aggregation
        )
        env.step(actions)
        active = ~finished
        moves[active] += 1
        ended = active & (env.outcomes != 0)
        wins[ended] = env.outcomes[ended] == 1
        finished[ended] = True
        if moves.max() >= 20000:
            raise RuntimeError("At least one game reached the move cap")
    env.close()
    count = int(wins.sum())
    result = {
        "config": vars(a),
        "checkpoint_sha256": hashlib.sha256(
            Path(a.checkpoint).read_bytes()
        ).hexdigest(),
        "checkpoint_metadata": agent.metadata,
        "environment": "native standard 2048; two initial tiles; 90% 2 / 10% 4",
        "search_at_inference": False,
        "summary": {
            "games": a.games,
            "wins": count,
            "wilson_95": wilson(count, a.games),
            "mean_moves": float(moves.mean()),
            "max_moves": int(moves.max()),
            "seconds": time.time() - started,
        },
        "games": [
            {"seed": a.seed + i, "won": bool(wins[i]), "moves": int(moves[i])}
            for i in range(a.games)
        ],
    }
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
