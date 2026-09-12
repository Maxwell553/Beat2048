"""Evaluate a neural afterstate value model in standard 2048 games."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import encode, policy_from_config
from scripts.evaluate import wilson
from scripts.ppo_rl import BatchEnv
from scripts.train_afterstate_value import MoveBatch, make_afterstates, unpack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=130000000)
    p.add_argument("--device", default="mps")
    p.add_argument("--symmetry", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    artifact = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    model = policy_from_config(artifact["metadata"]["network_config"]).to(a.device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    scale = artifact["metadata"].get("value_scale", 1000.0)
    env = BatchEnv(a.games, a.seed)
    moves = MoveBatch()
    finished = np.zeros(a.games, dtype=bool)
    wins = np.zeros(a.games, dtype=bool)
    move_counts = np.zeros(a.games, dtype=np.int32)
    with torch.no_grad():
        while not finished.all():
            packed = np.sum(
                env.boards.reshape(-1, 16).astype(np.uint64)
                << (4 * np.arange(16, dtype=np.uint64)), axis=1, dtype=np.uint64
            )
            dummy = np.where(env.legal, 0.0, -1e30).astype(np.float32)
            after, rewards, legal = make_afterstates(moves, packed, dummy)
            boards = unpack(after.reshape(-1))
            if a.symmetry:
                variants = []
                for reflection in (False, True):
                    for rotation in range(4):
                        view = np.rot90(boards, rotation, axes=(1, 2))
                        if reflection:
                            view = view[:, :, ::-1]
                        variants.append(view.copy())
                values = model(encode(np.concatenate(variants), a.device))
                values = values.reshape(8, len(boards)).mean(0)
            else:
                values = model(encode(boards, a.device))
            scores = values.reshape(a.games, 4).cpu().numpy()
            scores += rewards / scale
            scores[~legal] = -1e30
            env.step(scores.argmax(1).astype(np.int32))
            active = ~finished
            move_counts[active] += 1
            ended = active & (env.outcomes != 0)
            wins[ended] = env.outcomes[ended] == 1
            finished[ended] = True
    env.close()
    count = int(wins.sum())
    result = {
        "config": vars(a),
        "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "environment": "standard 2048; exactly two initial tiles; 90% 2 / 10% 4",
        "search": False,
        "inference": "one deterministic move layer and one shared neural value batch; no tree or chance nodes",
        "summary": {
            "games": a.games,
            "wins": count,
            "wilson_95": wilson(count, a.games),
            "mean_moves": float(move_counts.mean()),
        },
    }
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
