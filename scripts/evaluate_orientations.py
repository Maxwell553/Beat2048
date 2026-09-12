"""Evaluate each fixed symmetry view of a current-board neural policy."""

import argparse
import json

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent, encode
from scripts.ppo_rl import BatchEnv


def transformed_actions(agent, boards, legal, orientation):
    reflection, k = divmod(orientation, 4)
    transformed = np.rot90(boards, k, axes=(1, 2))
    mapping = (np.arange(4) - k) % 4
    if reflection:
        transformed = transformed[:, :, ::-1]
        mapping = (-mapping) % 4
    with torch.no_grad():
        raw = agent.model(encode(transformed.copy(), agent.device))
        aligned = raw[:, mapping.tolist()]
        aligned = aligned.masked_fill(
            ~torch.as_tensor(legal, dtype=torch.bool, device=agent.device), -1e9
        )
    return aligned.argmax(1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90000000)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    agent = ConvAgent(args.checkpoint, args.device)
    rows = []
    for orientation in range(8):
        env = BatchEnv(args.games, args.seed)
        finished = np.zeros(args.games, dtype=bool)
        wins = np.zeros(args.games, dtype=bool)
        while not finished.all():
            action = transformed_actions(
                agent, env.boards, env.legal, orientation
            )
            env.step(action)
            ended = ~finished & (env.outcomes != 0)
            wins[ended] = env.outcomes[ended] == 1
            finished[ended] = True
        env.close()
        row = {"orientation": orientation, "wins": int(wins.sum())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"checkpoint": args.checkpoint, "results": rows}))


if __name__ == "__main__":
    main()
