"""Screen fixed reliability weights for the eight neural symmetry views."""

import argparse
import json

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent, encode
from scripts.ppo_rl import BatchEnv


def actions(agent, boards, legal, weights):
    variants, mappings = [], []
    for reflection in (False, True):
        for k in range(4):
            variant = np.rot90(boards, k, axes=(1, 2))
            mapping = (np.arange(4) - k) % 4
            if reflection:
                variant = variant[:, :, ::-1]
                mapping = (-mapping) % 4
            variants.append(variant.copy())
            mappings.append(mapping.tolist())
    with torch.no_grad():
        raw = agent.model(encode(np.concatenate(variants), agent.device)).reshape(
            8, len(boards), 4
        )
        aligned = torch.stack(
            [raw[index][:, mapping] for index, mapping in enumerate(mappings)]
        )
        mask = torch.as_tensor(legal, dtype=torch.bool, device=agent.device)
        masked = aligned.masked_fill(~mask[None], -1e9)
        choices = masked.argmax(2)
        weight = torch.as_tensor(weights, device=agent.device)[:, None]
        score = torch.stack(
            [((choices == action) * weight).sum(0) for action in range(4)], dim=1
        )
        score += 0.01 * torch.softmax(masked, 2).mean(0)
        score = score.masked_fill(~mask, -1e9)
    return score.argmax(1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", type=float, nargs=8, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90000000)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    agent = ConvAgent(args.checkpoint, args.device)
    env = BatchEnv(args.games, args.seed)
    finished = np.zeros(args.games, dtype=bool)
    wins = np.zeros(args.games, dtype=bool)
    while not finished.all():
        env.step(actions(agent, env.boards, env.legal, args.weights))
        ended = ~finished & (env.outcomes != 0)
        wins[ended] = env.outcomes[ended] == 1
        finished[ended] = True
    env.close()
    print(json.dumps({"weights": args.weights, "games": args.games, "wins": int(wins.sum())}))


if __name__ == "__main__":
    main()
