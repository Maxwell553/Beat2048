"""Collect complete winning episodes from a direct neural policy for self-imitation."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent
from scripts.collect_rl_dagger import policy_actions
from scripts.ppo_rl import BatchEnv
from scripts.train_conv_rl import RECORD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=85000000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    agent = ConvAgent(args.checkpoint, args.device)
    env = BatchEnv(args.envs, args.seed)
    episodes = [deque() for _ in range(args.envs)]
    wins = losses = states = 0
    with open(args.output, "wb") as output:
        for step in range(args.steps):
            packed = np.sum(
                env.boards.reshape(-1, 16).astype(np.uint64)
                << (4 * np.arange(16, dtype=np.uint64)),
                axis=1,
                dtype=np.uint64,
            )
            actions = policy_actions(agent, env.boards, env.legal, True, "vote")
            for index in range(args.envs):
                record = np.zeros(1, dtype=RECORD)
                record["board"][0] = packed[index]
                record["q"][0] = np.where(env.legal[index], 0.0, -1e30)
                record["q"][0, actions[index]] = 1.0
                episodes[index].append(record[0].copy())
            env.step(actions)
            for index in np.flatnonzero(env.outcomes != 0):
                if env.outcomes[index] == 1:
                    episode = np.asarray(episodes[index], dtype=RECORD)
                    output.write(episode.tobytes())
                    states += len(episode)
                    wins += 1
                else:
                    losses += 1
                episodes[index].clear()
            if (step + 1) % 1000 == 0:
                print(json.dumps({"steps": step + 1, "wins": wins, "losses": losses, "states": states}), flush=True)
    env.close()
    metadata = {
        "config": vars(args),
        "wins": wins,
        "losses": losses,
        "states": states,
        "labels": "actions from complete winning episodes; no teacher or search",
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
