"""Estimate action win rates on states preceding neural-policy losses.

The resulting targets come from complete stochastic game outcomes. Rollouts are
used only to make training labels; deployed inference remains one neural pass.
"""

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


def pack(boards):
    return np.sum(
        boards.reshape(-1, 16).astype(np.uint64)
        << (4 * np.arange(16, dtype=np.uint64)),
        axis=1,
        dtype=np.uint64,
    )


def collect_candidates(agent, envs, seed, losses_needed, history, device):
    env = BatchEnv(envs, seed)
    episodes = [deque(maxlen=history) for _ in range(envs)]
    candidates = []
    losses = wins = 0
    offsets = (24, 48, 72, 96, 128, 160, 192, 224)
    while losses < losses_needed:
        current = pack(env.boards)
        for i, board in enumerate(current):
            episodes[i].append(int(board))
        actions = policy_actions(agent, env.boards, env.legal, True, "vote")
        env.step(actions)
        for i in np.flatnonzero(env.outcomes):
            if env.outcomes[i] == -1:
                losses += 1
                episode = list(episodes[i])
                for offset in offsets:
                    if len(episode) >= offset:
                        candidates.append(episode[-offset])
            else:
                wins += 1
            episodes[i].clear()
    env.close()
    return np.asarray(list(dict.fromkeys(candidates)), dtype=np.uint64), wins, losses


def unpack(packed):
    return (
        (packed[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & 15
    ).astype(np.uint8).reshape(-1, 4, 4)


def evaluate_chunk(agent, states, rollouts, seed):
    count = len(states)
    repeated = np.repeat(states, 4 * rollouts)
    first_actions = np.tile(np.repeat(np.arange(4), rollouts), count).astype(np.int32)
    group_state = np.repeat(np.arange(count), 4 * rollouts)
    group_action = np.tile(np.repeat(np.arange(4), rollouts), count)
    env = BatchEnv(len(repeated), seed)
    env.set_boards(repeated)
    legal_initial = env.legal.copy()
    valid = legal_initial[np.arange(len(repeated)), first_actions].astype(bool)
    # Invalid copies are excluded; give them a legal action so the batch can run.
    first_actions[~valid] = legal_initial[~valid].argmax(1)
    env.step(first_actions)
    finished = ~valid
    wins = np.zeros(len(repeated), dtype=np.float32)
    ended = (~finished) & (env.outcomes != 0)
    wins[ended] = env.outcomes[ended] == 1
    finished |= ended
    steps = 1
    while not finished.all() and steps < 3000:
        actions = policy_actions(agent, env.boards, env.legal, True, "vote")
        env.step(actions)
        ended = (~finished) & (env.outcomes != 0)
        wins[ended] = env.outcomes[ended] == 1
        finished |= ended
        steps += 1
    env.close()
    q = np.full((count, 4), -1e30, dtype=np.float32)
    for state_index in range(count):
        for action in range(4):
            mask = valid & (group_state == state_index) & (group_action == action)
            if mask.any():
                q[state_index, action] = wins[mask].mean()
    return q, steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--losses", type=int, default=100)
    p.add_argument("--candidate-envs", type=int, default=128)
    p.add_argument("--history", type=int, default=256)
    p.add_argument("--max-candidates", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=87000000)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    torch.set_num_threads(2)
    agent = ConvAgent(args.checkpoint, args.device)
    candidates, wins, losses = collect_candidates(
        agent, args.candidate_envs, args.seed, args.losses, args.history, args.device
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: args.max_candidates]
    targets = []
    max_steps = 0
    for offset in range(0, len(candidates), args.chunk_size):
        chunk = candidates[offset : offset + args.chunk_size]
        q, steps = evaluate_chunk(
            agent, chunk, args.rollouts, args.seed + 100000 + offset
        )
        targets.append(q)
        max_steps = max(max_steps, steps)
        print(json.dumps({"evaluated": offset + len(chunk), "candidates": len(candidates)}), flush=True)
    q = np.concatenate(targets)
    records = np.empty(len(candidates), dtype=RECORD)
    records["board"] = candidates
    records["q"] = q
    records.tofile(args.output)
    boards = unpack(candidates)
    legal = q > -1e20
    base = policy_actions(agent, boards, legal, True, "vote")
    best = q.argmax(1)
    base_rate = q[np.arange(len(q)), base]
    best_rate = q[np.arange(len(q)), best]
    metadata = {
        "config": vars(args),
        "candidate_collection_wins": wins,
        "candidate_collection_losses": losses,
        "states": len(candidates),
        "rollouts_per_action": args.rollouts,
        "best_action_differs": int((best != base).sum()),
        "mean_base_win_rate": float(base_rate.mean()),
        "mean_best_win_rate": float(best_rate.mean()),
        "mean_estimated_improvement": float((best_rate - base_rate).mean()),
        "max_rollout_steps": max_steps,
        "labels": "complete stochastic 2048 outcomes under the direct neural continuation policy",
        "search_at_inference": False,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
