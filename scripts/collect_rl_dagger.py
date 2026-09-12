"""Collect direct-policy rollouts, labeled by an independently frozen RL teacher."""

import argparse
from collections import deque
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from bot2048.deep_rl import ConvAgent, GatedSymmetryHead, ResidualSymmetryHead, encode
from scripts.ppo_rl import BatchEnv
from scripts.train_conv_rl import RECORD


def policy_actions(agent, boards, legal, ensemble=False, aggregation="mean"):
    """Batch the same root-board symmetry averaging used by the deployed agent."""
    with torch.no_grad():
        if ensemble:
            variants, mappings = [], []
            for reflection in (False, True):
                for k in range(4):
                    variant = np.rot90(boards, k, axes=(1, 2))
                    mapping = (np.arange(4) - k) % 4
                    if reflection:
                        variant = variant[:, :, ::-1]
                        mapping = (-mapping) % 4
                    variants.append(variant.copy())
                    mappings.append(mapping)
            encoded = encode(np.concatenate(variants), agent.device)
            raw = torch.cat([model(encoded) for model in agent.models])
            raw = raw.reshape(len(agent.models) * 8, len(boards), 4)
            aligned = torch.stack(
                [
                    raw[model_index * 8 + i][:, mapping.tolist()]
                    for model_index in range(len(agent.models))
                    for i, mapping in enumerate(mappings)
                ]
            )
            valid = torch.tensor(legal, dtype=torch.bool, device=agent.device)
            if agent.head is not None:
                head_views = aligned.permute(1, 0, 2)
                if isinstance(agent.head, (ResidualSymmetryHead, GatedSymmetryHead)):
                    logits = agent.head(
                        head_views, encode(boards, agent.device), valid
                    )
                else:
                    logits = agent.head(
                        head_views, encode(boards, agent.device)
                    )
                if agent.head_min_confidence:
                    confidence = torch.softmax(logits.masked_fill(~valid, -1e9), 1).max(
                        1
                    ).values
                    masked = aligned.masked_fill(~valid[None], -1e9)
                    choices = masked.argmax(2)
                    votes = torch.stack(
                        [(choices == action).sum(0) for action in range(4)], 1
                    ).float()
                    votes += 0.99 * torch.softmax(masked, 2).mean(0)
                    logits = torch.where(
                        (confidence >= agent.head_min_confidence)[:, None], logits, votes
                    )
            elif aggregation == "mean":
                logits = aligned.mean(0)
            elif aggregation == "median":
                logits = aligned.median(0).values
            elif aggregation == "probability":
                logits = torch.softmax(aligned.masked_fill(~valid[None], -1e9), 2).mean(
                    0
                )
            elif aggregation in ("vote", "rank"):
                masked = aligned.masked_fill(~valid[None], -1e9)
                if aggregation == "vote":
                    choices = masked.argmax(2)
                    logits = torch.stack(
                        [(choices == action).sum(0) for action in range(4)], 1
                    ).float()
                    logits += 0.99 * torch.softmax(masked, 2).mean(0)
                else:
                    order = masked.argsort(2, descending=True)
                    ranks = order.argsort(2)
                    logits = (3 - ranks).float().mean(0)
            else:
                raise ValueError(f"unknown aggregation: {aggregation}")
        else:
            encoded = encode(boards, agent.device)
            logits = torch.stack([model(encoded) for model in agent.models]).mean(0)
        logits = logits.masked_fill(
            ~torch.tensor(legal, dtype=torch.bool, device=agent.device), -1e9
        )
        return logits.argmax(1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/rl/conv_dagger_source.pt")
    p.add_argument(
        "--teacher",
        nargs="+",
        default=["models/rl/dagger_teacher.bin"],
        help="One or more learned teachers; action scores are averaged",
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=64000000)
    p.add_argument("--device", default="mps")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument(
        "--aggregation",
        choices=["mean", "median", "probability", "vote", "rank"],
        default="mean",
    )
    p.add_argument("--output", default="data/rl_dagger_train.bin")
    p.add_argument(
        "--failure-replay",
        type=int,
        default=0,
        help="Duplicate this many states preceding each loss to emphasize recovery",
    )
    a = p.parse_args()
    torch.set_num_threads(2)
    agent = ConvAgent(a.checkpoint, a.device)
    env = BatchEnv(a.envs, a.seed)
    lib = ctypes.CDLL(
        str(
            Path(
                "native/td_value.dylib"
                if sys.platform == "darwin"
                else "native/td_value.so"
            ).resolve()
        )
    )
    lib.teacher_open.argtypes = [ctypes.c_char_p]
    lib.teacher_open.restype = ctypes.c_void_p
    lib.teacher_scores.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.teacher_scores.restype = None
    lib.teacher_close.argtypes = [ctypes.c_void_p]
    lib.teacher_close.restype = None
    teachers = [lib.teacher_open(name.encode()) for name in a.teacher]
    if not all(teachers):
        raise RuntimeError("Could not load every frozen RL teacher")
    started = time.time()
    wins = losses = 0
    replayed_states = 0
    recent = (
        [deque(maxlen=a.failure_replay) for _ in range(a.envs)]
        if a.failure_replay
        else None
    )
    with open(a.output, "wb") as output:
        for step in range(a.steps):
            packed = np.sum(
                env.boards.reshape(-1, 16).astype(np.uint64)
                << (4 * np.arange(16, dtype=np.uint64)),
                axis=1,
                dtype=np.uint64,
            )
            q = np.zeros((a.envs, 4), dtype=np.float32)
            temporary = np.empty_like(q)
            for teacher in teachers:
                lib.teacher_scores(
                    teacher, packed.ctypes.data, a.envs, temporary.ctypes.data
                )
                q += temporary
            q /= len(teachers)
            record = np.empty(a.envs, dtype=RECORD)
            record["board"] = packed
            record["q"] = q
            output.write(record.tobytes())
            if recent is not None:
                for index in range(a.envs):
                    recent[index].append(record[index].copy())
            actions = policy_actions(
                agent, env.boards, env.legal, a.ensemble, a.aggregation
            )
            env.step(actions)
            wins += int((env.outcomes == 1).sum())
            losses += int((env.outcomes == -1).sum())
            if recent is not None:
                for index in np.flatnonzero(env.outcomes):
                    if env.outcomes[index] == -1:
                        replay = np.asarray(recent[index], dtype=RECORD)
                        output.write(replay.tobytes())
                        replayed_states += len(replay)
                    recent[index].clear()
            if (step + 1) % 500 == 0:
                print(
                    json.dumps(
                        {
                            "steps": step + 1,
                            "states": (step + 1) * a.envs + replayed_states,
                            "wins": wins,
                            "losses": losses,
                            "seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    for teacher in teachers:
        lib.teacher_close(teacher)
    env.close()
    metadata = {
        "config": vars(a),
        "states": a.steps * a.envs + replayed_states,
        "base_states": a.steps * a.envs,
        "failure_replay_states": replayed_states,
        "wins": wins,
        "losses": losses,
        "policy_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "teacher_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in a.teacher
        },
        "seconds": time.time() - started,
        "labels": "learned RL afterstate values; no expectimax",
        "rollout_policy": "direct convolutional policy; no search",
    }
    Path(a.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
