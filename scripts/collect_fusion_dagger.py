"""Collect fusion-policy states and label them with frozen learned TD teachers."""

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

from bot2048.deep_rl import ConvAgent, FusionPolicy
from bot2048.rl import RLAgent
from scripts.evaluate_fusion_policy import fusion_actions
from scripts.ppo_rl import BatchEnv
from scripts.train_conv_rl import RECORD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn", required=True)
    p.add_argument("--sparse", nargs="+", required=True)
    p.add_argument("--fusion", nargs="+", required=True)
    p.add_argument("--expert", nargs="*", default=[])
    p.add_argument("--teacher", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=99000000)
    p.add_argument("--failure-replay", type=int, default=256)
    p.add_argument("--device", default="mps")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)

    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    fusion = []
    for path in args.fusion:
        artifact = torch.load(path, map_location=args.device, weights_only=True)
        model = FusionPolicy(
            artifact["metadata"]["inputs"], artifact["metadata"]["hidden"]
        ).to(args.device)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        fusion.append(model)
    experts = [ConvAgent(path, args.device) for path in args.expert]
    env = BatchEnv(args.envs, args.seed)

    library = Path("native/td_value.dylib" if sys.platform == "darwin" else "native/td_value.so").resolve()
    lib = ctypes.CDLL(str(library))
    lib.teacher_open.argtypes = [ctypes.c_char_p]
    lib.teacher_open.restype = ctypes.c_void_p
    lib.teacher_scores.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.teacher_scores.restype = None
    lib.teacher_close.argtypes = [ctypes.c_void_p]
    teachers = [lib.teacher_open(path.encode()) for path in args.teacher]
    if not all(teachers):
        raise RuntimeError("could not load every frozen TD teacher")

    recent = [deque(maxlen=args.failure_replay) for _ in range(args.envs)]
    replayed = wins = losses = 0
    started = time.time()
    with open(args.output, "wb") as output:
        for step in range(args.steps):
            packed = np.sum(
                env.boards.reshape(-1, 16).astype(np.uint64)
                << (4 * np.arange(16, dtype=np.uint64)),
                axis=1,
                dtype=np.uint64,
            )
            q = np.zeros((args.envs, 4), dtype=np.float32)
            temporary = np.empty_like(q)
            for teacher in teachers:
                lib.teacher_scores(teacher, packed.ctypes.data, args.envs, temporary.ctypes.data)
                q += temporary
            q /= len(teachers)
            record = np.empty(args.envs, dtype=RECORD)
            record["board"], record["q"] = packed, q
            output.write(record.tobytes())
            for index in range(args.envs):
                recent[index].append(record[index].copy())

            actions = fusion_actions(
                cnn, sparse, fusion, env.boards, env.legal, args.device,
                experts=experts,
            )
            env.step(actions)
            wins += int((env.outcomes == 1).sum())
            losses += int((env.outcomes == -1).sum())
            for index in np.flatnonzero(env.outcomes):
                if env.outcomes[index] == -1:
                    replay = np.asarray(recent[index], dtype=RECORD)
                    output.write(replay.tobytes())
                    replayed += len(replay)
                recent[index].clear()
            if (step + 1) % 500 == 0:
                print(json.dumps({"steps": step + 1, "states": (step + 1) * args.envs + replayed, "wins": wins, "losses": losses, "seconds": time.time() - started}), flush=True)

    for teacher in teachers:
        lib.teacher_close(teacher)
    for model in sparse:
        model.close()
    env.close()
    metadata = {
        "config": vars(args),
        "states": args.steps * args.envs + replayed,
        "base_states": args.steps * args.envs,
        "failure_replay_states": replayed,
        "wins": wins,
        "losses": losses,
        "hashes": {
            "cnn": hashlib.sha256(Path(args.cnn).read_bytes()).hexdigest(),
            "fusion": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.fusion},
            "experts": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.expert},
            "sparse": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.sparse},
            "teacher": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.teacher},
        },
        "labels": "learned TD afterstate action values; no expectimax",
        "rollout_policy": "current-board neural fusion; no search",
        "seconds": time.time() - started,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
