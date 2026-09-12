"""Evaluate averaged snapshots of the training-only learned afterstate teacher."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from scripts.evaluate import wilson
from scripts.ppo_rl import BatchEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--wide-checkpoints", nargs="*", default=[])
    p.add_argument("--wide-weights", nargs="*", type=float)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=67200000)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.weights is not None and len(a.weights) != len(a.checkpoints):
        p.error("--weights must provide one value per checkpoint")
    if a.wide_weights is not None and len(a.wide_weights) != len(a.wide_checkpoints):
        p.error("--wide-weights must provide one value per wide checkpoint")
    library = Path(
        "native/td_value.dylib" if sys.platform == "darwin" else "native/td_value.so"
    )
    lib = ctypes.CDLL(str(library.resolve()))
    lib.teacher_open.argtypes = [ctypes.c_char_p]
    lib.teacher_open.restype = ctypes.c_void_p
    lib.teacher_scores.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.teacher_close.argtypes = [ctypes.c_void_p]
    handles = [lib.teacher_open(name.encode()) for name in a.checkpoints]
    if not all(handles):
        raise RuntimeError("Could not load every teacher checkpoint")
    wide_lib = None; wide_handles = []
    if a.wide_checkpoints:
        wide_library = Path(
            "native/td_value_wide.dylib" if sys.platform == "darwin"
            else "native/td_value_wide.so"
        )
        wide_lib = ctypes.CDLL(str(wide_library.resolve()))
        wide_lib.teacher_open.argtypes = [ctypes.c_char_p]
        wide_lib.teacher_open.restype = ctypes.c_void_p
        wide_lib.teacher_scores.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        wide_lib.teacher_close.argtypes = [ctypes.c_void_p]
        wide_handles = [wide_lib.teacher_open(name.encode()) for name in a.wide_checkpoints]
        if not all(wide_handles):
            raise RuntimeError("Could not load every wide teacher checkpoint")
    env = BatchEnv(a.games, a.seed)
    finished = np.zeros(a.games, dtype=bool)
    wins = np.zeros(a.games, dtype=bool)
    moves = np.zeros(a.games, dtype=np.int32)
    while not finished.all():
        packed = np.sum(
            env.boards.reshape(-1, 16).astype(np.uint64)
            << (4 * np.arange(16, dtype=np.uint64)),
            axis=1,
            dtype=np.uint64,
        )
        scores = np.zeros((a.games, 4), dtype=np.float32)
        temporary = np.empty_like(scores)
        weights = a.weights or [1.0] * len(handles)
        for handle, weight in zip(handles, weights):
            lib.teacher_scores(
                handle, packed.ctypes.data, a.games, temporary.ctypes.data
            )
            scores += weight * temporary
        wide_weights = a.wide_weights or [1.0] * len(wide_handles)
        for handle, weight in zip(wide_handles, wide_weights):
            wide_lib.teacher_scores(handle, packed.ctypes.data, a.games, temporary.ctypes.data)
            scores += weight * temporary
        scores /= sum(weights) + sum(wide_weights)
        scores[~env.legal.astype(bool)] = -1e30
        env.step(scores.argmax(1).astype(np.int32))
        active = ~finished
        moves[active] += 1
        ended = active & (env.outcomes != 0)
        wins[ended] = env.outcomes[ended] == 1
        finished[ended] = True
    env.close()
    for handle in handles:
        lib.teacher_close(handle)
    for handle in wide_handles:
        wide_lib.teacher_close(handle)
    count = int(wins.sum())
    result = {
        "config": vars(a),
        "checkpoint_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in [*a.checkpoints, *a.wide_checkpoints]
        },
        "role": "training-only learned afterstate teacher; not the deployed policy",
        "search": False,
        "summary": {
            "games": a.games,
            "wins": count,
            "wilson_95": wilson(count, a.games),
            "mean_moves": float(moves.mean()),
        },
    }
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
