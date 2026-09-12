"""Relabel existing state datasets with a frozen weighted RL-teacher ensemble."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scripts.train_conv_rl import RECORD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument(
        "--teacher",
        nargs="+",
        default=["models/rl/dagger_teacher.bin"],
        help="One or more learned teachers; action scores are averaged",
    )
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument(
        "--wide-teacher",
        nargs="*",
        default=[],
        help="Optional checkpoints using the wider training-only tuple layout",
    )
    p.add_argument("--wide-weights", nargs="*", type=float)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.weights is not None and len(a.weights) != len(a.teacher):
        p.error("--weights must provide one value per --teacher checkpoint")
    if a.wide_weights is not None and len(a.wide_weights) != len(a.wide_teacher):
        p.error("--wide-weights must provide one value per --wide-teacher checkpoint")
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
        raise RuntimeError("Could not load every teacher")
    wide_lib = None
    wide_teachers = []
    if a.wide_teacher:
        wide_lib = ctypes.CDLL(
            str(
                Path(
                    "native/td_value_wide.dylib"
                    if sys.platform == "darwin"
                    else "native/td_value_wide.so"
                ).resolve()
            )
        )
        wide_lib.teacher_open.argtypes = [ctypes.c_char_p]
        wide_lib.teacher_open.restype = ctypes.c_void_p
        wide_lib.teacher_scores.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        wide_lib.teacher_scores.restype = None
        wide_lib.teacher_close.argtypes = [ctypes.c_void_p]
        wide_lib.teacher_close.restype = None
        wide_teachers = [wide_lib.teacher_open(name.encode()) for name in a.wide_teacher]
        if not all(wide_teachers):
            raise RuntimeError("Could not load every wide teacher")
    weights = a.weights or [1.0] * len(teachers)
    wide_weights = a.wide_weights or [1.0] * len(wide_teachers)
    total_weight = sum(weights) + sum(wide_weights)
    if total_weight <= 0:
        p.error("teacher weights must have a positive sum")
    states = 0
    with open(a.output, "wb") as output:
        for name in a.inputs:
            raw = np.fromfile(name, dtype=RECORD)
            boards = np.ascontiguousarray(raw["board"])
            q = np.zeros((len(raw), 4), dtype=np.float32)
            temporary = np.empty_like(q)
            for teacher, weight in zip(teachers, weights):
                lib.teacher_scores(
                    teacher, boards.ctypes.data, len(raw), temporary.ctypes.data
                )
                q += weight * temporary
            for teacher, weight in zip(wide_teachers, wide_weights):
                wide_lib.teacher_scores(
                    teacher, boards.ctypes.data, len(raw), temporary.ctypes.data
                )
                q += weight * temporary
            q /= total_weight
            assert np.array_equal(q > -1e20, raw["q"] > -1e20)
            raw["q"] = q
            output.write(raw.tobytes())
            states += len(raw)
    for teacher in teachers:
        lib.teacher_close(teacher)
    for teacher in wide_teachers:
        wide_lib.teacher_close(teacher)
    metadata = {
        "config": vars(a),
        "states": states,
        "teacher_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in [*a.teacher, *a.wide_teacher]
        },
        "inputs": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in a.inputs
        },
    }
    Path(a.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
