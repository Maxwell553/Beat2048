"""Write weighted RL-teacher scores aligned with a fusion feature archive."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def open_library(wide=False):
    stem = "td_value_wide" if wide else "td_value"
    suffix = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(str(Path(f"native/{stem}.{suffix}").resolve()))
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
    return lib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--teacher", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--wide-teacher", nargs="*", default=[])
    p.add_argument("--wide-weights", nargs="*", type=float)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=65536)
    a = p.parse_args()
    if a.weights is not None and len(a.weights) != len(a.teacher):
        p.error("--weights must provide one value per --teacher checkpoint")
    if a.wide_weights is not None and len(a.wide_weights) != len(a.wide_teacher):
        p.error("--wide-weights must provide one value per --wide-teacher checkpoint")
    weights = a.weights or [1.0] * len(a.teacher)
    wide_weights = a.wide_weights or [1.0] * len(a.wide_teacher)
    total_weight = sum(weights) + sum(wide_weights)
    if total_weight <= 0:
        p.error("teacher weights must have a positive sum")

    normal_lib = open_library()
    normal = [normal_lib.teacher_open(name.encode()) for name in a.teacher]
    wide_lib = open_library(True) if a.wide_teacher else None
    wide = [wide_lib.teacher_open(name.encode()) for name in a.wide_teacher]
    if not all([*normal, *wide]):
        raise RuntimeError("Could not load every teacher checkpoint")

    with np.load(a.features, mmap_mode="r") as archive:
        boards = archive["boards"]
        old_targets = archive["targets"]
        output = np.lib.format.open_memmap(
            a.output, mode="w+", dtype=np.float32, shape=(len(boards), 4)
        )
        for offset in range(0, len(boards), a.batch_size):
            current = np.ascontiguousarray(boards[offset : offset + a.batch_size])
            packed = np.sum(
                current.reshape(-1, 16).astype(np.uint64)
                << (4 * np.arange(16, dtype=np.uint64)),
                axis=1,
                dtype=np.uint64,
            )
            scores = np.zeros((len(current), 4), dtype=np.float32)
            temporary = np.empty_like(scores)
            for handle, weight in zip(normal, weights):
                normal_lib.teacher_scores(
                    handle, packed.ctypes.data, len(current), temporary.ctypes.data
                )
                scores += weight * temporary
            for handle, weight in zip(wide, wide_weights):
                wide_lib.teacher_scores(
                    handle, packed.ctypes.data, len(current), temporary.ctypes.data
                )
                scores += weight * temporary
            scores /= total_weight
            if not np.array_equal(
                scores > -1e20, old_targets[offset : offset + len(current)] > -1e20
            ):
                raise RuntimeError("teacher legality differs from archived targets")
            output[offset : offset + len(current)] = scores
        output.flush()

    for handle in normal:
        normal_lib.teacher_close(handle)
    for handle in wide:
        wide_lib.teacher_close(handle)
    checkpoints = [*a.teacher, *a.wide_teacher]
    metadata = {
        "config": vars(a),
        "states": len(boards),
        "teacher_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in checkpoints
        },
    }
    Path(a.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
