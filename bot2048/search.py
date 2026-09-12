"""Native expectimax; full spawn enumeration with explicit probability pruning."""

import ctypes
from pathlib import Path
import subprocess
import os
import sys
import numpy as np
from .env import pack

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = (
    ROOT / "native" / ("search.dylib" if sys.platform == "darwin" else "search.so")
)


def build():
    source = ROOT / "native/search.cpp"
    if not LIBRARY.exists() or source.stat().st_mtime > LIBRARY.stat().st_mtime:
        temporary = LIBRARY.with_name(LIBRARY.name + f".{os.getpid()}.tmp")
        try:
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(temporary),
                ],
                check=True,
            )
            temporary.replace(LIBRARY)
        finally:
            temporary.unlink(missing_ok=True)
    return LIBRARY


class Planner:
    def __init__(self, depth=3, cutoff=0.0001):
        if depth < 1 or not 0 < cutoff <= 1:
            raise ValueError("depth >= 1 and cutoff in (0,1] required")
        self.depth, self.cutoff = depth, cutoff
        self.lib = ctypes.CDLL(str(build()))
        self.lib.search_scores.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.search_scores.restype = None
        self.lib.native_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.lib.native_move.restype = ctypes.c_uint64
        self.nodes = 0

    def scores(self, board):
        out = np.empty(4, dtype=np.float64)
        nodes = ctypes.c_uint64()
        self.lib.search_scores(
            pack(board),
            self.depth,
            self.cutoff,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(nodes),
        )
        self.nodes = nodes.value
        return out

    def act(self, board):
        scores = self.scores(board)
        if np.max(scores) < -1e90:
            raise ValueError("no legal actions")
        return int(np.argmax(scores))
