"""Search-free sparse neural action policy. Only current-board features enter Q."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from .env import pack, legal_actions

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "native" / ("rl.dylib" if sys.platform == "darwin" else "rl.so")


def build():
    source = ROOT / "native/rl.cpp"
    header = ROOT / "native/rl_engine.h"
    if (
        not LIB.exists()
        or max(source.stat().st_mtime, header.stat().st_mtime) > LIB.stat().st_mtime
    ):
        temporary = LIB.with_name(LIB.name + f".{os.getpid()}.tmp")
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-DRL_LIBRARY",
                str(source),
                "-o",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(LIB)
    return LIB


class RLAgent:
    """A feed-forward sparse network plus a legal-action mask; no lookahead."""

    def __init__(self, checkpoint="models/rl/direct_q.bin"):
        self.lib = ctypes.CDLL(str(build()))
        self.lib.rl_open.argtypes = [ctypes.c_char_p]
        self.lib.rl_open.restype = ctypes.c_void_p
        self.lib.rl_close.argtypes = [ctypes.c_void_p]
        self.lib.rl_close.restype = None
        self.lib.rl_values.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.rl_values.restype = None
        self.lib.rl_values_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.rl_values_batch.restype = None
        self.handle = self.lib.rl_open(os.fsencode(checkpoint))
        if not self.handle:
            raise ValueError("Could not load RL checkpoint")

    def values(self, board):
        values = np.zeros(4, dtype=np.float32)
        self.lib.rl_values(
            self.handle,
            pack(board),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return values

    def values_batch(self, packed_boards):
        packed_boards = np.ascontiguousarray(packed_boards, dtype=np.uint64)
        values = np.empty((len(packed_boards), 4), dtype=np.float32)
        self.lib.rl_values_batch(
            self.handle,
            packed_boards.ctypes.data,
            len(packed_boards),
            values.ctypes.data,
        )
        return values

    def act(self, board):
        legal = legal_actions(board)
        if not legal:
            raise ValueError("No legal actions")
        q = self.values(board)
        return int(legal[np.argmax(q[legal])])

    def close(self):
        if self.handle:
            self.lib.rl_close(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()
