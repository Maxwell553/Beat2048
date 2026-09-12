"""Search-free ensemble of reinforcement-learned sparse neural value models."""

import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .env import legal_actions, pack


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "native" / (
    "td_value.dylib" if sys.platform == "darwin" else "td_value.so"
)


def build():
    sources = [ROOT / "native/td_value.cpp", ROOT / "native/rl.cpp", ROOT / "native/rl_engine.h"]
    if not LIB.exists() or max(p.stat().st_mtime for p in sources) > LIB.stat().st_mtime:
        temporary = LIB.with_name(LIB.name + f".{os.getpid()}.tmp")
        subprocess.run(
            [
                "c++", "-O3", "-std=c++17", "-shared", "-fPIC",
                "-DVALUE_LIBRARY", str(sources[0]), "-o", str(temporary),
            ],
            check=True,
        )
        temporary.replace(LIB)
    return LIB


def materialize_checkpoint(path, compressed_sha256, uncompressed_sha256):
    """Verify and expand a GitHub-sized checkpoint into the system cache."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != compressed_sha256:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    if path.suffix != ".gz":
        return path
    cache = Path(tempfile.gettempdir()) / "beat2048-models"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{uncompressed_sha256}.bin"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == uncompressed_sha256:
            return destination
        destination.unlink()
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as source, temporary.open("wb") as output:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != uncompressed_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"expanded checkpoint hash mismatch: {path}")
    temporary.replace(destination)
    return destination


class ValueEnsembleAgent:
    """Score each legal deterministic afterstate once with learned neural values."""

    def __init__(self, manifest="models/rl/final_value_policy.json"):
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        self.manifest_path = manifest_path
        self.metadata = json.loads(manifest_path.read_text())
        if self.metadata.get("search") is not False:
            raise ValueError("value policy manifest must explicitly disable search")
        self.lib = ctypes.CDLL(str(build()))
        self.lib.teacher_open.argtypes = [ctypes.c_char_p]
        self.lib.teacher_open.restype = ctypes.c_void_p
        self.lib.teacher_scores.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]
        self.lib.teacher_scores.restype = None
        self.lib.teacher_close.argtypes = [ctypes.c_void_p]
        self.handles = []
        self.weights = []
        for entry in self.metadata["models"]:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = ROOT / path
            path = materialize_checkpoint(
                path, entry["sha256"], entry.get("uncompressed_sha256", entry["sha256"])
            )
            handle = self.lib.teacher_open(os.fsencode(path))
            if not handle:
                raise ValueError(f"could not load value checkpoint: {path}")
            self.handles.append(handle)
            self.weights.append(float(entry["weight"]))
        if not self.handles or sum(self.weights) <= 0:
            raise ValueError("value policy requires models with positive total weight")

    def values(self, board):
        packed = np.asarray([pack(board)], dtype=np.uint64)
        result = np.zeros(4, dtype=np.float32)
        temporary = np.empty((1, 4), dtype=np.float32)
        for handle, weight in zip(self.handles, self.weights):
            self.lib.teacher_scores(
                handle, packed.ctypes.data, 1, temporary.ctypes.data
            )
            result += weight * temporary[0]
        return result / sum(self.weights)

    def act(self, board):
        legal = legal_actions(board)
        if not legal:
            raise ValueError("No legal actions")
        values = self.values(board)
        return int(legal[np.argmax(values[legal])])

    def close(self):
        for handle in self.handles:
            self.lib.teacher_close(handle)
        self.handles = []

    def __del__(self):
        if getattr(self, "handles", None):
            self.close()
