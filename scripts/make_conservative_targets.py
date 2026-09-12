"""Preserve deployed actions except where a learned teacher finds costly regret."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent
from scripts.collect_rl_dagger import policy_actions
from scripts.train_conv_rl import RECORD, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-regret", type=float, default=1000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_num_threads(2)
    boards, teacher_q = read(a.input)
    raw = np.fromfile(a.input, dtype=RECORD)
    agent = ConvAgent(a.checkpoint, a.device)
    corrected = 0
    for offset in range(0, len(raw), a.batch_size):
        batch = boards[offset : offset + a.batch_size]
        q = teacher_q[offset : offset + a.batch_size]
        legal = q > -1e20
        actions = policy_actions(agent, batch, legal, True, "vote")
        regret = q.max(1) - q[np.arange(len(q)), actions]
        preserve = regret < a.min_regret
        ids = np.flatnonzero(preserve)
        q[ids] = np.where(legal[ids], 0, -1e30)
        q[ids, actions[ids]] = 1000
        raw["q"][offset : offset + len(batch)] = q
        corrected += int((~preserve).sum())
    raw.tofile(a.output)
    metadata = {
        "config": vars(a),
        "states": len(raw),
        "teacher_corrected_states": corrected,
        "preserved_policy_states": len(raw) - corrected,
        "corrected_fraction": corrected / len(raw),
        "policy_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        "method": "retain neural voting action unless learned-teacher regret crosses threshold",
    }
    Path(a.output).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
