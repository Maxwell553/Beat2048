"""Keep states where a deployed direct policy makes a costly teacher disagreement."""

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
    boards, q = read(a.input)
    agent = ConvAgent(a.checkpoint, a.device)
    selected = []
    regret_values = []
    for offset in range(0, len(boards), a.batch_size):
        batch = boards[offset : offset + a.batch_size]
        targets = q[offset : offset + a.batch_size]
        legal = targets > -1e20
        actions = policy_actions(agent, batch, legal, True, "vote")
        regret = targets.max(1) - targets[np.arange(len(targets)), actions]
        mask = regret >= a.min_regret
        selected.append(np.flatnonzero(mask) + offset)
        regret_values.append(regret[mask])
    selected = np.concatenate(selected)
    regrets = np.concatenate(regret_values)
    source = np.fromfile(a.input, dtype=RECORD)
    source[selected].tofile(a.output)
    metadata = {
        "config": vars(a),
        "source_states": len(source),
        "selected_states": len(selected),
        "selected_fraction": float(len(selected) / len(source)),
        "mean_selected_regret": float(regrets.mean()),
        "max_selected_regret": float(regrets.max()),
        "source_sha256": hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
    }
    Path(a.output).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
