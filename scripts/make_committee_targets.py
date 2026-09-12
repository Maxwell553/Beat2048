"""Create base-preserving targets with sparse neural committee tie corrections."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent
from bot2048.rl import RLAgent
from scripts.collect_rl_dagger import policy_actions
from scripts.evaluate_neural_tiebreak import actions as committee_actions
from scripts.train_conv_rl import RECORD, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn", required=True)
    p.add_argument("--sparse", nargs="+", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    torch.set_num_threads(2)
    boards, source_q = read(args.input)
    cnn = ConvAgent(args.cnn, args.device)
    sparse = [RLAgent(path) for path in args.sparse]
    records = np.fromfile(args.input, dtype=RECORD)
    corrections = 0
    for offset in range(0, len(boards), args.batch_size):
        current = boards[offset : offset + args.batch_size]
        legal = source_q[offset : offset + len(current)] > -1e20
        base = policy_actions(cnn, current, legal, True, "vote")
        committee = committee_actions(cnn, sparse, current, legal)
        corrections += int((base != committee).sum())
        target = np.where(legal, 0, -1e30).astype(np.float32)
        target[np.arange(len(current)), committee] = 1
        records["q"][offset : offset + len(current)] = target
    for model in sparse:
        model.close()
    records.tofile(args.output)
    metadata = {
        "config": vars(args),
        "states": len(records),
        "committee_corrections": corrections,
        "correction_fraction": corrections / len(records),
        "method": "preserve CNN neural vote except direct sparse neural committee tie-breaks",
        "search_at_inference": False,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
