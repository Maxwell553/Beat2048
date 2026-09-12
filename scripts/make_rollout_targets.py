"""Convert noisy rollout win rates into conservative base-preserving targets."""

import argparse
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
    p.add_argument("--minimum-improvement", type=float, default=0.5)
    p.add_argument("--minimum-best-rate", type=float, default=0.5)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    torch.set_num_threads(2)
    boards, rates = read(args.input)
    legal = rates > -1e20
    base = policy_actions(
        ConvAgent(args.checkpoint, args.device), boards, legal, True, "vote"
    )
    best = rates.argmax(1)
    base_rate = rates[np.arange(len(rates)), base]
    best_rate = rates[np.arange(len(rates)), best]
    improve = (
        (best != base)
        & (best_rate - base_rate >= args.minimum_improvement)
        & (best_rate >= args.minimum_best_rate)
    )
    labels = np.where(improve, best, base)
    records = np.fromfile(args.input, dtype=RECORD)
    records["q"] = np.where(legal, 0, -1e30)
    records["q"][np.arange(len(records)), labels] = 1
    records.tofile(args.output)
    metadata = {
        "config": vars(args),
        "states": len(records),
        "corrections": int(improve.sum()),
        "correction_fraction": float(improve.mean()),
        "mean_selected_improvement": float((best_rate - base_rate)[improve].mean())
        if improve.any()
        else 0,
        "method": "preserve the neural vote unless complete-game rollouts show a clear win-rate gain",
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
