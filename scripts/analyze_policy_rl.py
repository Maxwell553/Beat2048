"""Measure teacher-value regret and serious mistakes, not just label accuracy."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bot2048.deep_rl import ConvAgent, encode
from scripts.train_conv_rl import ensemble_logits, read


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/rl/conv_dagger_source.pt")
    p.add_argument("--data", default="data/rl_teacher_validation.bin")
    p.add_argument("--output", default="results/rl/policy_regret.json")
    p.add_argument("--ensemble", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    agent = ConvAgent(a.checkpoint, "mps")
    boards, q = read(a.data)
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(boards), 2048):
            batch = boards[offset : offset + 2048]
            logits = (
                (
                    ensemble_logits(agent.model, batch, "mps")
                    if a.ensemble
                    else agent.model(encode(batch, "mps"))
                )
                .cpu()
                .numpy()
            )
            logits[q[offset : offset + 2048] < -1e20] = -1e30
            predictions.extend(logits.argmax(1))
    predictions = np.asarray(predictions)
    regret = q.max(1) - q[np.arange(len(q)), predictions]
    ranks = boards.max(axis=(1, 2))

    def metrics(mask):
        r = regret[mask]
        return {
            "states": int(mask.sum()),
            "teacher_agreement": float(np.mean(predictions[mask] == q[mask].argmax(1))),
            "mean_regret": float(r.mean()),
            "regret_over_1000": float(np.mean(r > 1000)),
            "regret_over_5000": float(np.mean(r > 5000)),
            "regret_over_10000": float(np.mean(r > 10000)),
        }

    result = {
        "config": vars(a),
        "overall": metrics(np.ones(len(q), dtype=bool)),
        "by_largest_tile": {
            str(2 ** int(rank)): metrics(ranks == rank) for rank in np.unique(ranks)
        },
    }
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
