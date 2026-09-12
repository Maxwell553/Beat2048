"""Export the learned policy without its Python class; planner is separate."""

from pathlib import Path
import torch
import json
import hashlib
from bot2048.model import load_policy, encode
import numpy as np


def main():
    torch.set_num_threads(1)
    model, _ = load_policy("models/final_policy.pt")
    Path("models/final_bot.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "mode": "hybrid",
                "checkpoint": "final_policy.pt",
                "checkpoint_sha256": hashlib.sha256(
                    Path("models/final_policy.pt").read_bytes()
                ).hexdigest(),
                "target": 2048,
                "depth": 4,
                "cutoff": 0.0001,
                "neural_weight": 0.02,
                "symmetry_ensemble": True,
                "immediate_win_check": True,
                "architecture": "onehot256-128-64-4",
                "parameters": 41412,
                "description": "Small distilled neural policy with expectimax decision checking. No universal win guarantee.",
            },
            indent=2,
        )
    )
    example = encode(np.zeros((1, 4, 4), dtype=np.int16))
    traced = torch.jit.trace(model, example)
    traced.save("models/final_policy.torchscript")
    restored = torch.jit.load("models/final_policy.torchscript")
    batch = encode(np.random.default_rng(77).integers(0, 12, size=(8, 4, 4)))
    with torch.no_grad():
        torch.testing.assert_close(restored(batch), model(batch))
    print("Exported and verified models/final_policy.torchscript")


if __name__ == "__main__":
    main()
