"""Attach a zero-initialized trainable correction branch to a frozen policy."""

import argparse
import hashlib
from pathlib import Path

import torch

from bot2048.deep_rl import ConvPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    data = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(data["metadata"].get("network_config", {}))
    if config.get("adapter_hidden", 0):
        raise ValueError("source checkpoint already has an adapter")
    config["adapter_hidden"] = args.hidden
    model = ConvPolicy(**config)
    missing, unexpected = model.load_state_dict(data["state_dict"], strict=False)
    if unexpected or not missing or any(not name.startswith("adapter.") for name in missing):
        raise ValueError(f"unexpected checkpoint mismatch: {missing=}, {unexpected=}")

    source = ConvPolicy(**{**config, "adapter_hidden": 0})
    source.load_state_dict(data["state_dict"])
    sample = torch.randn(32, 16, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(model(sample), source(sample), rtol=0, atol=0)

    metadata = dict(data["metadata"])
    metadata.update(
        {
            "network_config": model.config,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "adapter": {"hidden": args.hidden, "initialization": "exact zero residual"},
            "source_sha256": hashlib.sha256(
                Path(args.checkpoint).read_bytes()
            ).hexdigest(),
        }
    )
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, args.output)


if __name__ == "__main__":
    main()
