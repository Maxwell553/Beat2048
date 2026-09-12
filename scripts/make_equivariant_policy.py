"""Wrap existing convolutional weights in differentiable D4 soft voting."""

import argparse
from pathlib import Path

import torch

from bot2048.deep_rl import ConvPolicy, EquivariantConvPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, default=0.05)
    args = p.parse_args()
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(artifact["metadata"].get("network_config", {}))
    if config.get("kind", "conv") != "conv":
        raise ValueError("source must be an ordinary convolutional policy")
    config.pop("kind", None)
    source = ConvPolicy(**config)
    source.load_state_dict(artifact["state_dict"])
    model = EquivariantConvPolicy(**config, vote_temperature=args.temperature)
    model.load_state_dict(artifact["state_dict"])
    metadata = {
        **artifact["metadata"],
        "network_config": model.config,
        "symmetry_aggregation": "differentiable low-temperature probability vote",
        "search_at_inference": False,
        "source_checkpoint": args.checkpoint,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, args.output)


if __name__ == "__main__":
    main()
