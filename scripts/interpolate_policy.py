"""Interpolate compatible policy checkpoints into one deployable neural network."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--fine-tuned", required=True)
    parser.add_argument("--fraction", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.fraction <= 1:
        parser.error("--fraction must be between 0 and 1")

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    tuned = torch.load(args.fine_tuned, map_location="cpu", weights_only=True)
    if base["metadata"].get("network_config") != tuned["metadata"].get(
        "network_config"
    ):
        raise ValueError("checkpoints use different network configurations")
    if base["state_dict"].keys() != tuned["state_dict"].keys():
        raise ValueError("checkpoints have different parameter sets")

    fraction = args.fraction
    state = {
        name: (1 - fraction) * tensor + fraction * tuned["state_dict"][name]
        for name, tensor in base["state_dict"].items()
    }
    metadata = dict(tuned["metadata"])
    metadata["weight_interpolation"] = {
        "base": args.base,
        "fine_tuned": args.fine_tuned,
        "fine_tuned_fraction": fraction,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "metadata": metadata}, output)


if __name__ == "__main__":
    main()
