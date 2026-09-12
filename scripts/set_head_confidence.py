"""Set the neural-head confidence below which symmetry voting is retained."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum", required=True, type=float)
    args = parser.parse_args()
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = dict(artifact["metadata"])
    metadata["symmetry_head"] = dict(metadata["symmetry_head"])
    metadata["symmetry_head"]["minimum_confidence"] = args.minimum
    artifact["metadata"] = metadata
    torch.save(artifact, args.output)


if __name__ == "__main__":
    main()
